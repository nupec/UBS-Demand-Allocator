import os
import time
import json
import logging
import pandas as pd
from datetime import datetime
from unidecode import unidecode

from app.preprocessing.common import prepare_data
from app.methods.knn_model import allocate_demands_knn

logger = logging.getLogger(__name__)

JOBS_STORE_PATH = os.getenv(
    "BATCH_JOBS_STORE",
    os.path.join(os.getcwd(), "data", "results", "batch_jobs.json")
)

def _load_batch_jobs():
    try:
        with open(JOBS_STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        logger.exception("Could not load batch job store from %s", JOBS_STORE_PATH)
        return {}

def _persist_batch_jobs():
    store_dir = os.path.dirname(JOBS_STORE_PATH)
    if store_dir:
        os.makedirs(store_dir, exist_ok=True)
    tmp_path = f"{JOBS_STORE_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(BATCH_JOBS, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, JOBS_STORE_PATH)

def set_batch_job(job_id: str, **updates):
    job_info = BATCH_JOBS.setdefault(job_id, {})
    job_info.update(updates)
    _persist_batch_jobs()
    return job_info

# Mantem consulta imediata em memoria e persiste o estado para sobreviver a restart simples.
BATCH_JOBS = _load_batch_jobs()

class BatchProcessorService:
    def __init__(self, demands_dir: str, opps_file_path: str):
        self.demands_dir = demands_dir
        self.opps_file_path = opps_file_path

    def _limpar_coordenadas_invalidas(self, gdf):
        """
        Sanitização: Remove coordenadas zeradas ou fora do Bounding Box aproximado do Brasil.
        Lat: 5.2 (Norte) a -33.7 (Sul) | Lon: -34.7 (Leste) a -73.9 (Oeste)
        Impede que o Valhalla tente calcular rotas intercontinentais (Erro Betim).
        """
        original_count = len(gdf)
        
        # Filtro de Bounding Box do Brasil
        gdf_clean = gdf.cx[-74.0:-34.0, -34.0:6.0]
        
        removed = original_count - len(gdf_clean)
        if removed > 0:
            logger.warning(f"Sanitização: Removidos {removed} pontos com coordenadas inválidas/fora do Brasil.")
            
        return gdf_clean

    def process_batch_async(self, job_id: str, input_csv_path: str, method: str, k: int = 1):
        """Função executada em Background Task."""
        start_time = time.time()
        set_batch_job(job_id, status="processing", progress="Iniciando leitura do CSV...")

        resultados_finais = []
        estatisticas = {
            "sucesso": 0,
            "falhas": 0,
            "fallbacks": 0,
            "erros": []
        }

        try:
            df_input = pd.read_csv(input_csv_path)
            df_input.columns = [unidecode(c.strip().upper()) for c in df_input.columns]
            total_cities = len(df_input)

            if 'UF' not in df_input.columns or 'MUNICIPIO' not in df_input.columns:
                raise ValueError("O CSV deve conter as colunas 'UF' e 'MUNICIPIO'.")

            for index, row in df_input.iterrows():
                uf = str(row['UF']).strip().upper()
                municipio = str(row['MUNICIPIO']).strip()
                
                # Atualiza o status em tempo real para o cliente consumir
                set_batch_job(job_id, progress=f"Processando [{index+1}/{total_cities}]: {municipio}/{uf}")
                logger.info(BATCH_JOBS[job_id]["progress"])

                municipio_norm = unidecode(municipio).upper().replace(" ", "_").replace("'", "")
                municipio_acentuado = municipio.upper().replace(" ", "_").replace("'", "")

                # Busca o GeoJSON
                demand_path = None
                for fname in [f"{municipio_norm}.geojson", f"{municipio_acentuado}.geojson"]:
                    path_candidate = os.path.join(self.demands_dir, uf, fname)
                    if os.path.exists(path_candidate):
                        demand_path = path_candidate
                        break

                if not demand_path:
                    estatisticas["erros"].append(f"{municipio}: GeoJSON não encontrado")
                    estatisticas["falhas"] += 1
                    continue

                try:
                    class MockFile:
                        def __init__(self, file_obj):
                            self.file = file_obj

                    with open(demand_path, "rb") as demands_f, open(self.opps_file_path, "rb") as opps_f:
                        error, demands_gdf, opps_gdf, col_did, col_name, col_city, col_state_opp, _ = prepare_data(
                            opportunities_file=MockFile(opps_f),
                            demands_file=MockFile(demands_f),
                            state=uf,
                            city=municipio
                        )

                    if error or demands_gdf.empty or opps_gdf.empty:
                        raise ValueError("Dados geográficos vazios ou corrompidos.")

                    # Sanitiza dados antes do Valhalla
                    demands_gdf = self._limpar_coordenadas_invalidas(demands_gdf)
                    
                    if demands_gdf.empty:
                        raise ValueError("Nenhuma coordenada válida restou após a sanitização.")

                    # Executa o modelo
                    df_result = allocate_demands_knn(
                        demands_gdf=demands_gdf,
                        opportunities_gdf=opps_gdf,
                        col_demand_id=col_did,
                        col_name=col_name,
                        col_city=col_city,
                        col_state=col_state_opp,
                        k=k,
                        method=method,
                        city_name=municipio
                    )

                    if 'fallback_used' in df_result.columns and df_result['fallback_used'].any():
                        estatisticas["fallbacks"] += 1
                        df_result = df_result.drop(columns=['fallback_used'])

                    df_result['Process_UF'] = uf
                    df_result['Process_Municipio'] = municipio
                    
                    resultados_finais.append(df_result)
                    estatisticas["sucesso"] += 1

                except Exception as e:
                    erro_msg = f"{municipio}: {str(e)}"
                    logger.error(f"Falha em {municipio}: {e}")
                    estatisticas["erros"].append(erro_msg)
                    estatisticas["falhas"] += 1

            # Finalização e Salvação
            set_batch_job(job_id, progress="Consolidando arquivos finais...")
            
            if resultados_finais:
                df_final = pd.concat(resultados_finais, ignore_index=True)
                output_dir = os.path.join(os.getcwd(), "data", "results")
                os.makedirs(output_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                final_filename = f"lote_{job_id}_{timestamp}.csv"
                final_path = os.path.join(output_dir, final_filename)
                
                df_final.to_csv(final_path, index=False)
                set_batch_job(job_id, result_file=final_path)
            else:
                set_batch_job(job_id, result_file=None)

            duracao = round(time.time() - start_time, 2)
            set_batch_job(
                job_id,
                status="completed",
                stats=estatisticas,
                progress=f"Finalizado em {duracao} segundos."
            )
            logger.info(f"Job {job_id} concluído com sucesso.")

        except Exception as global_e:
            logger.exception(f"Erro fatal no Job {job_id}")
            set_batch_job(job_id, status="failed", progress=f"Erro fatal: {str(global_e)}")
        finally:
            try:
                if os.path.exists(input_csv_path):
                    os.remove(input_csv_path)
            except OSError:
                logger.warning("Could not remove temporary batch input file: %s", input_csv_path)
