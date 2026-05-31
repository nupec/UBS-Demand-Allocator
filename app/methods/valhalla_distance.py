import logging
import os
import requests
import pandas as pd
import numpy as np
from fastapi import HTTPException

logger = logging.getLogger(__name__)

# URL do serviço Valhalla
# Padrão para o IP do Gateway Docker que funcionou nos testes
VALHALLA_API_URL = os.getenv("VALHALLA_URL", "http://172.17.0.1:8002")

# Limite de segurança do Valhalla (Origens x Destinos) para evitar erro 400.
# Deixamos uma pequena margem em relação ao limite padrão de pares da matriz.
VALHALLA_MATRIX_LIMIT = int(os.getenv("VALHALLA_MATRIX_LIMIT", "2400"))

# Deve acompanhar service_limits.<costing>.max_matrix_distance no valhalla.json.
VALHALLA_MAX_MATRIX_DISTANCE_METERS = int(os.getenv("VALHALLA_MAX_MATRIX_DISTANCE_METERS", "5000000"))

def get_valhalla_matrix(
    demands_gdf, 
    opportunities_gdf, 
    col_demand_id, 
    col_name, 
    costing="auto", 
    units="km"
):
    """
    Calcula a matriz de distâncias usando a API 'sources_to_targets' do Valhalla via POST.
    - Implementa loteamento dinâmico automático para evitar limites de capacidade.
    - Exibe o progresso detalhado de processamento por cidade.
    """
    endpoint = f"{VALHALLA_API_URL}/sources_to_targets"
    
    # 1. Preparar Origens (Sources) - Garantindo floats nativos para o JSON
    sources = []
    for idx, row in demands_gdf.iterrows():
        sources.append({
            "lat": float(row.geometry.y),
            "lon": float(row.geometry.x),
            "id": str(row[col_demand_id])
        })

    # 2. Preparar Destinos (Targets)
    targets = []
    for idx, row in opportunities_gdf.iterrows():
        targets.append({
            "lat": float(row.geometry.y),
            "lon": float(row.geometry.x),
            "id": str(row[col_name])
        })

    total_sources = len(sources)
    total_targets = len(targets)
    
    if total_sources == 0 or total_targets == 0:
        logger.warning("GeoDataFrames vazios fornecidos ao Valhalla.")
        return pd.DataFrame()

    # --- CÁLCULO DO LOTE DINÂMICO ---
    # Evita o erro 'Exceeded max locations' calculando quantas origens 
    # cabem em cada requisição baseando-se no número de destinos da cidade.
    if total_targets > VALHALLA_MATRIX_LIMIT:
        logger.error(f"Número de destinos ({total_targets}) excede o limite do Valhalla ({VALHALLA_MATRIX_LIMIT}).")
        # Fallback de segurança para lote unitário
        batch_size = 1
    else:
        calculated_batch_size = int(VALHALLA_MATRIX_LIMIT / total_targets)
        # Limita entre 1 e 200 para manter as requisições em tamanho saudável
        batch_size = max(1, min(calculated_batch_size, 200))
    
    logger.info(f"Configuração Valhalla: {total_sources} Origens x {total_targets} Destinos.")
    logger.info(f"⚡ Lote Dinâmico Calculado: {batch_size} origens por requisição.")

    # 3. Inicializar a matriz numpy com NaN
    matrix_values = np.full((total_sources, total_targets), np.nan)
    failed_batches = 0
    failed_max_distance_batches = 0
    
    # 4. Processamento em Lotes (Batching) das ORIGENS
    for i in range(0, total_sources, batch_size):
        batch_sources = sources[i : i + batch_size]
        
        payload = {
            "sources": batch_sources,
            "targets": targets,
            "costing": costing,
            "units": units,
            "costing_options": {
                costing: {
                    "max_distance": VALHALLA_MAX_MATRIX_DISTANCE_METERS
                }
            }
        }

        try:
            # Timeout de 60s por lote
            response = requests.post(endpoint, json=payload, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                batch_results = data.get("sources_to_targets", [])

                for local_src_idx, target_list in enumerate(batch_results):
                    global_src_idx = i + local_src_idx
                    
                    for item in target_list:
                        target_idx = item["to_index"]
                        distance = item["distance"]
                        matrix_values[global_src_idx, target_idx] = distance
                
                # Exibe o progresso solicitado
                progresso_atual = min(i + batch_size, total_sources)
                logger.info(f"Progresso: {progresso_atual}/{total_sources} origens processadas.")

            else:
                # Loga o erro mas continua o loop (as distâncias deste lote permanecerão NaN)
                # Se todos os lotes falharem, o fallback geodésico no knn_model será ativado.
                failed_batches += 1
                if "max distance limit" in response.text.lower():
                    failed_max_distance_batches += 1
                logger.warning(f"⚠️ Lote {i} falhou (Status {response.status_code}): {response.text}")

        except Exception as e:
            failed_batches += 1
            logger.error(f"❌ Erro de conexão/timeout no lote Valhalla {i}: {e}")

    # 5. Converter Numpy Array para Pandas DataFrame
    demands_ids = [s["id"] for s in sources]
    opps_names = [t["id"] for t in targets]

    df_matrix = pd.DataFrame(matrix_values, index=demands_ids, columns=opps_names)
    valid_distances = int(np.isfinite(matrix_values).sum())
    if failed_batches:
        logger.warning(
            "Valhalla finalizado com %d lote(s) falhos e %d distância(s) válida(s).",
            failed_batches,
            valid_distances,
        )
    if valid_distances == 0 and failed_max_distance_batches:
        logger.error(
            "Todos os lotes falharam por limite de distância da matriz. "
            "Verifique service_limits.%s.max_matrix_distance no valhalla.json "
            "(valor esperado para este backend: %d metros).",
            costing,
            VALHALLA_MAX_MATRIX_DISTANCE_METERS,
        )

    return df_matrix
