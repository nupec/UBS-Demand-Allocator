import os
import shutil
import uuid
from fastapi import APIRouter, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from app.services.batch_service import BatchProcessorService, BATCH_JOBS, set_batch_job

router = APIRouter(prefix="/v1/batch", tags=["Batch Processing"])

# Configurações de diretório estáticas (ajuste conforme seu servidor)
DEMANDS_DIR = os.path.join(os.getcwd(), "data", "geojson_por_estado_cidade")
OPPS_FILE = os.path.join(os.getcwd(), "data", "opportunities.geojson")
UPLOADS_DIR = os.path.join(os.getcwd(), "data", "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

ALLOWED_METHODS = {"valhalla", "pysal", "pandana_real_distance"}

@router.post("/start")
async def start_batch_processing(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    method: str = Form(default="valhalla", description="Métodos: valhalla, pysal, pandana_real_distance"),
    k: int = Form(default=1)
):
    """
    Inicia o processamento em lote de forma assíncrona.
    Recebe um CSV com as cidades e retorna um Job ID imediatamente.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="O arquivo deve ser um .csv")
    if method not in ALLOWED_METHODS:
        raise HTTPException(status_code=400, detail=f"Método inválido. Use: {', '.join(sorted(ALLOWED_METHODS))}.")
    if k < 1:
        raise HTTPException(status_code=400, detail="O parâmetro k deve ser maior ou igual a 1.")

    job_id = str(uuid.uuid4())
    
    # Salva o arquivo temporariamente
    input_csv_path = os.path.join(UPLOADS_DIR, f"input_{job_id}.csv")
    with open(input_csv_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Inicializa o status na memória
    set_batch_job(
        job_id,
        status="queued",
        progress="Aguardando processamento...",
        result_file=None,
        stats={}
    )

    # Instancia o serviço
    processor = BatchProcessorService(demands_dir=DEMANDS_DIR, opps_file_path=OPPS_FILE)

    # Delega a tarefa pesada para o Background (Não trava o HTTP)
    background_tasks.add_task(processor.process_batch_async, job_id, input_csv_path, method, k)

    return {
        "message": "Processamento em lote iniciado com sucesso.",
        "job_id": job_id,
        "method": method,
        "status_url": f"/api/v1/batch/status/{job_id}"
    }

@router.get("/status/{job_id}")
async def get_batch_status(job_id: str):
    """Consulta o status atual de um Job."""
    job_info = BATCH_JOBS.get(job_id)
    if not job_info:
        raise HTTPException(status_code=404, detail="Job ID não encontrado.")

    response = {
        "job_id": job_id,
        "status": job_info["status"],
        "progress": job_info["progress"]
    }

    if job_info["status"] == "completed":
        response["stats"] = job_info.get("stats", {})
        response["download_url"] = f"/api/v1/batch/download/{job_id}"
    elif job_info["status"] == "failed":
        response["errors"] = job_info.get("stats", {}).get("erros", [])
        
    return response

@router.get("/download/{job_id}")
async def download_batch_result(job_id: str):
    """Faz o download do arquivo CSV consolidado resultante do Job."""
    job_info = BATCH_JOBS.get(job_id)
    
    if not job_info:
        raise HTTPException(status_code=404, detail="Job ID não encontrado.")
    
    if job_info["status"] != "completed":
        raise HTTPException(status_code=400, detail="Processamento ainda não concluído.")
        
    result_path = job_info.get("result_file")
    
    if not result_path or not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Arquivo de resultado não encontrado no servidor.")

    filename = os.path.basename(result_path)
    return FileResponse(path=result_path, filename=filename, media_type='text/csv')
