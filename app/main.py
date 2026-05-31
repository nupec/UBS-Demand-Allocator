import logging
import os
from fastapi import FastAPI
from app.config import settings
from app.routes import router as api_router
from fastapi.middleware.cors import CORSMiddleware
from app.routes import batch

logger = logging.getLogger(__name__)

app = FastAPI(
  title=settings.APP_TITLE,
  description=settings.APP_DESCRIPTION,
  version=settings.APP_VERSION,
  root_path="/api/allocator" 
)

origins = [
    origin.strip()
    for origin in os.getenv("BACKEND_CORS_ORIGINS", "http://localhost,http://127.0.0.1").split(",")
    if origin.strip()
]
allow_credentials = "*" not in origins

# Middleware do CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=allow_credentials,
    allow_methods=["*"],   
    allow_headers=["*"],    
)

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(batch.router)
logger.info("Starting FastAPI application with title: %s", settings.APP_TITLE)
app.include_router(api_router, prefix="/allocator")
logger.info("Router has been included. Application setup complete.")

