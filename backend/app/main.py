import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.v1.router import api_router
from app.database import engine, init_models
from app.core.maintenance import backfill_model_metadata

logger = logging.getLogger("uvicorn.error")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure tables exist (and older schemas are upgraded) and directories are initialized
    settings.init_storage()
    await init_models()
    await backfill_model_metadata()
    if not settings.llm_is_local:
        logger.warning(
            "NFR-2 deviation: prompts are sent to a non-local Ollama host (%s).", settings.OLLAMA_BASE_URL
        )
    yield
    # Shutdown
    await engine.dispose()

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Backend API for local SHAP + Gemma predictive model explainability.",
    version="1.1.0",
    lifespan=lifespan
)

# CORS only matters when the UI is not served through the Vite /api proxy
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API v1 routes
app.include_router(api_router)

@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "llm_model": settings.OLLAMA_MODEL,
        "llm_local": settings.llm_is_local,
    }
