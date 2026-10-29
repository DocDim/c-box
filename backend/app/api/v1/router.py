from fastapi import APIRouter
from app.api.v1.endpoints import auth, models, datasets, predict, explain, audit

api_router = APIRouter(prefix="/api/v1")

# Mount sub-routers (routes follow SDD v1.0.0 section 3)
api_router.include_router(auth.router)
api_router.include_router(models.router)
api_router.include_router(datasets.router)
api_router.include_router(predict.router)
api_router.include_router(explain.router)
api_router.include_router(audit.router)
