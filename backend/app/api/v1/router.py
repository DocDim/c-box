from fastapi import APIRouter
from app.api.v1.endpoints import auth, upload, model_access, predict, explain, chat

api_router = APIRouter(prefix="/api/v1")

# Mount sub-routers
api_router.include_router(auth.router)
api_router.include_router(upload.router)
api_router.include_router(model_access.router)
api_router.include_router(predict.router)
api_router.include_router(explain.router)
api_router.include_router(chat.router)