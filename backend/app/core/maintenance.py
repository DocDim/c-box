import logging
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select

from app.database import AsyncSessionLocal
from app.models.artifact import ModelArtifact, TaskType
from app.core.model_loader import model_loader

logger = logging.getLogger("uvicorn.error")


async def backfill_model_metadata() -> None:
    """
    Models uploaded before introspection results were persisted have no task type or
    feature names. Re-inspect them once (in the sandboxed subprocess) at startup.
    """
    async with AsyncSessionLocal() as session:
        pending = (await session.execute(
            select(ModelArtifact).where(ModelArtifact.task_type == TaskType.UNKNOWN.value)
        )).scalars().all()
        for model in pending:
            try:
                info = await run_in_threadpool(model_loader.inspect_in_subprocess, model.file_path, model.file_format)
            except HTTPException as e:
                logger.warning("Could not re-inspect model %s: %s", model.id, e.detail)
                continue
            model.task_type = info["task_type"]
            model.n_features = info["n_features"]
            model.feature_names = info["feature_names"]
            model.class_labels = info["class_labels"]
            model.estimator_type = info["estimator_type"]
        if pending:
            await session.commit()
