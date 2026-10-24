from typing import AsyncGenerator
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from app.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False}  # Safe for SQLite + Async
)


@event.listens_for(engine.sync_engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_connection, _record):
    """SQLite ignores ON DELETE CASCADE unless foreign keys are enabled per connection."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    class_=AsyncSession
)

class Base(DeclarativeBase):
    pass

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI Dependency yielding an isolated database session per request."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# Columns added after the v1.0 schema. create_all() never alters existing tables,
# so pre-existing SQLite databases are upgraded in place with ALTER TABLE.
_ADDED_COLUMNS = {
    "model_artifacts": {
        "n_features": "INTEGER",
        "feature_names": "JSON",
        "task_type": "VARCHAR(32) NOT NULL DEFAULT 'unknown'",
        "class_labels": "JSON",
        "file_format": "VARCHAR(16) NOT NULL DEFAULT 'joblib'",
        "background_dataset_id": "INTEGER REFERENCES datasets(id) ON DELETE SET NULL",
    },
    "datasets": {
        "purpose": "VARCHAR(16) NOT NULL DEFAULT 'evaluation'",
    },
}

# Data fixes that run once, right after the column they depend on is added
_BACKFILLS = {
    ("datasets", "purpose"): (
        "UPDATE datasets SET purpose = 'background' "
        "WHERE id IN (SELECT background_dataset_id FROM model_artifacts WHERE background_dataset_id IS NOT NULL)"
    ),
}


async def _apply_lightweight_migrations(conn: AsyncConnection) -> None:
    for table, columns in _ADDED_COLUMNS.items():
        rows = (await conn.execute(text(f"PRAGMA table_info({table})"))).fetchall()
        existing = {row[1] for row in rows}
        for column, ddl in columns.items():
            if column not in existing:
                await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
                backfill = _BACKFILLS.get((table, column))
                if backfill:
                    await conn.execute(text(backfill))


async def init_models() -> None:
    """Creates missing tables and upgrades older SQLite schemas."""
    import app.models  # noqa: F401  (registers all mappers on Base.metadata)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _apply_lightweight_migrations(conn)
