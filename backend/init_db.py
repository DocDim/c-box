import asyncio
from app.database import engine, Base
from app.models import User, ModelArtifact, ModelShare

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Database schema successfully initialized.")

if __name__ == "__main__":
    asyncio.run(init_db())
