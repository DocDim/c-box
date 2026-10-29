import asyncio
from app.database import init_models

async def init_db():
    await init_models()
    print("Database schema successfully initialized.")

if __name__ == "__main__":
    asyncio.run(init_db())
