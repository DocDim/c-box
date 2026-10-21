from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # App Settings
    PROJECT_NAME: str = "SHAP-LLM Explainability Platform"
    DEBUG: bool = True
    
    # Filesystem Sandboxing
    STORAGE_DIR: Path = BASE_DIR / "app" / "storage"
    MODELS_DIR: Path = BASE_DIR / "app" / "storage" / "models"
    DATASETS_DIR: Path = BASE_DIR / "app" / "storage" / "datasets"
    
    # Ollama Local Service
    OLLAMA_BASE_URL: str = "http://51.254.124.101:8383"
    OLLAMA_MODEL: str = "gemma3:2b"
    
    # Authentication & Security
    SECRET_KEY: str = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours
    
    # Database Configuration (SQLite via aiosqlite)
    DATABASE_URL: str = "sqlite+aiosqlite:///./xai_app.db"
    
    # Allowed File Extensions
    ALLOWED_MODEL_EXTENSIONS: set = {".joblib", ".pkl"}
    ALLOWED_DATA_EXTENSIONS: set = {".csv"}

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def init_storage(self) -> None:
        """Ensures sandboxed storage scratch directories exist."""
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.DATASETS_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.init_storage()