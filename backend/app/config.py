from pathlib import Path
from typing import List
from urllib.parse import urlparse

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

# Placeholder values that must never be used to sign real tokens
_INSECURE_SECRET_KEYS = {
    "",
    "generate-a-secure-random-secret-key-here",
    "changeme",
    "secret",
}

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


class Settings(BaseSettings):
    # App Settings
    PROJECT_NAME: str = "SHAP-LLM Explainability Platform"
    DEBUG: bool = False
    BACKEND_HOST: str = "127.0.0.1"
    BACKEND_PORT: int = 8001
    # The Vite dev server proxies /api, so CORS is only needed when the UI is served elsewhere
    CORS_ORIGINS: List[str] = ["http://localhost:3001", "http://127.0.0.1:3001"]

    # Filesystem Sandboxing (models/ and datasets/ are derived from STORAGE_DIR)
    STORAGE_DIR: Path = BASE_DIR / "app" / "storage"

    # Ollama Local Service (NFR-2: zero egress, bound to localhost by default)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "gemma:2b"
    OLLAMA_TIMEOUT_SECONDS: float = 120.0
    # Explicit opt-in to send prompts to a non-loopback Ollama host (documented NFR-2 deviation)
    ALLOW_REMOTE_LLM: bool = False
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 768
    LLM_TOP_K_FEATURES: int = 5

    # Authentication & Security (SECRET_KEY has no default: it must come from the environment / .env)
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours
    BCRYPT_ROUNDS: int = 12

    # Database Configuration (SQLite via aiosqlite)
    DATABASE_URL: str = "sqlite+aiosqlite:///./xai_app.db"

    # Upload limits & formats (FR-2.1)
    MAX_MODEL_UPLOAD_MB: int = 250
    MAX_DATASET_UPLOAD_MB: int = 100
    ALLOWED_MODEL_EXTENSIONS: set = {".joblib", ".pkl", ".skops"}
    PICKLE_MODEL_EXTENSIONS: set = {".joblib", ".pkl"}
    ALLOWED_DATA_EXTENSIONS: set = {".csv"}
    # Pickle-based formats execute code on load. Disable to accept only .skops artifacts.
    ALLOW_PICKLE_MODELS: bool = True
    MODEL_INSPECTION_TIMEOUT_SECONDS: float = 60.0

    # SHAP Engine
    SHAP_BACKGROUND_MAX_ROWS: int = 100
    SHAP_KMEANS_K: int = 30
    SHAP_GLOBAL_MAX_SAMPLES: int = 300
    SHAP_ADDITIVITY_TOLERANCE: float = 1e-3
    EXPLAINER_CACHE_SIZE: int = 8

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @field_validator("SECRET_KEY")
    @classmethod
    def _reject_placeholder_secret(cls, value: str) -> str:
        if value.strip().lower() in _INSECURE_SECRET_KEYS or len(value) < 32:
            raise ValueError(
                "SECRET_KEY is missing, a placeholder, or shorter than 32 characters. "
                "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
            )
        return value

    @model_validator(mode="after")
    def _enforce_zero_egress(self) -> "Settings":
        if not self.ALLOW_REMOTE_LLM and not self.llm_is_local:
            raise ValueError(
                f"OLLAMA_BASE_URL '{self.OLLAMA_BASE_URL}' is not a loopback address. "
                "NFR-2 (zero egress) requires a local Ollama daemon. Set ALLOW_REMOTE_LLM=true "
                "to explicitly accept sending prompts off this workstation."
            )
        return self

    @property
    def llm_is_local(self) -> bool:
        return (urlparse(self.OLLAMA_BASE_URL).hostname or "") in _LOOPBACK_HOSTS

    @property
    def MODELS_DIR(self) -> Path:
        return self.STORAGE_DIR / "models"

    @property
    def DATASETS_DIR(self) -> Path:
        return self.STORAGE_DIR / "datasets"

    @property
    def max_model_bytes(self) -> int:
        return self.MAX_MODEL_UPLOAD_MB * 1024 * 1024

    @property
    def max_dataset_bytes(self) -> int:
        return self.MAX_DATASET_UPLOAD_MB * 1024 * 1024

    def allowed_model_extensions(self) -> set:
        if self.ALLOW_PICKLE_MODELS:
            return set(self.ALLOWED_MODEL_EXTENSIONS)
        return set(self.ALLOWED_MODEL_EXTENSIONS) - set(self.PICKLE_MODEL_EXTENSIONS)

    def init_storage(self) -> None:
        """Ensures sandboxed storage scratch directories exist."""
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.DATASETS_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.init_storage()
