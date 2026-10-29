import json
import os
from contextlib import ExitStack
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Dict, List

# Isolate tests from the developer's .env, database and storage BEFORE importing the app
_TMP_ROOT = Path(tempfile.mkdtemp(prefix="cbox_tests_"))
os.environ.update({
    "SECRET_KEY": "test-secret-key-" + "x" * 40,
    "DATABASE_URL": f"sqlite+aiosqlite:///{(_TMP_ROOT / 'test.db').as_posix()}",
    "STORAGE_DIR": str(_TMP_ROOT / "storage"),
    "OLLAMA_BASE_URL": "http://localhost:11434",
    "ALLOW_REMOTE_LLM": "false",
    "DEBUG": "false",
    "BCRYPT_ROUNDS": "4",  # fast hashing in tests
})

import httpx  # noqa: E402
import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from sklearn.datasets import load_iris  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from app.config import settings  # noqa: E402
from app.core.explainer_cache import explainer_cache  # noqa: E402
from app.core.llm_service import llm_service  # noqa: E402
from app.database import Base, engine, init_models  # noqa: E402
from app.main import app  # noqa: E402


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TMP_ROOT, ignore_errors=True)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session")
def iris_frame() -> pd.DataFrame:
    iris = load_iris(as_frame=True)
    df = iris.data.copy()
    df["species"] = iris.target
    return df


@pytest.fixture(scope="session")
def artifacts_dir() -> Path:
    path = _TMP_ROOT / "fixtures"
    path.mkdir(exist_ok=True)
    return path


@pytest.fixture(scope="session")
def iris_rf_path(iris_frame, artifacts_dir) -> Path:
    X, y = iris_frame.drop(columns="species"), iris_frame["species"]
    path = artifacts_dir / "iris_rf.joblib"
    joblib.dump(RandomForestClassifier(n_estimators=25, random_state=0).fit(X, y), path)
    return path


@pytest.fixture(scope="session")
def iris_logreg_path(iris_frame, artifacts_dir) -> Path:
    X, y = iris_frame.drop(columns="species"), iris_frame["species"]
    path = artifacts_dir / "iris_logreg.joblib"
    joblib.dump(LogisticRegression(max_iter=1000).fit(X, y), path)
    return path


@pytest.fixture(scope="session")
def iris_csv_path(iris_frame, artifacts_dir) -> Path:
    path = artifacts_dir / "iris_eval.csv"
    iris_frame.to_csv(path, index=False)
    return path


class FakeOllama:
    """Records requests and replies with a canned NDJSON token stream."""

    def __init__(self):
        self.requests: List[Dict] = []
        self.reply: Callable[[Dict], str] = lambda body: "## Executive Summary\nGrounded answer."
        self.status_code = 200

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        if self.status_code != 200:
            return httpx.Response(self.status_code, text="model not found")
        text = self.reply(body)
        words = text.split(" ")
        lines = [
            json.dumps({"message": {"role": "assistant", "content": w + (" " if i < len(words) - 1 else "")},
                        "done": False})
            for i, w in enumerate(words)
        ]
        lines.append(json.dumps({"message": {"role": "assistant", "content": ""}, "done": True}))
        return httpx.Response(200, text="\n".join(lines) + "\n")


@pytest.fixture
def fake_ollama():
    fake = FakeOllama()
    previous = llm_service.transport
    llm_service.transport = httpx.MockTransport(fake.handler)
    yield fake
    llm_service.transport = previous


@pytest.fixture
async def client():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await init_models()
    explainer_cache.clear()
    shutil.rmtree(settings.STORAGE_DIR, ignore_errors=True)
    settings.init_storage()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    # aiosqlite connections run on non-daemon threads; release them inside this event loop
    await engine.dispose()


async def register_and_login(client: httpx.AsyncClient, email: str, password: str = "password123") -> Dict[str, str]:
    r = await client.post("/api/v1/auth/register", json={"email": email, "password": password})
    assert r.status_code == 201, r.text
    r = await client.post("/api/v1/auth/login", data={"username": email, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


async def upload_model(client, headers, path: Path, name="Iris RF", domain="Botany",
                       background: Path | None = None) -> Dict:
    with ExitStack() as stack:
        files = {"file": (path.name, stack.enter_context(open(path, "rb")), "application/octet-stream")}
        if background is not None:
            files["background_file"] = (background.name, stack.enter_context(open(background, "rb")), "text/csv")
        r = await client.post(
            "/api/v1/models/upload",
            data={"name": name, "domain_context": domain},
            files=files,
            headers=headers,
        )
    assert r.status_code == 201, r.text
    return r.json()


async def upload_dataset(client, headers, path: Path, model_id=None, target=None, expect=201) -> Dict:
    data = {}
    if model_id is not None:
        data["model_id"] = str(model_id)
    if target is not None:
        data["target_column"] = target
    with open(path, "rb") as fh:
        r = await client.post(
            "/api/v1/datasets/upload", data=data,
            files={"file": (path.name, fh, "text/csv")}, headers=headers,
        )
    assert r.status_code == expect, r.text
    return r.json()
