import csv
import io
from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.config import Settings, settings
from app.core.storage import ensure_within_storage, sanitize_filename
from tests.conftest import register_and_login, upload_dataset, upload_model

pytestmark = pytest.mark.anyio

API = "/api/v1"


# --------------------------------------------------------------------------- auth (AC-1)

async def test_register_login_and_me(client):
    headers = await register_and_login(client, "alice@example.com")
    r = await client.get(f"{API}/auth/me", headers=headers)
    assert r.status_code == 200 and r.json()["email"] == "alice@example.com"


async def test_login_rejects_wrong_password(client):
    await register_and_login(client, "alice@example.com")
    r = await client.post(f"{API}/auth/login", data={"username": "alice@example.com", "password": "wrongpass1"})
    assert r.status_code == 401


async def test_short_password_rejected(client):
    r = await client.post(f"{API}/auth/register", json={"email": "a@example.com", "password": "short"})
    assert r.status_code == 422


async def test_protected_endpoint_requires_token(client):
    assert (await client.get(f"{API}/models/")).status_code == 401


# --------------------------------------------------------------------------- ingestion (AC-2)

async def test_model_upload_records_introspection(client, iris_rf_path):
    headers = await register_and_login(client, "alice@example.com")
    model = await upload_model(client, headers, iris_rf_path)
    assert model["estimator_type"] == "RandomForestClassifier"
    assert model["task_type"] == "classification"
    assert model["n_features"] == 4
    assert model["feature_names"][0] == "sepal length (cm)"
    assert model["class_labels"] == ["0", "1", "2"]
    stored = list((settings.MODELS_DIR / "1").iterdir())
    assert len(stored) == 1 and stored[0].name.endswith("_iris_rf.joblib")


async def test_h5_and_unknown_extensions_rejected(client, artifacts_dir):
    headers = await register_and_login(client, "alice@example.com")
    for name in ("model.h5", "model.exe"):
        path = artifacts_dir / name
        path.write_bytes(b"not a model")
        with open(path, "rb") as fh:
            r = await client.post(f"{API}/models/upload", data={"name": "x", "domain_context": "y"},
                                  files={"file": (name, fh)}, headers=headers)
        assert r.status_code == 400


async def test_corrupt_model_rejected_and_file_removed(client, artifacts_dir):
    headers = await register_and_login(client, "alice@example.com")
    path = artifacts_dir / "corrupt.joblib"
    path.write_bytes(b"definitely not a pickle")
    with open(path, "rb") as fh:
        r = await client.post(f"{API}/models/upload", data={"name": "x", "domain_context": "y"},
                              files={"file": (path.name, fh)}, headers=headers)
    assert r.status_code == 422
    assert not any((settings.MODELS_DIR / "1").glob("*"))


async def test_upload_size_limit(client, iris_rf_path, monkeypatch):
    headers = await register_and_login(client, "alice@example.com")
    monkeypatch.setattr(settings, "MAX_MODEL_UPLOAD_MB", 0)
    with open(iris_rf_path, "rb") as fh:
        r = await client.post(f"{API}/models/upload", data={"name": "x", "domain_context": "y"},
                              files={"file": (iris_rf_path.name, fh)}, headers=headers)
    assert r.status_code == 413


async def test_pickle_formats_can_be_disabled(client, iris_rf_path, monkeypatch):
    headers = await register_and_login(client, "alice@example.com")
    monkeypatch.setattr(settings, "ALLOW_PICKLE_MODELS", False)
    with open(iris_rf_path, "rb") as fh:
        r = await client.post(f"{API}/models/upload", data={"name": "x", "domain_context": "y"},
                              files={"file": (iris_rf_path.name, fh)}, headers=headers)
    assert r.status_code == 400


async def test_skops_model_upload(client, iris_frame, artifacts_dir):
    skops_io = pytest.importorskip("skops.io")
    from sklearn.ensemble import RandomForestClassifier
    X, y = iris_frame.drop(columns="species"), iris_frame["species"]
    path = artifacts_dir / "iris_rf.skops"
    skops_io.dump(RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y), path)
    headers = await register_and_login(client, "alice@example.com")
    model = await upload_model(client, headers, path)
    assert model["file_format"] == "skops" and model["n_features"] == 4


async def test_dataset_upload_is_evaluation_data_only(client, iris_rf_path, iris_csv_path):
    headers = await register_and_login(client, "alice@example.com")
    model = await upload_model(client, headers, iris_rf_path)
    ds = await upload_dataset(client, headers, iris_csv_path, model_id=model["id"])
    assert ds["target_column"] == "species"
    assert ds["n_rows"] == 150 and ds["purpose"] == "evaluation" and ds["is_background"] is False
    assert ds["summary"]["sepal length (cm)"]["min"] == pytest.approx(4.3)
    assert len(ds["preview_rows"]) == 5

    # Uploading data for prediction never touches the model's background
    r = await client.get(f"{API}/models/{model['id']}", headers=headers)
    assert r.json()["background_dataset_id"] is None


async def test_dataset_missing_features_rejected(client, iris_rf_path, iris_frame, artifacts_dir):
    headers = await register_and_login(client, "alice@example.com")
    model = await upload_model(client, headers, iris_rf_path)
    path = artifacts_dir / "missing.csv"
    iris_frame.drop(columns=["petal width (cm)"]).to_csv(path, index=False)
    await upload_dataset(client, headers, path, model_id=model["id"], expect=422)


async def test_dataset_rows_pagination(client, iris_rf_path, iris_csv_path):
    headers = await register_and_login(client, "alice@example.com")
    model = await upload_model(client, headers, iris_rf_path)
    ds = await upload_dataset(client, headers, iris_csv_path, model_id=model["id"])
    r = await client.get(f"{API}/datasets/{ds['id']}/rows", params={"offset": 140, "limit": 25}, headers=headers)
    body = r.json()
    assert r.status_code == 200 and body["total_rows"] == 150 and len(body["rows"]) == 10


# --------------------------------------------------------------------------- explain + chat (AC-4, AC-5)

async def _owner_with_model(client, iris_rf_path, iris_csv_path):
    """Owner registers the model with its background, then loads a separate evaluation dataset."""
    headers = await register_and_login(client, "alice@example.com")
    model = await upload_model(client, headers, iris_rf_path, background=iris_csv_path)
    ds = await upload_dataset(client, headers, iris_csv_path, model_id=model["id"])
    return headers, model, ds


async def test_explain_dataset_row_is_additive_and_fast(client, iris_rf_path, iris_csv_path):
    headers, model, ds = await _owner_with_model(client, iris_rf_path, iris_csv_path)
    r = await client.post(f"{API}/explain/instance",
                          json={"model_id": model["id"], "dataset_id": ds["id"], "row_index": 0}, headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()
    total = body["base_value"] + sum(c["shap_value"] for c in body["feature_contributions"])
    assert total == pytest.approx(body["explained_output"], abs=1e-6)
    assert body["output_units"] == "probability"
    assert body["explained_class"] == body["predicted_class"] == "0"
    assert "species" not in body["input_features"]
    assert body["shap_ms"] < 2500  # NFR-3 / AC-4


async def test_manual_inputs_are_reordered_and_coerced(client, iris_rf_path, iris_csv_path):
    headers, model, _ = await _owner_with_model(client, iris_rf_path, iris_csv_path)
    features = {"petal width (cm)": "0.2", "petal length (cm)": 1.4,
                "sepal width (cm)": 3.5, "sepal length (cm)": 5.1, "unused": 1}
    r = await client.post(f"{API}/explain/instance", json={"model_id": model["id"], "instance_features": features},
                          headers=headers)
    assert r.status_code == 200, r.text
    assert list(r.json()["input_features"]) == model["feature_names"]

    r = await client.post(f"{API}/explain/instance",
                          json={"model_id": model["id"], "instance_features": {"sepal length (cm)": 5}},
                          headers=headers)
    assert r.status_code == 422


async def test_predict_matches_explain(client, iris_rf_path, iris_csv_path):
    headers, model, ds = await _owner_with_model(client, iris_rf_path, iris_csv_path)
    src = {"model_id": model["id"], "dataset_id": ds["id"], "row_index": 77}
    pred = (await client.post(f"{API}/predict/", json=src, headers=headers)).json()
    expl = (await client.post(f"{API}/explain/instance", json=src, headers=headers)).json()
    assert pred["predicted_class"] == expl["predicted_class"]
    assert pred["prediction"] == pytest.approx(expl["prediction"])
    assert set(pred["probabilities"]) == {"0", "1", "2"}


async def test_kernel_model_requires_background(client, iris_logreg_path, iris_csv_path):
    headers = await register_and_login(client, "alice@example.com")
    model = await upload_model(client, headers, iris_logreg_path, name="LR")
    features = {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2}
    r = await client.post(f"{API}/explain/instance", json={"model_id": model["id"], "instance_features": features},
                          headers=headers)
    assert r.status_code == 422 and "reference data" in r.json()["detail"]

    # Evaluation data does not become the background...
    await upload_dataset(client, headers, iris_csv_path, model_id=model["id"])
    r = await client.post(f"{API}/explain/instance", json={"model_id": model["id"], "instance_features": features},
                          headers=headers)
    assert r.status_code == 422

    # ...only the owner's background upload does
    with open(iris_csv_path, "rb") as fh:
        r = await client.post(f"{API}/models/{model['id']}/background", headers=headers,
                              files={"file": (iris_csv_path.name, fh, "text/csv")})
    assert r.status_code == 200, r.text
    r = await client.post(f"{API}/explain/instance", json={"model_id": model["id"], "instance_features": features},
                          headers=headers)
    assert r.status_code == 200, r.text
    assert r.json()["explainer_type"].startswith("KernelExplainer")
    assert any(abs(c["shap_value"]) > 1e-3 for c in r.json()["feature_contributions"])


async def test_global_explanation(client, iris_rf_path, iris_csv_path):
    headers, model, ds = await _owner_with_model(client, iris_rf_path, iris_csv_path)
    # No dataset_id: computed on the model's background data
    r = await client.get(f"{API}/explain/global/{model['id']}", params={"max_samples": 50}, headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["samples_evaluated"] == 50
    assert len(body["beeswarm"]["shap_values"]) == 4


async def test_chat_is_grounded_on_stored_explanation(client, fake_ollama, iris_rf_path, iris_csv_path):
    headers, model, ds = await _owner_with_model(client, iris_rf_path, iris_csv_path)
    expl = (await client.post(f"{API}/explain/instance",
                              json={"model_id": model["id"], "dataset_id": ds["id"], "row_index": 0},
                              headers=headers)).json()

    fake_ollama.reply = lambda body: (
        "## Executive Summary\nProbability 100%.\n## Upward Pressures\nPetal length drove it.\n"
        "## Mitigating Factors\nNone.\n## Actionable Next Step\nNo action; the invented 0.777 is suspicious."
    )
    r = await client.post(f"{API}/explain/chat", json={"explanation_id": expl["explanation_id"]}, headers=headers)
    assert r.status_code == 200, r.text
    message = r.json()["message"]
    assert message["kind"] == "narrative"
    assert message["faithfulness"]["structure_complete"] is True
    assert message["faithfulness"]["unverified_numbers"] == ["0.777"]

    # Context sent to the LLM comes from the server-side record, including the domain
    sent = fake_ollama.requests[-1]["messages"]
    assert "Botany" in sent[0]["content"]
    assert "petal" in sent[1]["content"]

    # Re-requesting the narrative returns the stored one without calling the LLM again
    calls = len(fake_ollama.requests)
    await client.post(f"{API}/explain/chat", json={"explanation_id": expl["explanation_id"]}, headers=headers)
    assert len(fake_ollama.requests) == calls


async def test_followups_carry_conversation_history(client, fake_ollama, iris_rf_path, iris_csv_path):
    headers, model, ds = await _owner_with_model(client, iris_rf_path, iris_csv_path)
    expl_id = (await client.post(f"{API}/explain/instance",
                                 json={"model_id": model["id"], "dataset_id": ds["id"], "row_index": 0},
                                 headers=headers)).json()["explanation_id"]
    await client.post(f"{API}/explain/chat", json={"explanation_id": expl_id}, headers=headers)
    await client.post(f"{API}/explain/chat", json={"explanation_id": expl_id, "question": "First?"}, headers=headers)
    r = await client.post(f"{API}/explain/chat", json={"explanation_id": expl_id, "question": "Second?"},
                          headers=headers)
    assert r.status_code == 200
    roles = [m["role"] for m in fake_ollama.requests[-1]["messages"]]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]

    conv = (await client.get(f"{API}/explain/{expl_id}/conversation", headers=headers)).json()
    assert [m["kind"] for m in conv["messages"]] == ["narrative", "followup", "followup", "followup", "followup"]


async def test_chat_rejects_client_supplied_attributions(client, fake_ollama, iris_rf_path, iris_csv_path):
    headers, model, _ = await _owner_with_model(client, iris_rf_path, iris_csv_path)
    r = await client.post(f"{API}/explain/chat", json={
        "model_id": model["id"], "prediction": 0.99, "base_value": 0.1, "feature_contributions": [],
    }, headers=headers)
    assert r.status_code == 422


async def test_chat_stream_emits_tokens_then_done(client, fake_ollama, iris_rf_path, iris_csv_path):
    headers, model, ds = await _owner_with_model(client, iris_rf_path, iris_csv_path)
    expl_id = (await client.post(f"{API}/explain/instance",
                                 json={"model_id": model["id"], "dataset_id": ds["id"], "row_index": 0},
                                 headers=headers)).json()["explanation_id"]
    fake_ollama.reply = lambda body: "one two three"
    r = await client.post(f"{API}/explain/chat/stream", json={"explanation_id": expl_id}, headers=headers)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = [line.split(": ", 1)[1] for line in r.text.splitlines() if line.startswith("event: ")]
    assert events.count("token") == 3 and events[-1] == "done"


async def test_chat_stream_reports_llm_errors(client, fake_ollama, iris_rf_path, iris_csv_path):
    headers, model, ds = await _owner_with_model(client, iris_rf_path, iris_csv_path)
    expl_id = (await client.post(f"{API}/explain/instance",
                                 json={"model_id": model["id"], "dataset_id": ds["id"], "row_index": 0},
                                 headers=headers)).json()["explanation_id"]
    fake_ollama.status_code = 500
    r = await client.post(f"{API}/explain/chat/stream", json={"explanation_id": expl_id}, headers=headers)
    assert "event: error" in r.text
    r = await client.post(f"{API}/explain/chat", json={"explanation_id": expl_id}, headers=headers)
    assert r.status_code == 503


# --------------------------------------------------------------------------- RBAC (AC-3)

async def _shared_setup(client, iris_rf_path, iris_csv_path, role):
    owner, model, ds = await _owner_with_model(client, iris_rf_path, iris_csv_path)
    other = await register_and_login(client, "bob@example.com")
    r = await client.post(f"{API}/models/{model['id']}/share",
                          json={"user_email": "bob@example.com", "role": role}, headers=owner)
    assert r.status_code == 200, r.text
    return owner, other, model, ds


async def test_viewer_can_predict_but_not_explain(client, iris_rf_path, iris_csv_path):
    _, viewer, model, ds = await _shared_setup(client, iris_rf_path, iris_csv_path, "VIEWER")
    src = {"model_id": model["id"], "dataset_id": ds["id"], "row_index": 0}
    assert (await client.post(f"{API}/predict/", json=src, headers=viewer)).status_code == 200
    assert (await client.post(f"{API}/explain/instance", json=src, headers=viewer)).status_code == 403
    listed = (await client.get(f"{API}/models/", headers=viewer)).json()
    assert listed[0]["assigned_role"] == "VIEWER" and listed[0]["owner_email"] == "alice@example.com"


async def test_explainer_can_explain_but_not_share_or_audit(client, iris_rf_path, iris_csv_path):
    _, explainer, model, ds = await _shared_setup(client, iris_rf_path, iris_csv_path, "EXPLAINER")
    src = {"model_id": model["id"], "dataset_id": ds["id"], "row_index": 0}
    assert (await client.post(f"{API}/explain/instance", json=src, headers=explainer)).status_code == 200
    await register_and_login(client, "carol@example.com")
    r = await client.post(f"{API}/models/{model['id']}/share",
                          json={"user_email": "carol@example.com", "role": "VIEWER"}, headers=explainer)
    assert r.status_code == 403
    assert (await client.get(f"{API}/audit/models/{model['id']}/metrics", headers=explainer)).status_code == 403


async def test_admin_cannot_grant_admin_or_revoke_admin(client, iris_rf_path, iris_csv_path):
    owner, admin, model, _ = await _shared_setup(client, iris_rf_path, iris_csv_path, "ADMIN")
    await register_and_login(client, "carol@example.com")
    r = await client.post(f"{API}/models/{model['id']}/share",
                          json={"user_email": "carol@example.com", "role": "ADMIN"}, headers=admin)
    assert r.status_code == 403
    r = await client.post(f"{API}/models/{model['id']}/share",
                          json={"user_email": "carol@example.com", "role": "EXPLAINER"}, headers=admin)
    assert r.status_code == 200
    shares = (await client.get(f"{API}/models/{model['id']}/shares", headers=admin)).json()
    assert {s["email"] for s in shares} == {"bob@example.com", "carol@example.com"}
    carol_id = next(s["user_id"] for s in shares if s["email"] == "carol@example.com")
    assert (await client.delete(f"{API}/models/{model['id']}/share/{carol_id}", headers=admin)).status_code == 204
    bob_id = next(s["user_id"] for s in shares if s["email"] == "bob@example.com")
    assert (await client.delete(f"{API}/models/{model['id']}/share/{bob_id}", headers=admin)).status_code == 403
    assert (await client.delete(f"{API}/models/{model['id']}/share/{bob_id}", headers=owner)).status_code == 204


async def test_unshared_user_has_no_access(client, iris_rf_path, iris_csv_path):
    _, model, ds = await _owner_with_model(client, iris_rf_path, iris_csv_path)
    stranger = await register_and_login(client, "eve@example.com")
    assert (await client.get(f"{API}/models/{model['id']}", headers=stranger)).status_code == 403
    assert (await client.get(f"{API}/datasets/{ds['id']}/rows", headers=stranger)).status_code == 403
    assert (await client.get(f"{API}/models/", headers=stranger)).json() == []


async def test_conversation_is_private_to_its_creator(client, fake_ollama, iris_rf_path, iris_csv_path):
    owner, explainer, model, ds = await _shared_setup(client, iris_rf_path, iris_csv_path, "EXPLAINER")
    expl_id = (await client.post(f"{API}/explain/instance",
                                 json={"model_id": model["id"], "dataset_id": ds["id"], "row_index": 0},
                                 headers=owner)).json()["explanation_id"]
    r = await client.post(f"{API}/explain/chat", json={"explanation_id": expl_id}, headers=explainer)
    assert r.status_code == 404


async def test_owner_deletes_model_and_artifacts(client, fake_ollama, iris_rf_path, iris_csv_path):
    owner, other, model, ds = await _shared_setup(client, iris_rf_path, iris_csv_path, "ADMIN")
    await client.post(f"{API}/explain/instance",
                      json={"model_id": model["id"], "dataset_id": ds["id"], "row_index": 0}, headers=owner)
    assert (await client.delete(f"{API}/models/{model['id']}", headers=other)).status_code == 403
    assert (await client.delete(f"{API}/models/{model['id']}", headers=owner)).status_code == 204
    assert (await client.get(f"{API}/models/", headers=owner)).json() == []
    assert not any((settings.MODELS_DIR / "1").glob("*.joblib"))
    # Background data is deleted with the model; the evaluation dataset survives, unlinked
    datasets = (await client.get(f"{API}/datasets/", headers=owner)).json()
    assert len(datasets) == 1
    assert datasets[0]["model_id"] is None and datasets[0]["purpose"] == "evaluation"


# --------------------------------------------------------------------------- audit / evaluation

async def test_audit_metrics_and_export(client, fake_ollama, iris_rf_path, iris_csv_path):
    headers, model, ds = await _owner_with_model(client, iris_rf_path, iris_csv_path)
    for row in (0, 60):
        expl_id = (await client.post(f"{API}/explain/instance",
                                     json={"model_id": model["id"], "dataset_id": ds["id"], "row_index": row},
                                     headers=headers)).json()["explanation_id"]
        await client.post(f"{API}/explain/chat", json={"explanation_id": expl_id}, headers=headers)

    metrics = (await client.get(f"{API}/audit/models/{model['id']}/metrics", headers=headers)).json()
    assert metrics["explanations"] == 2 and metrics["narratives"] == 2
    assert metrics["shap_ms"]["count"] == 2 and metrics["llm_ttft_ms"]["count"] == 2
    assert metrics["faithfulness_zero_fabrication_rate"] == 1.0

    trail = (await client.get(f"{API}/audit/models/{model['id']}/explanations", headers=headers)).json()
    assert trail["total"] == 2 and trail["items"][0]["user_email"] == "alice@example.com"

    r = await client.get(f"{API}/audit/models/{model['id']}/export.csv", headers=headers)
    assert r.status_code == 200
    rows = list(csv.reader(io.StringIO(r.text)))
    assert len(rows) == 3 and rows[1][12] == "narrative"


# --------------------------------------------------------------------------- security helpers & config

def test_path_sandbox_rejects_traversal_and_sibling_prefix():
    with pytest.raises(HTTPException):
        ensure_within_storage(settings.STORAGE_DIR / ".." / ".." / "etc" / "passwd")
    sibling = Path(str(settings.STORAGE_DIR) + "_evil") / "x.joblib"
    with pytest.raises(HTTPException):
        ensure_within_storage(sibling)
    assert ensure_within_storage(settings.MODELS_DIR / "1" / "ok.joblib")


def test_sanitize_filename():
    assert sanitize_filename("../../evil name!.joblib") == "evil_name"
    assert sanitize_filename("") == "upload"


def test_placeholder_secret_key_rejected():
    with pytest.raises(ValidationError):
        Settings(SECRET_KEY="generate-a-secure-random-secret-key-here")


def test_remote_llm_requires_explicit_opt_in():
    with pytest.raises(ValidationError):
        Settings(SECRET_KEY="k" * 40, OLLAMA_BASE_URL="http://203.0.113.5:11434", ALLOW_REMOTE_LLM=False)
    ok = Settings(SECRET_KEY="k" * 40, OLLAMA_BASE_URL="http://203.0.113.5:11434", ALLOW_REMOTE_LLM=True)
    assert ok.llm_is_local is False


# --------------------------------------------------------------------------- model + background upload

async def _upload_with_background(client, headers, model_path, csv_path, target=None):
    data = {"name": "Iris RF", "domain_context": "Botany"}
    if target:
        data["target_column"] = target
    with open(model_path, "rb") as mf, open(csv_path, "rb") as cf:
        return await client.post(
            f"{API}/models/upload", data=data, headers=headers,
            files={"file": (model_path.name, mf), "background_file": (csv_path.name, cf, "text/csv")},
        )


async def test_model_upload_with_background_dataset(client, iris_logreg_path, iris_csv_path):
    headers = await register_and_login(client, "alice@example.com")
    r = await _upload_with_background(client, headers, iris_logreg_path, iris_csv_path)
    assert r.status_code == 201, r.text
    model = r.json()
    assert model["background_dataset_id"] is not None

    datasets = (await client.get(f"{API}/datasets/", params={"model_id": model["id"]}, headers=headers)).json()
    assert datasets[0]["is_background"] and datasets[0]["purpose"] == "background"
    assert datasets[0]["target_column"] == "species"

    # Kernel model can be explained immediately because the background arrived with it
    features = {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2}
    r = await client.post(f"{API}/explain/instance", headers=headers,
                          json={"model_id": model["id"], "instance_features": features})
    assert r.status_code == 200, r.text


async def test_predictions_never_run_on_background_data(client, iris_rf_path, iris_csv_path):
    headers, model, _ = await _owner_with_model(client, iris_rf_path, iris_csv_path)
    bg_id = model["background_dataset_id"]
    row = {"model_id": model["id"], "dataset_id": bg_id, "row_index": 0}
    for path in ("/predict/", "/explain/instance"):
        r = await client.post(f"{API}{path}", json=row, headers=headers)
        assert r.status_code == 422 and "background data" in r.json()["detail"]
    r = await client.post(f"{API}/predict/batch", json={"model_id": model["id"], "dataset_id": bg_id}, headers=headers)
    assert r.status_code == 422
    r = await client.get(f"{API}/predict/batch/export.csv", params={"model_id": model["id"], "dataset_id": bg_id},
                         headers=headers)
    assert r.status_code == 422

    listed = (await client.get(f"{API}/datasets/", params={"model_id": model["id"], "purpose": "evaluation"},
                               headers=headers)).json()
    assert bg_id not in [d["id"] for d in listed]


async def test_only_owner_can_replace_background(client, iris_rf_path, iris_csv_path):
    owner, admin, model, _ = await _shared_setup(client, iris_rf_path, iris_csv_path, "ADMIN")
    old_bg = model["background_dataset_id"]

    with open(iris_csv_path, "rb") as fh:
        r = await client.post(f"{API}/models/{model['id']}/background", headers=admin,
                              files={"file": (iris_csv_path.name, fh, "text/csv")})
    assert r.status_code == 403
    assert (await client.delete(f"{API}/datasets/{old_bg}", headers=admin)).status_code == 403

    with open(iris_csv_path, "rb") as fh:
        r = await client.post(f"{API}/models/{model['id']}/background", headers=owner,
                              files={"file": ("new_background.csv", fh, "text/csv")})
    assert r.status_code == 200, r.text
    new_bg = r.json()["background_dataset_id"]
    assert new_bg != old_bg

    backgrounds = (await client.get(f"{API}/datasets/", params={"model_id": model["id"], "purpose": "background"},
                                    headers=owner)).json()
    assert [d["id"] for d in backgrounds] == [new_bg]  # the previous background was removed
    assert backgrounds[0]["name"] == "new_background.csv"


async def test_background_replacement_is_validated(client, iris_rf_path, iris_csv_path, iris_frame, artifacts_dir):
    headers, model, _ = await _owner_with_model(client, iris_rf_path, iris_csv_path)
    bad = artifacts_dir / "bad_background2.csv"
    iris_frame.drop(columns=["sepal width (cm)"]).to_csv(bad, index=False)
    with open(bad, "rb") as fh:
        r = await client.post(f"{API}/models/{model['id']}/background", headers=headers,
                              files={"file": (bad.name, fh, "text/csv")})
    assert r.status_code == 422
    current = (await client.get(f"{API}/models/{model['id']}", headers=headers)).json()
    assert current["background_dataset_id"] == model["background_dataset_id"]


async def test_model_upload_rolls_back_when_background_invalid(client, iris_rf_path, iris_frame, artifacts_dir):
    headers = await register_and_login(client, "alice@example.com")
    bad = artifacts_dir / "bad_background.csv"
    iris_frame.drop(columns=["petal width (cm)"]).to_csv(bad, index=False)
    r = await _upload_with_background(client, headers, iris_rf_path, bad)
    assert r.status_code == 422 and "Background dataset rejected" in r.json()["detail"]
    assert (await client.get(f"{API}/models/", headers=headers)).json() == []
    assert not any((settings.MODELS_DIR / "1").glob("*"))
    assert not any((settings.DATASETS_DIR / "1").glob("*"))


# --------------------------------------------------------------------------- batch prediction

async def test_batch_prediction_pages_and_metrics(client, iris_rf_path, iris_csv_path):
    headers, model, ds = await _owner_with_model(client, iris_rf_path, iris_csv_path)
    r = await client.post(f"{API}/predict/batch", headers=headers,
                          json={"model_id": model["id"], "dataset_id": ds["id"], "offset": 140, "limit": 20})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total_rows"] == 150 and len(body["rows"]) == 10
    assert body["rows"][0]["row_index"] == 140
    assert sum(body["class_counts"].values()) == 150
    assert body["target_column"] == "species"
    assert body["metrics"]["accuracy"] > 0.9
    assert body["rows"][0]["actual"] == 2 and "species" not in body["rows"][0]["features"]

    single = (await client.post(f"{API}/predict/", headers=headers,
                                json={"model_id": model["id"], "dataset_id": ds["id"], "row_index": 140})).json()
    assert single["predicted_class"] == body["rows"][0]["predicted_class"]
    assert single["prediction"] == pytest.approx(body["rows"][0]["prediction"])


async def test_batch_prediction_export(client, iris_rf_path, iris_csv_path):
    headers, model, ds = await _owner_with_model(client, iris_rf_path, iris_csv_path)
    r = await client.get(f"{API}/predict/batch/export.csv", headers=headers,
                         params={"model_id": model["id"], "dataset_id": ds["id"]})
    assert r.status_code == 200
    rows = list(csv.reader(io.StringIO(r.text)))
    assert len(rows) == 151
    assert rows[0][-5:] == ["predicted_class", "predicted_probability", "prob_0", "prob_1", "prob_2"]


async def test_viewer_can_batch_predict_on_own_dataset(client, iris_rf_path, iris_csv_path):
    _, viewer, model, _ = await _shared_setup(client, iris_rf_path, iris_csv_path, "VIEWER")
    ds = await upload_dataset(client, viewer, iris_csv_path, model_id=model["id"])
    assert ds["is_background"] is False and ds["purpose"] == "evaluation"
    r = await client.post(f"{API}/predict/batch", headers=viewer,
                          json={"model_id": model["id"], "dataset_id": ds["id"]})
    assert r.status_code == 200 and len(r.json()["rows"]) == 50


async def test_batch_regression_metrics(client, artifacts_dir):
    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import RandomForestRegressor
    import joblib
    data = load_diabetes(as_frame=True)
    path = artifacts_dir / "diabetes_rf.joblib"
    joblib.dump(RandomForestRegressor(n_estimators=10, random_state=0).fit(data.data, data.target), path)
    csv_path = artifacts_dir / "diabetes.csv"
    data.frame.to_csv(csv_path, index=False)

    headers = await register_and_login(client, "alice@example.com")
    model = await upload_model(client, headers, path, name="Diabetes")
    ds = await upload_dataset(client, headers, csv_path, model_id=model["id"])
    body = (await client.post(f"{API}/predict/batch", headers=headers,
                              json={"model_id": model["id"], "dataset_id": ds["id"], "limit": 5})).json()
    assert body["task_type"] == "regression"
    assert set(body["metrics"]) >= {"mae", "rmse", "r2"} and body["prediction_stats"]["max"] > 0
