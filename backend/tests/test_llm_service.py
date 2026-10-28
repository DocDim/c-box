import json
from types import SimpleNamespace

import httpx
import pytest

from app.core.faithfulness import check_faithfulness
from app.core.llm_service import (
    ChatTurn,
    GenerationStats,
    LLMUnavailableError,
    OllamaGemmaService,
    build_context_payload,
)
from app.core.prompt_builder import Audience, ModelMetadata, PromptBuilder


def _prompt(domain="Clinical Cardiology", audience=Audience.DECISION_MAKER, **meta):
    return PromptBuilder(ModelMetadata(domain=domain, **meta), audience)


def _record(**overrides):
    base = dict(
        task_type="classification",
        explainer_type="TreeExplainer",
        additivity_gap=1.2e-7,
        output_units="probability",
        base_value=0.354,
        explained_output=0.842,
        prediction=0.842,
        predicted_class="1",
        explained_class="1",
        probabilities={"0": 0.158, "1": 0.842},
        feature_contributions=[
            {"feature": "st_depression", "actual_value": 2.3, "shap_value": 0.284},
            {"feature": "max_heart_rate", "actual_value": 112.0, "shap_value": 0.156},
            {"feature": "cholesterol", "actual_value": 245.0, "shap_value": 0.068},
            {"feature": "age", "actual_value": 58.0, "shap_value": -0.020},
            {"feature": "feature_3", "actual_value": 1.0, "shap_value": 0.0},
        ],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
def payload():
    return build_context_payload(_record(), _prompt(), top_k=2)


def test_context_payload_splits_top_k_and_keeps_remainder(payload):
    assert [f["feature"] for f in payload["top_positive_factors"]] == ["st_depression", "max_heart_rate"]
    assert [f["feature"] for f in payload["top_negative_factors"]] == ["age"]
    assert payload["other_features_net_shap"] == pytest.approx(0.068)
    assert payload["prediction_probability"] == 0.842
    assert payload["domain"] == "Clinical Cardiology"
    assert "probabilities" in payload["units_note"]


def test_regression_payload_has_predicted_value():
    p = build_context_payload(
        _record(task_type="regression", output_units="target units", predicted_class=None,
                explained_class=None, probabilities=None, prediction=151.2),
        _prompt("Real Estate"), top_k=3,
    )
    assert p["predicted_value"] == 151.2
    assert "predicted_class" not in p


def test_narrative_messages_contain_only_context_and_instruction(payload):
    service = OllamaGemmaService()
    history = [ChatTurn("assistant", "narrative", "old narrative")]
    messages = service.build_messages(_prompt(), payload, history, None)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "## Bottom Line" in messages[1]["content"]
    assert "st_depression" in messages[1]["content"]


def test_followup_messages_replay_history(payload):
    service = OllamaGemmaService()
    history = [
        ChatTurn("assistant", "narrative", "N"),
        ChatTurn("user", "followup", "Q1"),
        ChatTurn("assistant", "followup", "A1"),
    ]
    messages = service.build_messages(_prompt(), payload, history, "Q2")
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert "Q1" in messages[3]["content"] and "Q2" in messages[5]["content"]
    assert messages[4]["content"] == "A1"


def test_faithfulness_accepts_rounded_and_percentage_values(payload):
    text = (
        "## Executive Summary\nThe predicted probability is 84.2%, versus a baseline of 0.35, "
        "an increase of 0.49.\n## Upward Pressures\nst_depression (2.3) added 0.28.\n"
        "## Mitigating Factors\nage (58) reduced it slightly by 0.02.\n## Actionable Next Step\nReview."
    )
    result = check_faithfulness(text, payload, expect_sections=True)
    assert result["fabricated"] is False, result
    assert result["numbers_checked"] >= 6
    assert result["structure_complete"] is True
    assert set(result["features_mentioned"]) == {"st_depression", "age"}


def test_faithfulness_flags_invented_numbers(payload):
    result = check_faithfulness("Cholesterol of 310 added 0.9 to the risk.", payload, expect_sections=False)
    assert result["fabricated"] is True
    assert "310" in result["unverified_numbers"] and "0.9" in result["unverified_numbers"]


def test_faithfulness_percentages_must_match_scaled_values():
    # Regression: Gemma reported "1%" for a probability of 1.0 (i.e. 100%)
    payload = build_context_payload(_record(prediction=1.0, explained_output=1.0), _prompt("Botany"), top_k=2)
    assert check_faithfulness("The probability is 1%.", payload, expect_sections=False)["fabricated"] is True
    assert check_faithfulness("The probability is 100%.", payload, expect_sections=False)["fabricated"] is False
    assert check_faithfulness("That is 100 percent.", payload, expect_sections=False)["fabricated"] is False


def test_faithfulness_ignores_digits_inside_feature_names_and_list_markers(payload):
    text = "1. feature_3 had no effect.\n2. Two factors matter most."
    result = check_faithfulness(text, payload, expect_sections=False)
    assert result["fabricated"] is False


def test_faithfulness_reports_missing_sections(payload):
    result = check_faithfulness("## Executive Summary\nOnly a summary.", payload, expect_sections=True)
    assert result["structure_complete"] is False


def _ndjson(*tokens):
    lines = [json.dumps({"message": {"content": t}, "done": False}) for t in tokens]
    lines.append(json.dumps({"message": {"content": ""}, "done": True}))
    return "\n".join(lines)


@pytest.mark.anyio
async def test_stream_chat_collects_tokens_and_ttft():
    seen = {}

    def handler(request: httpx.Request):
        seen["body"] = request.read()
        return httpx.Response(200, text=_ndjson("Hello", " world"))

    service = OllamaGemmaService("http://localhost:11434", "gemma:2b", transport=httpx.MockTransport(handler))
    stats = GenerationStats()
    tokens = [t async for t in service.stream_chat([{"role": "user", "content": "hi"}], stats)]
    assert tokens == ["Hello", " world"]
    assert stats.text == "Hello world"
    assert stats.ttft_ms is not None and stats.latency_ms >= stats.ttft_ms
    body = json.loads(seen["body"])
    assert body["stream"] is True and body["options"]["temperature"] == 0.2


@pytest.mark.anyio
async def test_stream_chat_maps_http_errors():
    service = OllamaGemmaService(
        "http://localhost:11434", "gemma:2b",
        transport=httpx.MockTransport(lambda r: httpx.Response(404, text="model not found")),
    )
    with pytest.raises(LLMUnavailableError, match="404"):
        await service.generate([{"role": "user", "content": "hi"}])


@pytest.mark.anyio
async def test_stream_chat_maps_connection_errors():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    service = OllamaGemmaService("http://localhost:11434", "gemma:2b", transport=httpx.MockTransport(handler))
    with pytest.raises(LLMUnavailableError, match="Cannot connect"):
        await service.generate([{"role": "user", "content": "hi"}])


def test_prediction_summary_is_correct_and_passes_the_checker():
    # Mirrors the screenshot case: P(class 0) = 1.0 with a 0.35 baseline
    record = _record(prediction=1.0, explained_output=1.0, base_value=0.3497, predicted_class="0",
                     explained_class="0", probabilities={"0": 1.0, "1": 0.0, "2": 0.0})
    payload = build_context_payload(record, _prompt("Botany"), top_k=2)
    summary = payload["prediction_summary"]
    assert "100.0%" in summary and "35.0%" in summary and "65.0 percentage points above" in summary
    assert check_faithfulness(summary, payload, expect_sections=False)["fabricated"] is False


def test_factors_carry_explicit_direction(payload):
    assert all(f["effect"] == "raises the prediction" for f in payload["top_positive_factors"])
    assert all(f["effect"] == "lowers the prediction" for f in payload["top_negative_factors"])
    assert payload["top_positive_factors"][0]["shap_percentage_points"] == 28.4


def test_sign_flip_is_flagged(payload):
    # Gemma described a +0.284 driver as -0.284
    result = check_faithfulness("st_depression has a SHAP value of -0.284.", payload, expect_sections=False)
    assert result["fabricated"] is True and result["unverified_numbers"] == ["-0.284"]


def test_regression_and_log_odds_summaries():
    reg = build_context_payload(
        _record(task_type="regression", output_units="target units", predicted_class=None, explained_class=None,
                probabilities=None, prediction=151.2, explained_output=151.2, base_value=152.1),
        _prompt("Real Estate"), top_k=2)
    assert "151.2" in reg["prediction_summary"] and "below" in reg["prediction_summary"]
    assert check_faithfulness(reg["prediction_summary"], reg, expect_sections=False)["fabricated"] is False

    logodds = build_context_payload(_record(output_units="log-odds", explained_output=1.67, base_value=-0.6),
                                    _prompt("Cardiology"), top_k=2)
    assert "above the baseline of -0.600" in logodds["prediction_summary"]
    assert "shap_percentage_points" not in logodds["top_positive_factors"][0]


def test_direction_check_flags_factors_in_the_wrong_section(payload):
    # Mirrors gemma:2b output: a lowering factor presented as an upward pressure and vice versa
    text = (
        "## Executive Summary\nThe prediction is high.\n"
        "## Upward Pressures\nAge pushed the risk up.\n"
        "## Mitigating Factors\nSt depression lowered it.\n"
        "## Actionable Next Step\nReview the case."
    )
    result = check_faithfulness(text, payload, expect_sections=True)
    assert result["direction_errors"] is True
    assert {(m["feature"], m["section"]) for m in result["misplaced_features"]} == {
        ("age", "upward_pressures"), ("st_depression", "mitigating_factors")
    }


def test_direction_check_passes_correct_narrative(payload):
    text = (
        "## Executive Summary\nHigh.\n## Upward Pressures\nst_depression and max_heart_rate raised it.\n"
        "## Mitigating Factors\nAge lowered it slightly.\n## Actionable Next Step\nReview."
    )
    result = check_faithfulness(text, payload, expect_sections=True)
    assert result["direction_errors"] is False and result["misplaced_features"] == []


# --------------------------------------------------------------------------- prompt-engineering layer

def test_each_audience_has_its_own_persona_and_headings():
    seen = set()
    for audience in Audience:
        prompt = _prompt(audience=audience)
        headings = tuple(prompt.section_headings.values())
        assert set(prompt.section_headings) == {
            "executive_summary", "upward_pressures", "mitigating_factors", "actionable_next_step"
        }
        assert all(f"## {h}" in prompt.narrative_instruction() for h in headings)
        assert "STRICT RULES" in prompt.system_prompt()  # grounding rules are shared
        seen.add(headings)
    assert len(seen) == len(Audience)
    assert "no data-science background" in _prompt().system_prompt()
    assert "data scientist" in _prompt(audience=Audience.DATA_SCIENTIST).system_prompt()


def test_followup_word_limit_depends_on_audience():
    assert "at most 100 words" in _prompt().followup_instruction("Why?")
    assert "at most 150 words" in _prompt(audience=Audience.DATA_SCIENTIST).followup_instruction("Why?")


def test_metadata_reaches_the_payload():
    prompt = _prompt(
        model_name="Heart RF",
        feature_glossary={"st_depression": "ST segment depression on ECG"},
        outcome_description="Risk of heart disease",
        class_descriptions={"1": "high cardiac risk"},
        decision_context="Refer to a cardiologist or schedule routine follow-up",
    )
    payload = build_context_payload(_record(), prompt, top_k=2)
    labels = {f["feature"]: f["label"] for f in payload["top_positive_factors"]}
    assert labels == {"st_depression": "ST segment depression on ECG", "max_heart_rate": "max heart rate"}
    assert "class 1 (high cardiac risk)" in payload["prediction_summary"]
    assert payload["model_context"]["decision_context"].startswith("Refer to a cardiologist")
    assert payload["audience"] == "Decision-maker"
    assert "additivity_gap" not in payload


def test_data_scientist_payload_has_diagnostics():
    payload = build_context_payload(_record(), _prompt(audience=Audience.DATA_SCIENTIST), top_k=2)
    assert payload["explainer_type"] == "TreeExplainer"
    assert payload["additivity_gap"] == pytest.approx(1.2e-7)


def test_faithfulness_uses_audience_headings_and_glossary_labels():
    prompt = _prompt(feature_glossary={"st_depression": "ST segment depression", "age": "Patient age"})
    payload = build_context_payload(_record(), prompt, top_k=2)
    text = (
        "## Bottom Line\nHigh risk.\n"
        "## What Is Driving This Result\nPatient age pushed the result up.\n"
        "## What Is Holding It Back\nST segment depression held it back.\n"
        "## Recommended Action\nReview."
    )
    result = check_faithfulness(text, payload, expect_sections=True, section_headings=prompt.section_headings)
    assert result["structure_complete"] is True
    assert set(result["features_mentioned"]) == {"st_depression", "age"}
    assert {(m["feature"], m["section"]) for m in result["misplaced_features"]} == {
        ("age", "upward_pressures"), ("st_depression", "mitigating_factors")
    }

    # The generic patterns would not find these headings: the audience map is what makes them count
    partial = check_faithfulness("## Bottom Line\nOnly this.", payload, True, prompt.section_headings)
    assert partial["sections_found"] == ["executive_summary"] and partial["structure_complete"] is False


def test_numbers_from_owner_metadata_are_not_flagged():
    prompt = _prompt(decision_context="Refer when the risk exceeds 70%; re-test within 14 days.")
    payload = build_context_payload(_record(), prompt, top_k=2)
    result = check_faithfulness("Refer when risk exceeds 70% and re-test within 14 days.", payload, False)
    assert result["fabricated"] is False, result
