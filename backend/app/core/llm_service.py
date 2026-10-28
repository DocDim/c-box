import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence

import httpx
from app.config import settings
from app.core.prompt_builder import Audience, PromptBuilder

_UNIT_EXPLANATIONS = {
    "probability": (
        "Baseline, contributions and final output are probabilities between 0 and 1 for the explained class. "
        "To state one as a percentage, multiply by 100 (0.85 = 85%, 1.0 = 100%)."
    ),
    "log-odds": (
        "Baseline and contributions are in log-odds for the explained class; "
        "'prediction_probability' is the same output converted to a probability."
    ),
    "target units": "Baseline, contributions and final output are in the units of the predicted target value.",
    "raw model margin": "Baseline and contributions are on the model's internal score scale.",
}

def _prediction_summary(record: Any, prompt: PromptBuilder) -> str:
    """
    A ready-made, numerically correct sentence comparing prediction and baseline, so the LLM
    restates it instead of converting units itself (gemma:2b reported P = 1.0 as "1.0%").
    """
    delta = record.explained_output - record.base_value
    direction = "above" if delta >= 0 else "below"
    if record.task_type == "classification":
        outcome = prompt.metadata.class_label(record.predicted_class)
        named = f"class {record.predicted_class}" + (f" ({outcome})" if outcome else "")
        head = f"The model predicts {named} with probability {record.prediction * 100:.1f}%."
        if record.output_units == "probability":
            return (
                f"{head} The baseline expectation for this class is {record.base_value * 100:.1f}%, "
                f"so this prediction is {abs(delta) * 100:.1f} percentage points {direction} the baseline."
            )
        return (
            f"{head} On the model's {record.output_units} scale the output is {record.explained_output:.3f}, "
            f"{direction} the baseline of {record.base_value:.3f}."
        )
    return (
        f"The model predicts a value of {record.prediction:.4g}, which is {abs(delta):.4g} {direction} "
        f"the baseline expectation of {record.base_value:.4g}."
    )


def build_context_payload(record: Any, prompt: PromptBuilder, top_k: int) -> Dict[str, Any]:
    """
    Packages a stored ExplanationRecord into the immutable JSON grounding payload (FR-4.1):
    top-K upward and top-K downward factors plus the net effect of everything else,
    with the model's business metadata for the prompt's audience.
    """
    contributions: List[Dict[str, Any]] = list(record.feature_contributions)
    positive = sorted((c for c in contributions if c["shap_value"] > 0), key=lambda c: -c["shap_value"])
    negative = sorted((c for c in contributions if c["shap_value"] < 0), key=lambda c: c["shap_value"])
    shown = {c["feature"] for c in positive[:top_k] + negative[:top_k]}
    remainder = sum(c["shap_value"] for c in contributions if c["feature"] not in shown)

    in_probability = record.output_units == "probability"

    def _fmt(c: Dict[str, Any]) -> Dict[str, Any]:
        value = c["actual_value"]
        item = {
            "feature": c["feature"],
            "label": prompt.metadata.feature_label(c["feature"]),
            "value": round(value, 4) if isinstance(value, float) else value,
            "shap": round(c["shap_value"], 4),
            # Explicit direction: small models misread signs (e.g. listing a +0.30 driver as -0.30)
            "effect": "raises the prediction" if c["shap_value"] > 0 else "lowers the prediction",
        }
        if in_probability:
            item["shap_percentage_points"] = round(c["shap_value"] * 100, 2)
        return item

    payload: Dict[str, Any] = {
        "domain": prompt.metadata.domain,
        **prompt.context_extras(),
        "task_type": record.task_type,
        "output_units": record.output_units,
        "units_note": _UNIT_EXPLANATIONS.get(record.output_units, ""),
        "prediction_summary": _prediction_summary(record, prompt),
        "baseline_expectation": round(record.base_value, 4),
        "explained_output": round(record.explained_output, 4),
        "top_positive_factors": [_fmt(c) for c in positive[:top_k]],
        "top_negative_factors": [_fmt(c) for c in negative[:top_k]],
        "other_features_net_shap": round(remainder, 4),
        "all_feature_names": [c["feature"] for c in contributions],
    }
    if record.task_type == "classification":
        payload["predicted_class"] = record.predicted_class
        payload["explained_class"] = record.explained_class
        payload["prediction_probability"] = round(record.prediction, 4)
        if record.probabilities:
            payload["class_probabilities"] = {k: round(v, 4) for k, v in record.probabilities.items()}
    else:
        payload["predicted_value"] = round(record.prediction, 4)
    if prompt.audience == Audience.DATA_SCIENTIST:
        payload["explainer_type"] = record.explainer_type
        payload["additivity_gap"] = float(f"{record.additivity_gap:.3g}")
    return payload


@dataclass
class ChatTurn:
    role: str
    kind: str
    content: str


@dataclass
class GenerationStats:
    text: str = ""
    ttft_ms: Optional[float] = None
    latency_ms: float = 0.0
    model: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


class LLMUnavailableError(RuntimeError):
    pass


class OllamaGemmaService:
    def __init__(
        self,
        base_url: str = settings.OLLAMA_BASE_URL,
        model_name: str = settings.OLLAMA_MODEL,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        # Injectable transport lets tests mock Ollama without a running daemon
        self.transport = transport

    def build_messages(
        self,
        prompt: PromptBuilder,
        payload: Dict[str, Any],
        history: Sequence[ChatTurn],
        question: Optional[str],
    ) -> List[Dict[str, str]]:
        """
        Rebuilds the full multi-turn conversation (FR-4.4). The grounding context is
        always the first user turn, so every answer stays bounded by the same payload.
        """
        context_block = "Context Data (verified SHAP attributions):\n" + json.dumps(payload, indent=2)
        has_narrative = any(t.kind == "narrative" and t.role == "assistant" for t in history)
        opening = prompt.narrative_instruction() if (has_narrative or question is None) else (
            "Use this context to answer the stakeholder's questions."
        )
        messages = [
            {"role": "system", "content": prompt.system_prompt()},
            {"role": "user", "content": f"{context_block}\n\n{opening}"},
        ]
        if question is None:
            return messages  # narrative request: context + 4-part instruction only
        for turn in history:
            if turn.role == "assistant":
                messages.append({"role": "assistant", "content": turn.content})
            elif turn.kind == "followup":
                messages.append({"role": "user", "content": prompt.followup_instruction(turn.content)})
        if question is not None:
            messages.append({"role": "user", "content": prompt.followup_instruction(question)})
        return messages

    async def stream_chat(self, messages: List[Dict[str, str]], stats: GenerationStats) -> AsyncIterator[str]:
        """Streams tokens from Ollama's /api/chat NDJSON stream, recording latency metrics (NFR-3)."""
        stats.model = self.model_name
        started = time.perf_counter()
        body = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "options": {"temperature": settings.LLM_TEMPERATURE, "num_predict": settings.LLM_MAX_TOKENS},
        }
        try:
            async with httpx.AsyncClient(timeout=settings.OLLAMA_TIMEOUT_SECONDS, transport=self.transport) as client:
                async with client.stream("POST", f"{self.base_url}/api/chat", json=body) as response:
                    if response.status_code >= 400:
                        detail = (await response.aread()).decode(errors="replace")[:300]
                        raise LLMUnavailableError(f"Ollama returned HTTP {response.status_code}: {detail}")
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        chunk = json.loads(line)
                        if chunk.get("error"):
                            raise LLMUnavailableError(f"Ollama error: {chunk['error']}")
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            if stats.ttft_ms is None:
                                stats.ttft_ms = (time.perf_counter() - started) * 1000
                            stats.text += token
                            yield token
                        if chunk.get("done"):
                            break
        except httpx.ConnectError:
            raise LLMUnavailableError(
                f"Cannot connect to Ollama daemon at {self.base_url}. Ensure 'ollama serve' is running."
            )
        except httpx.TimeoutException:
            raise LLMUnavailableError("Ollama did not respond before the timeout.")
        finally:
            stats.latency_ms = (time.perf_counter() - started) * 1000

    async def generate(self, messages: List[Dict[str, str]]) -> GenerationStats:
        stats = GenerationStats()
        async for _ in self.stream_chat(messages, stats):
            pass
        return stats


llm_service = OllamaGemmaService()
