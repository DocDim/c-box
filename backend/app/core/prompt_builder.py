"""
Prompt-engineering layer: turns model metadata and the reader's audience level into the
system prompt, the narrative instruction and the follow-up instruction sent to the LLM.

The grounding rules are shared by every audience; only the persona, the vocabulary, the
depth and the section headings change. Each section carries a semantic key so the
faithfulness checker can verify structure and factor directions whatever the headings are.
"""
import enum
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

# Semantic section keys, shared with app.core.faithfulness
SUMMARY = "executive_summary"
UPWARD = "upward_pressures"
DOWNWARD = "mitigating_factors"
NEXT_STEP = "actionable_next_step"


class Audience(str, enum.Enum):
    DECISION_MAKER = "decision_maker"
    DOMAIN_ANALYST = "domain_analyst"
    DATA_SCIENTIST = "data_scientist"


DEFAULT_AUDIENCE = Audience.DECISION_MAKER


@dataclass(frozen=True)
class Section:
    key: str
    heading: str
    instruction: str


@dataclass(frozen=True)
class AudienceProfile:
    label: str
    reader: str
    style: Tuple[str, ...]
    sections: Tuple[Section, ...]
    narrative_words: int
    followup_words: int


@dataclass(frozen=True)
class ModelMetadata:
    """Business context recorded on the model artifact by its owner."""
    domain: str
    model_name: str = ""
    task_type: str = "unknown"
    feature_glossary: Dict[str, str] = field(default_factory=dict)
    outcome_description: Optional[str] = None
    class_descriptions: Dict[str, str] = field(default_factory=dict)
    decision_context: Optional[str] = None

    @classmethod
    def from_artifact(cls, model: Any) -> "ModelMetadata":
        return cls(
            domain=model.domain_context,
            model_name=model.name,
            task_type=model.task_type,
            feature_glossary=dict(model.feature_glossary or {}),
            outcome_description=model.outcome_description,
            class_descriptions=dict(model.class_descriptions or {}),
            decision_context=model.decision_context,
        )

    def feature_label(self, feature: str) -> str:
        return self.feature_glossary.get(feature) or feature.replace("_", " ")

    def class_label(self, cls: Optional[str]) -> Optional[str]:
        return self.class_descriptions.get(str(cls)) if cls is not None else None


PROFILES: Dict[Audience, AudienceProfile] = {
    Audience.DECISION_MAKER: AudienceProfile(
        label="Decision-maker",
        reader="a {domain} decision-maker with no data-science background who must act on this single prediction",
        style=(
            "Refer to every feature by its 'label'. Never show raw column names or the words SHAP, log-odds, "
            "baseline, model margin or attribution.",
            "Describe each factor's weight as strong, moderate or slight; quote at most one number per factor, "
            "as percentage points when 'shap_percentage_points' is given.",
            "Lead with what the result means for the decision, in short sentences without formulas.",
        ),
        sections=(
            Section(SUMMARY, "Bottom Line",
                    "begin by restating 'prediction_summary' exactly as written, then one sentence on what it "
                    "means for the decision."),
            Section(UPWARD, "What Is Driving This Result",
                    "only the factors in 'top_positive_factors', most influential first, each with its value in "
                    "plain words."),
            Section(DOWNWARD, "What Is Holding It Back",
                    "only the factors in 'top_negative_factors'. If that list is empty, write that nothing "
                    "pulled the result down."),
            Section(NEXT_STEP, "Recommended Action",
                    "one concrete action, chosen from 'model_context.decision_context' when it is given, and a "
                    "reminder that the model supports rather than replaces professional judgment."),
        ),
        narrative_words=220,
        followup_words=100,
    ),
    Audience.DOMAIN_ANALYST: AudienceProfile(
        label="Domain analyst",
        reader="a {domain} analyst who knows the domain and basic statistics but is not a machine-learning specialist",
        style=(
            "Name each feature by its 'label' followed by the raw column name in parentheses the first time.",
            "For each factor quote its actual value and its effect, in percentage points when "
            "'shap_percentage_points' is given, otherwise in the output units.",
            "You may relate a factor to domain knowledge, but mark it as interpretation, not as a model fact.",
        ),
        sections=(
            Section(SUMMARY, "Prediction Overview",
                    "begin by restating 'prediction_summary' exactly as written, then one sentence putting it "
                    "in domain context."),
            Section(UPWARD, "Key Contributing Factors",
                    "only the factors in 'top_positive_factors', each with its value and effect size."),
            Section(DOWNWARD, "Counterbalancing Factors",
                    "only the factors in 'top_negative_factors', each with its value and effect size. If that list "
                    "is empty, write that no feature lowered the prediction."),
            Section(NEXT_STEP, "Points to Verify",
                    "two or three input values or factors worth checking before relying on this result, in light "
                    "of 'model_context.decision_context' when it is given."),
        ),
        narrative_words=320,
        followup_words=150,
    ),
    Audience.DATA_SCIENTIST: AudienceProfile(
        label="Data scientist",
        reader="a data scientist auditing how the model reached this prediction",
        style=(
            "Use the exact feature column names and SHAP values with their units ('output_units', 'units_note').",
            "Relate the output to the baseline: baseline plus all contributions equals the explained output; "
            "mention 'other_features_net_shap' and the 'additivity_gap'.",
            "Be precise and concise; technical vocabulary is expected.",
        ),
        sections=(
            Section(SUMMARY, "Model Output",
                    "restate 'prediction_summary' exactly as written, then the explained output against the "
                    "baseline expectation in the output units."),
            Section(UPWARD, "Positive Contributions",
                    "only the factors in 'top_positive_factors', each as feature = value, SHAP value."),
            Section(DOWNWARD, "Negative Contributions",
                    "only the factors in 'top_negative_factors', each as feature = value, SHAP value. If that list "
                    "is empty, say so."),
            Section(NEXT_STEP, "Diagnostics & Caveats",
                    "the explainer type, the additivity gap, the net effect of the remaining features, and the "
                    "limits of SHAP (correlated features, no causal claim)."),
        ),
        narrative_words=300,
        followup_words=150,
    ),
}


class PromptBuilder:
    """Builds every prompt for one (model metadata, audience) pair."""

    def __init__(self, metadata: ModelMetadata, audience: Audience = DEFAULT_AUDIENCE):
        self.metadata = metadata
        self.audience = Audience(audience)
        self.profile = PROFILES[self.audience]

    @property
    def section_headings(self) -> Dict[str, str]:
        """Semantic key -> heading, used by the faithfulness checker."""
        return {s.key: s.heading for s in self.profile.sections}

    def _reader(self) -> str:
        return self.profile.reader.format(domain=self.metadata.domain)

    def system_prompt(self) -> str:
        style = "\n".join(f"- {rule}" for rule in self.profile.style)
        return (
            f"You are an Explainable AI specialist in {self.metadata.domain}. You explain one prediction of a "
            f"machine learning model to {self._reader()}.\n"
            "STRICT RULES:\n"
            "1. Use ONLY the verified SHAP attributions in the JSON context. Do not invent features, values or causes.\n"
            "2. Only quote numbers that appear in the context; round them rather than computing new ones.\n"
            "3. A positive SHAP value pushed the prediction up, a negative one pushed it down. SHAP describes "
            "the model's reasoning, not real-world causation; say so if asked about causes.\n"
            "4. If the context cannot answer a question, say that plainly instead of guessing.\n"
            "5. 'model_context' describes what the model predicts and the decision it supports; use it for wording "
            "and recommendations, never as evidence about this case.\n"
            f"WRITING STYLE FOR THIS READER:\n{style}"
        )

    def narrative_instruction(self) -> str:
        headings = "\n".join(f"## {s.heading}" for s in self.profile.sections)
        content = "\n".join(f"- {s.heading}: {s.instruction}" for s in self.profile.sections)
        upward, downward = (self.section_headings[k] for k in (UPWARD, DOWNWARD))
        return (
            "Write the explanation under exactly these four markdown headings, in this order, "
            f"copying each heading line verbatim:\n{headings}\n\n"
            f"Content for each section:\n{content}\n"
            f"Factors that raise the prediction belong only under '{upward}', factors that lower it only under "
            f"'{downward}'. Never change the sign of a SHAP value. Keep the whole answer under "
            f"{self.profile.narrative_words} words."
        )

    def followup_instruction(self, question: str) -> str:
        return (
            f"Answer the follow-up question directly in at most {self.profile.followup_words} words, using only the "
            f"context above and the same writing style. Do not repeat the four-part structure.\nQuestion: {question}"
        )

    def context_extras(self) -> Dict[str, Any]:
        """Metadata added to the grounding payload so the LLM (and the checker) see the same facts."""
        model_context = {
            k: v for k, v in {
                "model_name": self.metadata.model_name,
                "outcome_description": self.metadata.outcome_description,
                "decision_context": self.metadata.decision_context,
                "class_descriptions": self.metadata.class_descriptions or None,
            }.items() if v
        }
        return {"audience": self.profile.label, "model_context": model_context}
