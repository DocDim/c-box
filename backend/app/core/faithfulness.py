"""
Automated grounding check for LLM narratives (AC-5).

Every number in the LLM output is compared with the numbers present in (or trivially
derived from) the SHAP context payload. A number counts as verified when it is a
rounding of an allowed value. This gives a measurable "zero fabricated values" rate
for the evaluation report instead of anecdotal review.
"""
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# group(1) = the number, group(2) = a trailing percent marker
_NUMBER = re.compile(
    r"(?<![\w.])([-+−]?\d+(?:,\d{3})*(?:\.\d+)?)(\s*(?:%|percent(?:age\s+points?)?\b|pp\b))?", re.IGNORECASE
)
_LIST_MARKER = re.compile(r"^\s*(?:#+\s*)?\d+[.)]\s+", re.MULTILINE)
_STRUCTURAL_MAX_INT = 10  # section numbers, "two factors", "top 5" ...

SECTION_PATTERNS = {
    "executive_summary": re.compile(r"executive\s+summary|summary", re.IGNORECASE),
    "upward_pressures": re.compile(r"upward|increas|driv", re.IGNORECASE),
    "mitigating_factors": re.compile(r"mitigat|downward|decreas|reduc", re.IGNORECASE),
    "actionable_next_step": re.compile(r"next\s+step|actionable|recommend", re.IGNORECASE),
}


def _collect_numbers(obj: Any, out: List[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, str):
        try:
            out.append(float(obj))
        except ValueError:
            pass
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_numbers(v, out)


def _allowed_values(payload: Dict[str, Any]) -> Tuple[Set[float], Set[float]]:
    """Returns (plain values, percentage values). A '%' figure must match value x 100."""
    raw: List[float] = []
    _collect_numbers(payload, raw)
    base = payload.get("baseline_expectation")
    final = payload.get("explained_output")
    if isinstance(base, (int, float)) and isinstance(final, (int, float)):
        raw.append(final - base)
    plain: Set[float] = set()
    for v in raw:
        plain.update({v, abs(v)})
    percent = {v * 100 for v in plain}
    # Numbers the owner wrote in the model metadata ("refer above 70% risk") may be quoted as written
    for text in _metadata_texts(payload.get("model_context") or {}):
        for m in _NUMBER.finditer(text):
            value = abs(float(m.group(1).replace(",", "").replace("−", "-")))
            (percent if m.group(2) else plain).add(value)
    return plain, percent


def _metadata_texts(context: Dict[str, Any]) -> List[str]:
    texts = [v for v in context.values() if isinstance(v, str)]
    texts += [v for v in (context.get("class_descriptions") or {}).values() if isinstance(v, str)]
    return texts


def _decimals(token: str) -> int:
    return len(token.split(".")[1]) if "." in token else 0


def _is_verified(token: str, value: float, allowed: Iterable[float], is_percent: bool) -> bool:
    # Small bare integers are structural ("two factors", "top 5"); percentages never are
    if not is_percent and float(value).is_integer() and 0 <= abs(value) <= _STRUCTURAL_MAX_INT:
        return True
    tolerance = 0.5 * 10 ** (-_decimals(token)) + 1e-9
    for a in allowed:
        if abs(a - value) <= tolerance or (a != 0 and abs(a - value) / abs(a) <= 0.01):
            return True
    return False


def _strip_feature_names(text: str, names: List[str]) -> str:
    # Feature names like "feature_3" or "sepal length (cm)" must not be read as numbers
    for name in sorted(names, key=len, reverse=True):
        if name:
            text = re.sub(re.escape(name), " ", text, flags=re.IGNORECASE)
    return text


def _feature_aliases(name: str, label: Optional[str] = None) -> List[str]:
    aliases = {name, name.replace("_", " "), re.sub(r"\s*\(.*?\)", "", name).strip()}
    if label:
        # Business label from the model's feature glossary ("ST segment depression")
        aliases.update({label, re.sub(r"\s*\(.*?\)", "", label).strip()})
    return list(aliases)


def check_faithfulness(
    text: str,
    payload: Dict[str, Any],
    expect_sections: bool,
    section_headings: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    section_headings maps each semantic section key to the heading the audience's prompt
    asked for; without it, the generic SECTION_PATTERNS are used.
    """
    factors = [f for key in ("top_positive_factors", "top_negative_factors") for f in payload.get(key, [])]
    features: List[str] = [f["feature"] for f in factors]
    aliases = {f["feature"]: _feature_aliases(f["feature"], f.get("label")) for f in factors}
    class_labels = list((payload.get("class_probabilities") or {}).keys())
    names_to_strip = [a for f in features for a in aliases[f]] + list(
        payload.get("all_feature_names", [])
    )

    scrubbed = _LIST_MARKER.sub(" ", text)
    scrubbed = _strip_feature_names(scrubbed, names_to_strip)

    plain, percent = _allowed_values(payload)
    for lbl in class_labels:
        try:
            plain.add(float(lbl))
        except ValueError:
            pass

    checked, unverified = 0, []
    for match in _NUMBER.finditer(scrubbed):
        token = match.group(1).replace(",", "").replace("−", "-").lstrip("+")
        is_percent = bool(match.group(2))
        try:
            value = float(token)
        except ValueError:
            continue
        checked += 1
        allowed = percent if is_percent else plain
        if not _is_verified(token.lstrip("-"), value, allowed, is_percent):
            unverified.append(match.group(0).strip())

    lowered = text.lower()
    mentioned = [
        f for f in features
        if any(alias.lower() in lowered for alias in aliases[f] if alias)
    ]

    result: Dict[str, Any] = {
        "numbers_checked": checked,
        "numbers_verified": checked - len(unverified),
        "unverified_numbers": unverified[:20],
        "fabricated": bool(unverified),
        "features_in_context": len(features),
        "features_mentioned": mentioned,
    }
    if expect_sections:
        sections = _split_sections(text, section_headings)
        if section_headings:
            found = [key for key in section_headings if key in sections]
        else:
            found = [name for name, pat in SECTION_PATTERNS.items() if pat.search(text)]
        result["sections_found"] = found
        result["structure_complete"] = len(found) == len(SECTION_PATTERNS)

        # Direction check: a feature that lowers the prediction must not be presented as an
        # upward pressure, and vice versa. Catches sign errors even when no number is quoted.
        positives = [f["feature"] for f in payload.get("top_positive_factors", [])]
        negatives = [f["feature"] for f in payload.get("top_negative_factors", [])]
        misplaced = [
            {"feature": f, "section": "upward_pressures", "actual_effect": "lowers the prediction"}
            for f in negatives if _mentions(sections.get("upward_pressures", ""), aliases[f])
        ] + [
            {"feature": f, "section": "mitigating_factors", "actual_effect": "raises the prediction"}
            for f in positives if _mentions(sections.get("mitigating_factors", ""), aliases[f])
        ]
        result["misplaced_features"] = misplaced
        result["direction_errors"] = bool(misplaced)
    return result


_HEADING = re.compile(r"^\s*(?:#{1,4}\s*|\*\*)(.+?)(?:\*\*)?\s*:?\s*$")
# Checked in this order so "Executive Summary" is not mistaken for another section
_HEADING_ORDER = ("mitigating_factors", "actionable_next_step", "upward_pressures", "executive_summary")


def _normalize_heading(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _heading_key(heading: str, section_headings: Optional[Dict[str, str]]) -> Optional[str]:
    if section_headings:
        found = _normalize_heading(heading)
        return next((k for k, h in section_headings.items() if _normalize_heading(h) in found), None)
    return next((n for n in _HEADING_ORDER if SECTION_PATTERNS[n].search(heading)), None)


def _split_sections(text: str, section_headings: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Maps each narrative section to its body, using the markdown heading lines."""
    sections: Dict[str, List[str]] = {}
    current = None
    for line in text.splitlines():
        heading = _HEADING.match(line)
        if heading:
            label = _heading_key(heading.group(1), section_headings)
            if label:
                current = label
                sections.setdefault(current, [])
                continue
        if current:
            sections[current].append(line)
    return {k: "\n".join(v) for k, v in sections.items()}


def _mentions(text: str, aliases: List[str]) -> bool:
    lowered = text.lower()
    return any(alias and alias.lower() in lowered for alias in aliases)
