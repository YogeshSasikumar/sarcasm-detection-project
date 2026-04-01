"""
explainability.py - Generate human-readable explanations for sarcasm predictions
"""

import re
from utils.logger import get_logger

logger = get_logger(__name__)

# ─── Lexicons for rule-based reasoning ──────────────────────────────────────────

POSITIVE_WORDS = {
    "great", "wonderful", "fantastic", "amazing", "excellent", "brilliant",
    "perfect", "awesome", "superb", "outstanding", "incredible", "love",
    "best", "good", "nice", "fine", "beautiful", "happy", "pleased",
    "delightful", "joy", "glad", "excited", "pleased", "stunning",
}

NEGATIVE_CONTEXT_WORDS = {
    "wrong", "fail", "broken", "bad", "terrible", "awful", "hate",
    "problem", "issue", "error", "bug", "delay", "late", "missed",
    "lost", "hurt", "sick", "tired", "again", "always", "never",
    "monday", "traffic", "crash", "down", "breaks", "flat", "cold",
    "deadline", "pressure", "worst", "horrible", "disaster",
}

SARCASM_MARKERS = {
    "oh sure", "yeah right", "clearly", "obviously", "naturally", "of course",
    "as always", "how wonderful", "what a surprise", "nobody saw", "love how",
    "just great", "wow amazing", "brilliant timing", "stellar", "super helpful",
}

NEGATION_WORDS = {"not", "no", "never", "n't", "cannot", "won't", "can't",
                  "don't", "isn't", "wasn't", "weren't", "haven't", "hadn't"}


def _find_keywords(text: str, word_set: set) -> list[str]:
    """Find words from a set that appear in the text."""
    text_lower = text.lower()
    return [w for w in word_set if w in text_lower]


def _find_markers(text: str) -> list[str]:
    """Find multi-word sarcasm markers in the text."""
    text_lower = text.lower()
    return [m for m in SARCASM_MARKERS if m in text_lower]


def _build_reasoning(
    text: str,
    context: str,
    is_sarcastic: bool,
    confidence: float,
    top_features: list[tuple[str, float]],
) -> str:
    """Build a human-readable reasoning string."""
    reasons = []
    text_lower = (text + " " + context).lower()

    pos_hits  = _find_keywords(text_lower, POSITIVE_WORDS)
    neg_hits  = _find_keywords(text_lower, NEGATIVE_CONTEXT_WORDS)
    neg_words = _find_keywords(text_lower, NEGATION_WORDS)
    markers   = _find_markers(text_lower)

    if is_sarcastic:
        if markers:
            reasons.append(f"Sarcasm marker detected: '{markers[0]}'")
        if pos_hits and neg_hits:
            reasons.append(
                f"Positive word(s) ({', '.join(pos_hits[:2])}) used in a negative context "
                f"({', '.join(neg_hits[:2])})"
            )
        elif pos_hits and neg_words:
            reasons.append(
                f"Positive language ({', '.join(pos_hits[:2])}) paired with negation "
                f"({', '.join(neg_words[:2])})"
            )
        elif pos_hits:
            reasons.append(
                f"Overly positive phrasing ({', '.join(pos_hits[:3])}) inconsistent with context"
            )
        if top_features:
            kw = [f[0] for f in top_features[:3] if f[1] > 0]
            if kw:
                reasons.append(f"Key influencing words: {', '.join(kw)}")
        if not reasons:
            reasons.append("Model detected ironic or contradictory tone in the text")
    else:
        if pos_hits and not neg_hits:
            reasons.append(f"Genuinely positive sentiment ({', '.join(pos_hits[:2])})")
        elif neg_hits and not pos_hits:
            reasons.append("Straightforward negative sentiment — no ironic contrast detected")
        else:
            reasons.append("Text tone is consistent and sincere")
        if top_features:
            kw = [f[0] for f in top_features[:3] if f[1] < 0]
            if kw:
                reasons.append(f"Non-sarcastic indicators: {', '.join(kw)}")
        if not reasons:
            reasons.append("Model found no sarcastic patterns in the text or context")

    return ". ".join(reasons) + "."


def explain(
    text: str,
    context: str,
    sarcasm_result: dict,
    top_features: list[tuple[str, float]] | None = None,
) -> dict:
    """
    Generate an explanation for a sarcasm prediction.

    Args:
        text           : The input message.
        context        : The conversation context.
        sarcasm_result : Output from SarcasmDetector.predict().
        top_features   : Optional list from SarcasmDetector.get_top_features().

    Returns:
        {
            "label"     : str,
            "confidence": float,
            "confidence_pct": str,
            "reasoning" : str,
            "keywords"  : {"positive": [...], "negative": [...], "markers": [...]},
            "top_features": [(word, weight), ...]
        }
    """
    if top_features is None:
        top_features = []

    is_sarcastic = sarcasm_result["label"] == 1
    confidence   = sarcasm_result["confidence"]

    combined_text = (text + " " + context).lower()
    pos_hits  = _find_keywords(combined_text, POSITIVE_WORDS)
    neg_hits  = _find_keywords(combined_text, NEGATIVE_CONTEXT_WORDS)
    markers   = _find_markers(combined_text)

    reasoning = _build_reasoning(text, context, is_sarcastic, confidence, top_features)

    logger.debug("Explanation generated for label=%s conf=%.4f", sarcasm_result["label_str"], confidence)

    return {
        "label"          : sarcasm_result["label_str"],
        "confidence"     : confidence,
        "confidence_pct" : f"{confidence * 100:.1f}%",
        "reasoning"      : reasoning,
        "keywords"       : {
            "positive": pos_hits[:5],
            "negative": neg_hits[:5],
            "sarcasm_markers": markers[:3],
        },
        "top_features"   : top_features[:10],
    }


if __name__ == "__main__":
    sample_result = {
        "label": 1,
        "label_str": "Sarcastic",
        "confidence": 0.89,
        "probabilities": {"Not Sarcastic": 0.11, "Sarcastic": 0.89},
    }
    exp = explain(
        text="Oh great, another Monday!",
        context="I hate the start of the week",
        sarcasm_result=sample_result,
        top_features=[("great", 0.8), ("monday", 0.6), ("another", 0.4)],
    )
    print("\n🔍 Explanation:")
    for k, v in exp.items():
        print(f"  {k}: {v}")
