"""
context_processor.py - Combine conversation context with current message
"""

from utils.logger import get_logger

logger = get_logger(__name__)

_SEPARATOR = " [CTX] "


class ContextProcessor:
    """
    Merges conversation history with the current utterance to create
    a context-aware input string for the sarcasm classifier.
    """

    def __init__(self, separator: str = _SEPARATOR, max_context_tokens: int = 60):
        self.separator = separator
        self.max_context_tokens = max_context_tokens

    def combine(self, text: str, context: str = "") -> str:
        """
        Concatenate context and text.

        If context is provided:
            "[truncated context] [CTX] [current text]"
        If context is empty:
            "[current text]"

        Args:
            text: The current message.
            context: Previous message(s) from the conversation.

        Returns:
            Combined string ready for the model.
        """
        text = (text or "").strip()
        context = (context or "").strip()

        if not context:
            logger.debug("No context provided — using text only.")
            return text

        # Truncate context to avoid excessive input length
        context_tokens = context.split()
        if len(context_tokens) > self.max_context_tokens:
            context = " ".join(context_tokens[-self.max_context_tokens :])
            logger.debug("Context truncated to %d tokens.", self.max_context_tokens)

        combined = context + self.separator + text
        logger.debug("Combined input (len=%d): %s ...", len(combined), combined[:80])
        return combined

    def combine_batch(
        self, texts: list[str], contexts: list[str] | None = None
    ) -> list[str]:
        """Vectorised version of combine()."""
        if contexts is None:
            contexts = [""] * len(texts)
        return [self.combine(t, c) for t, c in zip(texts, contexts)]


# Module-level convenience instance
_processor = ContextProcessor()


def combine_context(text: str, context: str = "") -> str:
    """Module-level convenience function."""
    return _processor.combine(text, context)


def combine_context_batch(
    texts: list[str], contexts: list[str] | None = None
) -> list[str]:
    """Module-level batch convenience function."""
    return _processor.combine_batch(texts, contexts)


if __name__ == "__main__":
    examples = [
        ("Oh sure, that always works perfectly!", "We tried the same fix three times already."),
        ("I love Mondays.", ""),
        ("हाँ बिलकुल, बहुत अच्छा!", "यह योजना पहले भी फेल हो चुकी है"),
    ]
    proc = ContextProcessor()
    for text, ctx in examples:
        result = proc.combine(text, ctx)
        print(f"Context : {ctx!r}")
        print(f"Text    : {text!r}")
        print(f"Combined: {result!r}")
        print()
