"""
language_detection.py - Detect language and route to correct pipeline
"""

from utils.config import SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    from langdetect import detect, LangDetectException
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False
    logger.warning("langdetect not installed — defaulting to 'en'.")


def detect_language(text: str) -> str:
    """
    Detect the ISO 639-1 language code of the input text.

    Supported: en (English), hi (Hindi), ta (Tamil)
    Falls back to DEFAULT_LANGUAGE if detection fails or language unsupported.

    Args:
        text: Input text string.

    Returns:
        Language code string (e.g. 'en', 'hi', 'ta').
    """
    if not text or not text.strip():
        logger.debug("Empty text — returning default language.")
        return DEFAULT_LANGUAGE

    if not _LANGDETECT_AVAILABLE:
        return DEFAULT_LANGUAGE

    try:
        detected = detect(text)
        logger.debug("Detected language: %s", detected)

        if detected in SUPPORTED_LANGUAGES:
            return detected

        # Some detectors emit 'zh-cn', 'pt', etc. — map to default
        logger.info(
            "Unsupported language '%s' detected — falling back to '%s'.",
            detected,
            DEFAULT_LANGUAGE,
        )
        return DEFAULT_LANGUAGE

    except LangDetectException as exc:
        logger.warning("Language detection failed: %s — using default.", exc)
        return DEFAULT_LANGUAGE


def get_language_name(code: str) -> str:
    """Return human-readable language name for a given ISO code."""
    return SUPPORTED_LANGUAGES.get(code, "English")


def is_supported(code: str) -> bool:
    """Return True if the language code is in the supported set."""
    return code in SUPPORTED_LANGUAGES


def detect_and_route(text: str) -> dict:
    """
    Detect the language and return a routing dict.

    Returns:
        {
            "language_code": str,
            "language_name": str,
            "is_supported": bool,
            "pipeline": str   # 'english' | 'hindi' | 'tamil' | 'generic'
        }
    """
    code = detect_language(text)
    pipeline_map = {"en": "english", "hi": "hindi", "ta": "tamil"}
    return {
        "language_code": code,
        "language_name": get_language_name(code),
        "is_supported": is_supported(code),
        "pipeline": pipeline_map.get(code, "generic"),
    }


if __name__ == "__main__":
    test_phrases = [
        "Oh great, another Monday! This is just wonderful.",
        "वाह, क्या शानदार काम किया!",
        "ஆமா, இது மிகவும் சிறந்த யோசனை!",
        "Bonjour tout le monde",
        "",
    ]
    for phrase in test_phrases:
        result = detect_and_route(phrase)
        print(f"Text    : {phrase[:50]!r}")
        print(f"Detected: {result['language_name']} ({result['language_code']}) → {result['pipeline']}")
        print()
