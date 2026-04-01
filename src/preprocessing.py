"""
preprocessing.py - Text cleaning and preprocessing pipeline
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from utils.logger import get_logger

logger = get_logger(__name__)

# Download required NLTK data (silent if already present)
for _pkg in ["stopwords", "wordnet", "punkt", "punkt_tab", "omw-1.4"]:
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass

_LEMMATIZER = WordNetLemmatizer()
_STOP_WORDS = set(stopwords.words("english"))

# Keep negation words — they are sarcasm signals
_KEEP_WORDS = {"not", "no", "never", "neither", "nor", "nothing", "nobody",
               "nowhere", "n't", "cannot", "won't", "can't", "don't",
               "isn't", "wasn't", "weren't", "haven't", "hadn't"}
_STOP_WORDS -= _KEEP_WORDS


def remove_noise(text: str) -> str:
    """Remove URLs, mentions, hashtags, special characters, and extra spaces."""
    text = re.sub(r"http\S+|www\S+", " ", text)          # URLs
    text = re.sub(r"@\w+", " ", text)                     # @mentions
    text = re.sub(r"#\w+", " ", text)                     # hashtags
    text = re.sub(r"[^\w\s.,!?']", " ", text)             # special chars
    text = re.sub(r"\s+", " ", text).strip()              # extra spaces
    return text


def tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer (avoids heavy spacy on cloud)."""
    return text.split()


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove stopwords while keeping negation tokens."""
    return [t for t in tokens if t not in _STOP_WORDS]


def lemmatize(tokens: list[str]) -> list[str]:
    """Lemmatize each token."""
    return [_LEMMATIZER.lemmatize(t) for t in tokens]


def preprocess(text: str, language: str = "en") -> str:
    """
    Full preprocessing pipeline.

    Steps:
        1. Lowercase
        2. Remove noise (URLs, mentions, etc.)
        3. Tokenize
        4. Remove stopwords (English only)
        5. Lemmatize (English only)
        6. Rejoin tokens

    Args:
        text: Raw input text.
        language: ISO language code ('en', 'hi', 'ta').

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = remove_noise(text)

    if language == "en":
        tokens = tokenize(text)
        tokens = remove_stopwords(tokens)
        tokens = lemmatize(tokens)
        text = " ".join(tokens)
    else:
        # For non-English: only lowercase + noise removal (no NLTK stopwords)
        text = re.sub(r"\s+", " ", text).strip()

    logger.debug("Preprocessed [%s]: %s", language, text[:80])
    return text


def preprocess_batch(texts: list[str], languages: list[str] | None = None) -> list[str]:
    """Preprocess a list of texts."""
    if languages is None:
        languages = ["en"] * len(texts)
    return [preprocess(t, l) for t, l in zip(texts, languages)]


if __name__ == "__main__":
    samples = [
        ("Oh great, another Monday! This is just wonderful.", "en"),
        ("वाह, क्या शानदार काम किया!", "hi"),
        ("ஆமா, இது மிகவும் சிறந்த யோசனை!", "ta"),
    ]
    for text, lang in samples:
        print(f"[{lang}] Input  : {text}")
        print(f"[{lang}] Output : {preprocess(text, lang)}")
        print()
