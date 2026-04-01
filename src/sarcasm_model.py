"""
sarcasm_model.py - TF-IDF + Logistic Regression sarcasm classifier
                   with optional BERT-based feature enhancement.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from utils.config import SARCASM_MODEL_PATH, RANDOM_STATE, TEST_SIZE
from utils.logger import get_logger

logger = get_logger(__name__)


class SarcasmDetector:
    """
    Context-aware sarcasm classifier.

    Pipeline:
        TF-IDF (char + word n-grams)  →  Logistic Regression

    The model is trained on the combined (context + text) string produced
    by context_processor.py.
    """

    def __init__(self, model_path: str = SARCASM_MODEL_PATH):
        self.model_path = model_path
        self.pipeline: Pipeline | None = None
        self._is_trained = False

    # ── Build ─────────────────────────────────────────────────────────────────
    def _build_pipeline(self) -> Pipeline:
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            max_features=30_000,
            sublinear_tf=True,
            min_df=1,
        )
        # Char n-gram vectorizer for catching sarcastic punctuation patterns
        char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=10_000,
            sublinear_tf=True,
            min_df=1,
        )

        from sklearn.pipeline import FeatureUnion

        combined = FeatureUnion([
            ("word", vectorizer),
            ("char", char_vectorizer),
        ])

        clf = LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )
        return Pipeline([("features", combined), ("clf", clf)])

    # ── Train ─────────────────────────────────────────────────────────────────
    def train(self, texts: list[str], labels: list[int]) -> dict:
        """
        Train the sarcasm detection pipeline.

        Args:
            texts : List of combined (context + text) strings.
            labels: Binary labels (0 = not sarcastic, 1 = sarcastic).

        Returns:
            Dict with train/test accuracy.
        """
        logger.info("Training SarcasmDetector on %d samples …", len(texts))
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
        )
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_train, y_train)
        self._is_trained = True

        train_acc = self.pipeline.score(X_train, y_train)
        test_acc  = self.pipeline.score(X_test, y_test)
        logger.info("Train acc: %.4f | Test acc: %.4f", train_acc, test_acc)

        report = classification_report(
            y_test, self.pipeline.predict(X_test),
            target_names=["Not Sarcastic", "Sarcastic"], output_dict=True
        )
        self.save()
        return {"train_accuracy": train_acc, "test_accuracy": test_acc, "report": report}

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict(self, text: str) -> dict:
        """
        Predict sarcasm for a single combined-context string.

        Returns:
            {
                "label": int,
                "label_str": str,
                "confidence": float,
                "probabilities": {"Not Sarcastic": float, "Sarcastic": float}
            }
        """
        self._check_trained()
        proba = self.pipeline.predict_proba([text])[0]
        label = int(np.argmax(proba))
        return {
            "label": label,
            "label_str": "Sarcastic" if label == 1 else "Not Sarcastic",
            "confidence": float(round(proba[label], 4)),
            "probabilities": {
                "Not Sarcastic": float(round(proba[0], 4)),
                "Sarcastic":     float(round(proba[1], 4)),
            },
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Batch predict."""
        return [self.predict(t) for t in texts]

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self) -> None:
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)
        logger.info("Sarcasm model saved → %s", self.model_path)

    def load(self) -> "SarcasmDetector":
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Sarcasm model not found at {self.model_path}. Run train_models.py first."
            )
        self.pipeline = joblib.load(self.model_path)
        self._is_trained = True
        logger.info("Sarcasm model loaded from %s", self.model_path)
        return self

    def _check_trained(self):
        if not self._is_trained or self.pipeline is None:
            raise RuntimeError("Model not trained/loaded. Call train() or load() first.")

    # ── Top features ──────────────────────────────────────────────────────────
    def get_top_features(self, text: str, n: int = 10) -> list[tuple[str, float]]:
        """
        Return top-n influential word features for a given text.
        Used by the explainability module.
        """
        self._check_trained()
        try:
            word_vec = self.pipeline.named_steps["features"].transformer_list[0][1]
            features = word_vec.get_feature_names_out()
            log_proba = self.pipeline.named_steps["clf"].coef_[0]

            from sklearn.pipeline import FeatureUnion
            fu: FeatureUnion = self.pipeline.named_steps["features"]
            X = fu.transform([text])

            # Only take the word-tfidf part
            n_word = len(features)
            word_weights = X[0, :n_word].toarray().flatten() * log_proba[:n_word]
            top_idx = np.argsort(np.abs(word_weights))[-n:][::-1]
            return [(features[i], float(round(word_weights[i], 4))) for i in top_idx]
        except Exception:
            return []


if __name__ == "__main__":
    from src.data_loader import load_data
    from src.preprocessing import preprocess
    from src.context_processor import combine_context

    df = load_data()
    combined = [combine_context(r["text"], r["context"]) for _, r in df.iterrows()]
    processed = [preprocess(c) for c in combined]

    detector = SarcasmDetector()
    metrics = detector.train(processed, df["label"].tolist())
    print(f"\n✅ Training complete!")
    print(f"   Train Acc : {metrics['train_accuracy']:.4f}")
    print(f"   Test Acc  : {metrics['test_accuracy']:.4f}")

    sample = "Oh great, another Monday! [CTX] I hate the start of the week"
    result = detector.predict(preprocess(sample))
    print(f"\n🔍 Sample prediction: {result}")
