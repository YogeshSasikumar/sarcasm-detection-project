"""
emotion_model.py - Multi-class emotion detection classifier
Emotions: Happy, Angry, Sad, Neutral
"""

import os
import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from utils.config import EMOTION_MODEL_PATH, EMOTION_LABELS, RANDOM_STATE, TEST_SIZE
from utils.logger import get_logger

logger = get_logger(__name__)


class EmotionDetector:
    """
    Multi-class emotion classifier.

    Pipeline: TF-IDF (word n-grams, 1-2) → SGD classifier (log loss = probabilities)

    Detects: Happy, Angry, Sad, Neutral
    """

    def __init__(self, model_path: str = EMOTION_MODEL_PATH):
        self.model_path = model_path
        self.pipeline: Pipeline | None = None
        self._is_trained = False
        self.classes_ = EMOTION_LABELS

    # ── Build ─────────────────────────────────────────────────────────────────
    def _build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=20_000,
                sublinear_tf=True,
                min_df=1,
            )),
            ("clf", SGDClassifier(
                loss="log_loss",
                max_iter=1000,
                random_state=RANDOM_STATE,
                tol=1e-3,
                n_jobs=-1,
            )),
        ])

    # ── Train ─────────────────────────────────────────────────────────────────
    def train(self, texts: list[str], emotions: list[str]) -> dict:
        """
        Train the emotion detection pipeline.

        Args:
            texts   : Preprocessed text strings.
            emotions: Emotion labels ('Happy', 'Angry', 'Sad', 'Neutral').

        Returns:
            Dict with train/test accuracy.
        """
        logger.info("Training EmotionDetector on %d samples …", len(texts))
        X_train, X_test, y_train, y_test = train_test_split(
            texts, emotions,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=emotions,
        )
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_train, y_train)
        self._is_trained = True
        self.classes_ = list(self.pipeline.classes_)

        train_acc = self.pipeline.score(X_train, y_train)
        test_acc  = self.pipeline.score(X_test, y_test)
        logger.info("Emotion Train acc: %.4f | Test acc: %.4f", train_acc, test_acc)

        report = classification_report(
            y_test, self.pipeline.predict(X_test),
            target_names=self.classes_, output_dict=True, zero_division=0
        )
        self.save()
        return {"train_accuracy": train_acc, "test_accuracy": test_acc, "report": report}

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict(self, text: str) -> dict:
        """
        Predict emotion for a single text.

        Returns:
            {
                "emotion": str,
                "confidence": float,
                "all_scores": {emotion: probability}
            }
        """
        self._check_trained()
        proba = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        idx = int(np.argmax(proba))
        return {
            "emotion": classes[idx],
            "confidence": float(round(proba[idx], 4)),
            "all_scores": {c: float(round(p, 4)) for c, p in zip(classes, proba)},
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(t) for t in texts]

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self) -> None:
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)
        logger.info("Emotion model saved → %s", self.model_path)

    def load(self) -> "EmotionDetector":
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Emotion model not found at {self.model_path}. Run train_models.py first."
            )
        self.pipeline = joblib.load(self.model_path)
        self._is_trained = True
        self.classes_ = list(self.pipeline.classes_)
        logger.info("Emotion model loaded from %s", self.model_path)
        return self

    def _check_trained(self):
        if not self._is_trained or self.pipeline is None:
            raise RuntimeError("Model not trained/loaded. Call train() or load() first.")


if __name__ == "__main__":
    from src.data_loader import load_data
    from src.preprocessing import preprocess

    df = load_data()
    texts    = df["text"].apply(preprocess).tolist()
    emotions = df["emotion"].tolist()

    detector = EmotionDetector()
    metrics  = detector.train(texts, emotions)
    print(f"\n✅ Emotion model trained!")
    print(f"   Train Acc : {metrics['train_accuracy']:.4f}")
    print(f"   Test Acc  : {metrics['test_accuracy']:.4f}")

    result = detector.predict(preprocess("I am so happy and excited today!"))
    print(f"\n🎭 Sample: {result}")
