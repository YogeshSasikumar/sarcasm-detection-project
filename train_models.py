"""
train_models.py - One-shot script to generate dataset and train both models.
Run: python train_models.py
"""

import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import get_logger
logger = get_logger("train_models")

def main():
    logger.info("=" * 60)
    logger.info("  Sarcasm Detection System — Model Training")
    logger.info("=" * 60)

    # ── Step 1: Generate dataset ──────────────────────────────────────────────
    logger.info("[1/4] Generating synthetic dataset …")
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    import importlib, runpy
    runpy.run_path("data/generate_dataset.py")

    # ── Step 2: Load data ─────────────────────────────────────────────────────
    logger.info("[2/4] Loading dataset …")
    from src.data_loader import load_data
    from src.preprocessing import preprocess
    from src.context_processor import combine_context

    df = load_data()
    logger.info("  Loaded %d rows", len(df))

    # ── Step 3: Train sarcasm model ───────────────────────────────────────────
    logger.info("[3/4] Training sarcasm detection model …")
    from src.sarcasm_model import SarcasmDetector

    combined_texts = [
        combine_context(row["text"], row["context"]) for _, row in df.iterrows()
    ]
    processed_texts = [preprocess(t, row["language"]) for t, (_, row) in zip(combined_texts, df.iterrows())]

    sarcasm_detector = SarcasmDetector()
    sarcasm_metrics = sarcasm_detector.train(processed_texts, df["label"].tolist())
    logger.info("  Sarcasm — Train: %.4f | Test: %.4f",
                sarcasm_metrics["train_accuracy"], sarcasm_metrics["test_accuracy"])

    # ── Step 4: Train emotion model ───────────────────────────────────────────
    logger.info("[4/4] Training emotion detection model …")
    from src.emotion_model import EmotionDetector

    emotion_texts = [preprocess(t, lang) for t, lang in zip(df["text"], df["language"])]
    emotion_detector = EmotionDetector()
    emotion_metrics = emotion_detector.train(emotion_texts, df["emotion"].tolist())
    logger.info("  Emotion — Train: %.4f | Test: %.4f",
                emotion_metrics["train_accuracy"], emotion_metrics["test_accuracy"])

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("✅  Training Complete!")
    logger.info("   Sarcasm Model  — Test Accuracy : %.4f", sarcasm_metrics["test_accuracy"])
    logger.info("   Emotion Model  — Test Accuracy : %.4f", emotion_metrics["test_accuracy"])
    logger.info("   Models saved in: models/")
    logger.info("=" * 60)

    print("\n" + "=" * 60)
    print("  ✅  All models trained and saved successfully!")
    print(f"  📊  Sarcasm Test Accuracy : {sarcasm_metrics['test_accuracy']:.4f}")
    print(f"  🎭  Emotion  Test Accuracy : {emotion_metrics['test_accuracy']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
