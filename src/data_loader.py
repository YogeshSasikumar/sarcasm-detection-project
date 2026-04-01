import os
import gdown
import pandas as pd
from utils.config import DATASET_PATH, REQUIRED_COLUMNS
from utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Loads and validates the sarcasm dataset from CSV."""

    def __init__(self, path: str = DATASET_PATH):
        self.path = path
        self.df: pd.DataFrame = pd.DataFrame()

    def load(self) -> pd.DataFrame:
        """Load dataset from CSV, validate columns, and handle missing values."""
        
        if not os.path.exists(self.path):
            logger.info("Dataset not found at %s. Attempting to download from Google Drive...", self.path)
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            url = f'https://drive.google.com/uc?id=1cGOJlV_snlmTI5YZ3iNH066ph5M_3bEO'
            try:
                gdown.download(url, self.path, quiet=False)
            except Exception as e:
                logger.error("Failed to download dataset from Google Drive: %s", str(e))
                raise

        logger.info("Loading dataset from: %s", self.path)

        try:
            self.df = pd.read_csv(self.path)
        except Exception as e:
            logger.error("Error reading CSV file: %s", str(e))
            raise

        logger.info("Raw dataset shape: %s", self.df.shape)

        # Validate required columns
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Drop rows where text or label is null
        before = len(self.df)
        self.df = self.df.dropna(subset=["text", "label"])
        dropped = before - len(self.df)
        if dropped:
            logger.warning("Dropped %d rows with null text/label", dropped)

        # Fill missing context with empty string
        self.df["context"] = self.df["context"].fillna("")

        # Fill missing language with 'en'
        self.df["language"] = self.df["language"].fillna("en")

        # Fill missing emotion with 'Neutral'
        self.df["emotion"] = self.df["emotion"].fillna("Neutral")

        # Ensure label is integer
        self.df["label"] = self.df["label"].astype(int)

        logger.info("Clean dataset shape: %s", self.df.shape)
        return self.df

    def preview(self, n: int = 5) -> pd.DataFrame:
        """Return the first n rows of the loaded dataset."""
        if self.df.empty:
            logger.warning("Dataset not loaded yet. Call load() first.")
        return self.df.head(n)

    def get_stats(self) -> dict:
        """Return high-level statistics about the dataset."""
        if self.df.empty:
            return {}
        return {
            "total_rows": len(self.df),
            "sarcastic": int(self.df["label"].sum()),
            "not_sarcastic": int((self.df["label"] == 0).sum()),
            "languages": self.df["language"].value_counts().to_dict(),
            "emotions": self.df["emotion"].value_counts().to_dict(),
        }


def load_data(path: str = DATASET_PATH) -> pd.DataFrame:
    """Convenience function: load and return a clean dataframe."""
    loader = DataLoader(path)
    return loader.load()


if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load()
    print("\n📊 Dataset Preview:")
    print(loader.preview())
    print("\n📈 Dataset Stats:")
    for k, v in loader.get_stats().items():
        print(f"  {k}: {v}")
