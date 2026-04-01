# Context-Aware Multilingual Sarcasm Detection System

> **Multilingual · Explainable AI · Emotion-Aware · Streamlit Dashboard**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Overview

An end-to-end NLP system that:

- 🎭 **Detects sarcasm** using context-aware conversation history
- 🌐 **Supports multilingual input** — English, Hindi, Tamil
- 🧠 **Explains predictions** with keywords, confidence scores, and reasoning
- 😊 **Detects emotions** — Happy, Angry, Sad, Neutral
- 📊 **Real-time Streamlit dashboard** with charts and session analytics

---

## 🗂️ Project Structure

```
sarcasm_detection_project/
├── data/
│   ├── sample_dataset.csv          # Auto-generated synthetic dataset
│   └── generate_dataset.py         # Dataset generation script
├── models/
│   ├── sarcasm_model.pkl           # Trained sarcasm classifier
│   └── emotion_model.pkl           # Trained emotion classifier
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # CSV loading + validation
│   ├── preprocessing.py            # Text cleaning pipeline
│   ├── context_processor.py        # Context + message merger
│   ├── sarcasm_model.py            # TF-IDF + LogReg sarcasm model
│   ├── emotion_model.py            # TF-IDF + SGD emotion model
│   ├── explainability.py           # XAI reasoning engine
│   ├── language_detection.py       # langdetect wrapper
│   └── evaluation.py               # Metrics + confusion matrix
├── ui/
│   ├── __init__.py
│   └── dashboard.py                # Streamlit UI components
├── utils/
│   ├── __init__.py
│   ├── config.py                   # Central configuration
│   └── logger.py                   # Rotating file logger
├── main.py                         # ⬅ Streamlit entry point
├── train_models.py                 # One-shot model training script
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start (Local)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/sarcasm-detection-system.git
cd sarcasm-detection-system/sarcasm_detection_project
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# .\venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train models (first-time only)

> **Note:** The app auto-trains models on first launch if they are missing.
> You can also train manually:

```bash
python train_models.py
```

### 5. Run the application

```bash
streamlit run main.py
```

The app opens at `http://localhost:8501`

---

## ☁️ Streamlit Cloud Deployment

### Step 1 — Push to GitHub

```bash
git add .
git commit -m "Initial release: Sarcasm Detection System"
git push origin main
```

### Step 2 — Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Select your GitHub repository
4. Set **Main file path** to: `sarcasm_detection_project/main.py`
5. Click **Deploy**

> ⚠️ The app auto-trains models on first deployment. This takes ~30 seconds.

---

## 🔧 Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| ML Framework | scikit-learn 1.5 |
| NLP | nltk, langdetect |
| Visualisation | Plotly, Seaborn, Matplotlib |
| UI | Streamlit 1.35 |
| Deployment | Streamlit Cloud |

---

## 📊 Model Architecture

### Sarcasm Detection
- **Feature extraction:** TF-IDF with word (1–3) + char (2–4) n-gram FeatureUnion
- **Classifier:** Logistic Regression (L2, C=1.0, lbfgs)
- **Input:** Context-concatenated, preprocessed text

### Emotion Detection
- **Feature extraction:** TF-IDF (1–2 word n-grams)
- **Classifier:** SGD Classifier (log-loss for calibrated probabilities)
- **Classes:** Happy, Angry, Sad, Neutral

---

## 🌐 Supported Languages

| Language | Code | Notes |
|----------|------|-------|
| English  | `en` | Full preprocessing (stopword + lemmatization) |
| Hindi    | `hi` | Noise removal only |
| Tamil    | `ta` | Noise removal only |

---

## 💡 System Features

| Feature | Status |
|---------|--------|
| Context-aware sarcasm detection | ✅ |
| Multilingual support (EN/HI/TA) | ✅ |
| Explainable AI (XAI) | ✅ |
| Emotion detection | ✅ |
| Confidence scoring + gauge chart | ✅ |
| Real-time session analytics dashboard | ✅ |
| Prediction history table | ✅ |
| Auto model training on first run | ✅ |
| Streamlit Cloud deployment ready | ✅ |

---

## 📄 License

MIT © 2024 — Free to use and modify.
