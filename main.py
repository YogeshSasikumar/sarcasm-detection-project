"""
main.py - Streamlit entry point for the Sarcasm Detection System.
Run: streamlit run main.py
"""

import sys
import os

# ─── Path setup ─────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import streamlit as st

# Import UI components first (sets page config)
from ui.dashboard import (
    configure_page, inject_css, render_header,
    render_sidebar, render_input_form,
    render_results, render_dashboard, render_model_metrics,
)

# ─── Configure page (MUST be before any other st call) ──────────────────────────
configure_page()
inject_css()

# ─── Lazy imports (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔧 Loading models…")
def load_models():
    """Load sarcasm + emotion models. Train if not present."""
    from src.sarcasm_model import SarcasmDetector
    from src.emotion_model import EmotionDetector
    from utils.config import SARCASM_MODEL_PATH, EMOTION_MODEL_PATH
    from src.data_loader import load_data
    from src.preprocessing import preprocess
    from src.context_processor import combine_context

    sarc_model = SarcasmDetector()
    emo_model  = EmotionDetector()

    # Auto-train if models don't exist
    if not (os.path.exists(SARCASM_MODEL_PATH) and os.path.exists(EMOTION_MODEL_PATH)):
        st.toast("⚙️ First run: training models on synthetic dataset…", icon="🤖")
        df = load_data()
        combined = [combine_context(r["text"], r["context"]) for _, r in df.iterrows()]
        processed = [preprocess(t, r["language"]) for t, (_, r) in zip(combined, df.iterrows())]
        sarc_model.train(processed, df["label"].tolist())

        emo_texts = [preprocess(t, l) for t, l in zip(df["text"], df["language"])]
        emo_model.train(emo_texts, df["emotion"].tolist())
    else:
        sarc_model.load()
        emo_model.load()

    return sarc_model, emo_model


@st.cache_data(show_spinner=False)
def get_eval_metrics():
    """Compute and cache evaluation metrics on the training dataset."""
    from src.data_loader import load_data
    from src.preprocessing import preprocess
    from src.context_processor import combine_context
    from src.evaluation import compute_metrics

    try:
        sarc_model, emo_model = load_models()
        df = load_data()
        combined  = [combine_context(r["text"], r["context"]) for _, r in df.iterrows()]
        processed = [preprocess(t, r["language"]) for t, (_, r) in zip(combined, df.iterrows())]

        sarc_preds = [sarc_model.predict(t)["label"] for t in processed]
        sarc_m = compute_metrics(df["label"].tolist(), sarc_preds)

        emo_texts = [preprocess(t, l) for t, l in zip(df["text"], df["language"])]
        emo_preds = [emo_model.predict(t)["emotion"] for t in emo_texts]
        emo_m = compute_metrics(df["emotion"].tolist(), emo_preds)

        return {"sarcasm": sarc_m, "emotion": emo_m}
    except Exception:
        return {}


# ─── Session-state history ───────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


def run_prediction(text: str, context: str, sarc_model, emo_model) -> dict:
    """Run full prediction pipeline on user input."""
    from src.preprocessing import preprocess
    from src.context_processor import combine_context
    from src.language_detection import detect_and_route
    from src.explainability import explain

    # Language detection
    lang_info = detect_and_route(text)
    lang_code = lang_info["language_code"]

    # Preprocess
    combined   = combine_context(text, context)
    processed  = preprocess(combined, lang_code)
    proc_text  = preprocess(text, lang_code)

    # Sarcasm prediction
    sarc_result = sarc_model.predict(processed)

    # Top features for XAI
    try:
        top_feats = sarc_model.get_top_features(processed, n=8)
    except Exception:
        top_feats = []

    # Emotion prediction
    emo_result = emo_model.predict(proc_text)

    # Explanation
    explanation = explain(text, context, sarc_result, top_feats)

    return {
        "text"         : text,
        "context"      : context,
        "label"        : sarc_result["label"],
        "label_str"    : sarc_result["label_str"],
        "confidence"   : sarc_result["confidence"],
        "sarcasm"      : sarc_result,
        "emotion"      : emo_result,
        "emotion_str"  : emo_result["emotion"],
        "language_code": lang_code,
        "language_name": lang_info["language_name"],
        "explanation"  : explanation,
    }


# ─── Main App ────────────────────────────────────────────────────────────────────
def main():
    render_header()

    # Load models (cached)
    with st.spinner("Loading AI models…"):
        sarc_model, emo_model = load_models()

    # Sidebar
    settings = render_sidebar(st.session_state.history)

    # ── Input section ────────────────────────────────────────────────────────
    text, context = render_input_form()

    col_btn, col_clear = st.columns([3, 1])
    with col_btn:
        predict_clicked = st.button("🎭 Analyse Text", key="predict_btn", type="primary")
    with col_clear:
        if st.button("🗑️ Clear History", key="clear_btn"):
            st.session_state.history = []
            st.rerun()

    # ── Prediction ───────────────────────────────────────────────────────────
    if predict_clicked:
        if not text:
            st.warning("⚠️ Please enter some text to analyse.")
        else:
            with st.spinner("🧠 Analysing…"):
                try:
                    result = run_prediction(text, context, sarc_model, emo_model)
                    st.session_state.history.append(result)
                    st.markdown("---")
                    render_results(result, settings)
                except Exception as exc:
                    st.error(f"❌ Prediction failed: {exc}")
                    st.exception(exc)
    elif st.session_state.history:
        # Show last result if no new prediction
        st.markdown("---")
        render_results(st.session_state.history[-1], settings)

    # ── Quick examples ───────────────────────────────────────────────────────
    with st.expander("💡 Try example inputs"):
        examples = [
            ("Oh great, another Monday!", "I hate the start of the week", "English sarcasm"),
            ("I really enjoyed the movie last night.", "We watched it together.", "Sincere statement"),
            ("वाह, क्या शानदार काम किया!", "उसने फिर से गड़बड़ी की", "Hindi sarcasm"),
            ("ஆமா, இது மிகவும் சிறந்த யோசனை!", "அவர் மீண்டும் தவறான முடிவு எடுத்தார்", "Tamil sarcasm"),
            ("Yeah, because that always works perfectly.", "He tried the same failed approach again.", "Ironic"),
        ]
        for ex_text, ex_ctx, label in examples:
            if st.button(f"📝 {label}: {ex_text[:40]}…" if len(ex_text) > 40 else f"📝 {label}: {ex_text}", key=f"ex_{label}"):
                st.session_state["input_text"]    = ex_text
                st.session_state["input_context"] = ex_ctx
                st.rerun()

    # ── Dashboard ────────────────────────────────────────────────────────────
    render_dashboard(st.session_state.history)

    # ── Model performance ────────────────────────────────────────────────────
    metrics = get_eval_metrics()
    render_model_metrics(metrics)


if __name__ == "__main__":
    main()
