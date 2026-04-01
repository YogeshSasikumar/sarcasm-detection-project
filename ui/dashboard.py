"""
dashboard.py - Professional Streamlit UI for the Sarcasm Detection System
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from utils.config import APP_TITLE, APP_SUBTITLE, APP_ICON, APP_VERSION, EMOTION_LABELS


# ─── Page config (must be first Streamlit call) ─────────────────────────────────
def configure_page():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )


# ─── Custom CSS ─────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main { background: #0A0E1A; }

    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, #1a1f35 0%, #0d1117 50%, #1a1025 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(120deg, #818cf8, #c084fc, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        letter-spacing: 0.08em;
    }
    .version-badge {
        display: inline-block;
        background: rgba(99,102,241,0.15);
        color: #818cf8;
        border: 1px solid rgba(99,102,241,0.4);
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.75rem;
        margin-top: 0.6rem;
    }

    /* Result cards */
    .result-card {
        background: linear-gradient(135deg, #1e2433, #161b2e);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid rgba(99,102,241,0.2);
        margin-bottom: 1rem;
        transition: border-color 0.3s;
    }
    .result-card:hover { border-color: rgba(99,102,241,0.5); }
    .result-label { color: #64748b; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.1em; }
    .result-value { font-size: 1.5rem; font-weight: 700; margin-top: 0.25rem; }
    .result-sarcastic { color: #f87171; }
    .result-not-sarcastic { color: #4ade80; }
    .result-emotion-happy { color: #facc15; }
    .result-emotion-angry { color: #f87171; }
    .result-emotion-sad   { color: #60a5fa; }
    .result-emotion-neutral { color: #94a3b8; }

    /* Reasoning box */
    .reasoning-box {
        background: rgba(30,36,51,0.9);
        border-left: 3px solid #818cf8;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        color: #cbd5e1;
        font-size: 0.9rem;
        line-height: 1.6;
        margin-top: 0.5rem;
    }

    /* Keyword pills */
    .keyword-pill {
        display: inline-block;
        background: rgba(99,102,241,0.15);
        color: #a5b4fc;
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.8rem;
        margin: 2px;
    }
    .keyword-pill-neg {
        background: rgba(248,113,113,0.12);
        color: #fca5a5;
        border-color: rgba(248,113,113,0.3);
    }
    .keyword-pill-marker {
        background: rgba(250,204,21,0.12);
        color: #fde68a;
        border-color: rgba(250,204,21,0.3);
    }

    /* Stat metric card */
    .metric-card {
        background: linear-gradient(135deg, #1e2433, #161b2e);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-number { font-size: 2rem; font-weight: 700; color: #818cf8; }
    .metric-label  { color: #64748b; font-size: 0.8rem; margin-top: 0.2rem; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid rgba(99,102,241,0.15); }

    /* Inputs */
    .stTextArea textarea, .stTextInput input {
        background: #1e2433 !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(99,102,241,0.3) !important;
        border-radius: 8px !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.6rem 2rem !important;
        font-size: 0.95rem !important;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Divider */
    hr { border-color: rgba(99,102,241,0.2) !important; }

    /* Hide default Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)


# ─── Hero Header ────────────────────────────────────────────────────────────────
def render_header():
    st.markdown(f"""
    <div class="hero-header">
        <div class="hero-title">🎭 {APP_TITLE}</div>
        <div class="hero-subtitle">{APP_SUBTITLE}</div>
        <div class="version-badge">v{APP_VERSION}</div>
    </div>
    """, unsafe_allow_html=True)


# ─── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar(history: list) -> dict:
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.markdown("---")

        show_confidence = st.toggle("Show confidence scores", value=True)
        show_keywords   = st.toggle("Show keyword analysis", value=True)
        show_features   = st.toggle("Show top model features", value=False)

        st.markdown("---")
        st.markdown("### 📊 Session Stats")

        total   = len(history)
        sarc    = sum(1 for h in history if h.get("label") == 1)
        not_s   = total - sarc
        c1, c2  = st.columns(2)
        with c1:
            st.metric("Total", total)
            st.metric("Sarcastic", sarc)
        with c2:
            st.metric("Not Sarcastic", not_s)
            if total > 0:
                st.metric("Sarc. Rate", f"{sarc/total*100:.0f}%")

        st.markdown("---")
        st.markdown("### 🌐 Supported Languages")
        st.markdown("🇬🇧 English &nbsp;&nbsp; 🇮🇳 Hindi &nbsp;&nbsp; 🇮🇳 Tamil")

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown(
            "Context-aware sarcasm detection system with "
            "multilingual support, emotion recognition, and "
            "explainable AI predictions."
        )

    return {
        "show_confidence": show_confidence,
        "show_keywords"  : show_keywords,
        "show_features"  : show_features,
    }


# ─── Input Form ─────────────────────────────────────────────────────────────────
def render_input_form() -> tuple[str, str]:
    st.markdown("#### 💬 Input")
    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        text = st.text_area(
            "Current Message",
            placeholder="e.g. Oh great, another Monday!",
            height=120,
            key="input_text",
        )
    with col2:
        context = st.text_area(
            "Conversation Context (optional)",
            placeholder="e.g. I really hate the start of the work week.",
            height=120,
            key="input_context",
        )
    return text.strip(), context.strip()


# ─── Result Cards ────────────────────────────────────────────────────────────────
def render_results(result: dict, settings: dict):
    sarc_class = "result-sarcastic" if result["sarcasm"]["label"] == 1 else "result-not-sarcastic"
    sarc_icon  = "🎭" if result["sarcasm"]["label"] == 1 else "✅"
    emotion    = result["emotion"]["emotion"]
    emo_class  = f"result-emotion-{emotion.lower()}"
    emo_icons  = {"Happy": "😊", "Angry": "😠", "Sad": "😢", "Neutral": "😐"}
    lang_flag  = {"English": "🇬🇧", "Hindi": "🇮🇳", "Tamil": "🇮🇳"}.get(result["language_name"], "🌐")

    st.markdown("#### 🔍 Prediction Results")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Sarcasm</div>
            <div class="result-value {sarc_class}">{sarc_icon} {result['sarcasm']['label_str']}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Emotion</div>
            <div class="result-value {emo_class}">{emo_icons.get(emotion,'🎭')} {emotion}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        conf_pct = result["explanation"]["confidence_pct"]
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Confidence</div>
            <div class="result-value" style="color:#818cf8;">📊 {conf_pct}</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Language</div>
            <div class="result-value" style="color:#38bdf8;">{lang_flag} {result['language_name']}</div>
        </div>""", unsafe_allow_html=True)

    # ── Reasoning ────────────────────────────────────────────────────────────
    st.markdown("#### 🧠 AI Explanation")
    st.markdown(f"""
    <div class="reasoning-box">💡 {result['explanation']['reasoning']}</div>
    """, unsafe_allow_html=True)

    # ── Keywords ─────────────────────────────────────────────────────────────
    if settings["show_keywords"]:
        kw = result["explanation"]["keywords"]
        pills_html = ""
        for w in kw.get("positive", []):
            pills_html += f'<span class="keyword-pill">✨ {w}</span> '
        for w in kw.get("negative", []):
            pills_html += f'<span class="keyword-pill keyword-pill-neg">⚠️ {w}</span> '
        for w in kw.get("sarcasm_markers", []):
            pills_html += f'<span class="keyword-pill keyword-pill-marker">🎯 {w}</span> '
        if pills_html:
            st.markdown(f"**Keywords:** {pills_html}", unsafe_allow_html=True)

    # ── Confidence gauge ─────────────────────────────────────────────────────
    if settings["show_confidence"]:
        conf = result["sarcasm"]["confidence"]
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=conf * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Confidence %", "font": {"color": "#94a3b8", "size": 14}},
            number={"suffix": "%", "font": {"color": "#818cf8", "size": 24}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748b",
                         "tickfont": {"color": "#64748b"}},
                "bar": {"color": "#818cf8"},
                "bgcolor": "#1e2433",
                "bordercolor": "#334155",
                "steps": [
                    {"range": [0, 40],  "color": "#1e2433"},
                    {"range": [40, 70], "color": "#1e2744"},
                    {"range": [70, 100],"color": "#1e2455"},
                ],
                "threshold": {"line": {"color": "#c084fc", "width": 3},
                              "thickness": 0.8, "value": 70},
            }
        ))
        fig.update_layout(
            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            height=220, margin=dict(t=40, b=10, l=30, r=30),
        )
        st.plotly_chart(fig, use_container_width=True, key="gauge_chart")

    # ── Top features ─────────────────────────────────────────────────────────
    if settings["show_features"] and result["explanation"]["top_features"]:
        feats = result["explanation"]["top_features"]
        words  = [f[0] for f in feats]
        weights = [f[1] for f in feats]
        colors  = ["#f87171" if w > 0 else "#4ade80" for w in weights]

        fig2 = go.Figure(go.Bar(
            x=weights, y=words, orientation="h",
            marker_color=colors,
            text=[f"{w:.3f}" for w in weights],
            textposition="outside",
            textfont={"color": "#94a3b8"},
        ))
        fig2.update_layout(
            title={"text": "Top Model Features", "font": {"color": "#94a3b8"}},
            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            xaxis={"color": "#64748b", "showgrid": False},
            yaxis={"color": "#94a3b8", "autorange": "reversed"},
            height=280, margin=dict(t=40, b=20, l=20, r=60),
        )
        st.plotly_chart(fig2, use_container_width=True, key="features_chart")


# ─── Session History Charts ──────────────────────────────────────────────────────
def render_dashboard(history: list):
    if not history:
        st.info("Run some predictions to populate the dashboard 📊")
        return

    df = pd.DataFrame(history)
    st.markdown("---")
    st.markdown("### 📊 Session Analytics Dashboard")

    # ── Summary metrics ───────────────────────────────────────────────────────
    total  = len(df)
    sarc   = int(df["label"].sum())
    not_s  = total - sarc
    avg_cf = df["confidence"].mean() if "confidence" in df.columns else 0

    mc1, mc2, mc3, mc4 = st.columns(4)
    for col, num, lbl in zip(
        [mc1, mc2, mc3, mc4],
        [total, sarc, not_s, f"{avg_cf*100:.1f}%"],
        ["Total Predictions", "Sarcastic", "Not Sarcastic", "Avg Confidence"],
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number">{num}</div>
                <div class="metric-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2, gap="medium")

    # ── Sarcasm donut ─────────────────────────────────────────────────────────
    with col_left:
        fig_donut = go.Figure(go.Pie(
            labels=["Sarcastic", "Not Sarcastic"],
            values=[sarc, not_s],
            hole=0.6,
            marker_colors=["#f87171", "#4ade80"],
            textfont={"color": "white"},
        ))
        fig_donut.update_layout(
            title={"text": "Sarcasm Distribution", "font": {"color": "#94a3b8"}},
            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            legend={"font": {"color": "#94a3b8"}},
            height=300, margin=dict(t=40, b=20),
            annotations=[{"text": f"{sarc}/{total}", "x": 0.5, "y": 0.5,
                          "font": {"size": 20, "color": "#818cf8"}, "showarrow": False}],
        )
        st.plotly_chart(fig_donut, use_container_width=True, key="donut_chart")

    # ── Emotion bar ───────────────────────────────────────────────────────────
    with col_right:
        emo_counts = df["emotion"].value_counts().reindex(EMOTION_LABELS, fill_value=0)
        fig_emo = go.Figure(go.Bar(
            x=emo_counts.index.tolist(),
            y=emo_counts.values.tolist(),
            marker_color=["#facc15", "#f87171", "#60a5fa", "#94a3b8"],
            text=emo_counts.values.tolist(),
            textposition="outside",
            textfont={"color": "#94a3b8"},
        ))
        fig_emo.update_layout(
            title={"text": "Emotion Distribution", "font": {"color": "#94a3b8"}},
            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            xaxis={"color": "#64748b", "showgrid": False},
            yaxis={"color": "#64748b", "showgrid": True, "gridcolor": "#1e2433"},
            height=300, margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_emo, use_container_width=True, key="emotion_chart")

    # ── Confidence over time ───────────────────────────────────────────────────
    if "confidence" in df.columns and len(df) > 1:
        fig_conf = go.Figure()
        fig_conf.add_trace(go.Scatter(
            y=df["confidence"] * 100,
            mode="lines+markers",
            line={"color": "#818cf8", "width": 2},
            marker={"color": "#c084fc", "size": 6},
            name="Confidence %",
        ))
        fig_conf.update_layout(
            title={"text": "Prediction Confidence Over Time", "font": {"color": "#94a3b8"}},
            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            xaxis={"color": "#64748b", "showgrid": False, "title": "Prediction #"},
            yaxis={"color": "#64748b", "showgrid": True, "gridcolor": "#1e2433",
                   "title": "Confidence %", "range": [0, 105]},
            height=260, margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig_conf, use_container_width=True, key="confidence_chart")

    # ── Language breakdown ─────────────────────────────────────────────────────
    if "language" in df.columns:
        lang_counts = df["language"].value_counts()
        fig_lang = go.Figure(go.Pie(
            labels=lang_counts.index.tolist(),
            values=lang_counts.values.tolist(),
            marker_colors=["#38bdf8", "#818cf8", "#c084fc"],
            textfont={"color": "white"},
        ))
        fig_lang.update_layout(
            title={"text": "Language Breakdown", "font": {"color": "#94a3b8"}},
            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            legend={"font": {"color": "#94a3b8"}},
            height=260, margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_lang, use_container_width=True, key="lang_chart")

    # ── Prediction history table ───────────────────────────────────────────────
    with st.expander("📋 Full Prediction History"):
        display_cols = ["text", "context", "label_str", "emotion", "language", "confidence"]
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(
            df[display_cols].rename(columns={
                "label_str": "Sarcasm", "text": "Text",
                "context": "Context", "emotion": "Emotion",
                "language": "Language", "confidence": "Confidence",
            }),
            use_container_width=True,
            hide_index=True,
        )


def render_model_metrics(metrics: dict):
    """Render model performance metrics panel."""
    if not metrics:
        return
    with st.expander("🏆 Model Performance Metrics"):
        sc = metrics.get("sarcasm", {})
        em = metrics.get("emotion", {})
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Sarcasm Model**")
            if sc:
                st.metric("Accuracy",  f"{sc.get('accuracy',0)*100:.1f}%")
                st.metric("F1-Score",  f"{sc.get('f1_score',0)*100:.1f}%")
                st.metric("Precision", f"{sc.get('precision',0)*100:.1f}%")
                st.metric("Recall",    f"{sc.get('recall',0)*100:.1f}%")
        with col2:
            st.markdown("**Emotion Model**")
            if em:
                st.metric("Accuracy",  f"{em.get('accuracy',0)*100:.1f}%")
                st.metric("F1-Score",  f"{em.get('f1_score',0)*100:.1f}%")
                st.metric("Precision", f"{em.get('precision',0)*100:.1f}%")
                st.metric("Recall",    f"{em.get('recall',0)*100:.1f}%")
