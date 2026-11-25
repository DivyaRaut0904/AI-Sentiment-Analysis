"""Modern Streamlit interface for customer sentiment analysis."""

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from src.models.bilstm_model import BiLSTMClassifier
from src.models.distilbert_model import load_distilbert
from src.preprocessing.text_preprocessing import Vocabulary
from src.utils import LABEL_NAMES, MODEL_DIR, OUTPUT_DIR, PROJECT_ROOT


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');
        html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
        .stApp { background: #f4f6fb; color: #172033; }
        [data-testid="stHeader"] { background: #f4f6fb; }
        [data-testid="stAppViewContainer"] .main .block-container {
            max-width: 1440px;
            padding: 2rem 2.5rem 3rem;
        }
        h1, h2, h3, h4, p, label, .stMarkdown, .stText, [data-testid="stCaptionContainer"] {
            color: #172033;
        }
        h1 { font-weight: 800 !important; }
        [data-testid="stCaptionContainer"] { color: #536078 !important; }
        .stMarkdown p { color: #344158; font-size: 15px; line-height: 1.55; }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff !important;
            border: 1px solid #dce3ee !important;
            border-radius: 18px !important;
            box-shadow: 0 8px 24px rgba(23,32,51,.05);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #111827 0%, #1e1b4b 100%);
            border-right: 1px solid rgba(255,255,255,.08);
        }
        [data-testid="stSidebar"] * { color: #e8eaf6 !important; }
        [data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem 2rem; }
        [data-testid="stSidebarNav"] { display: none; }
        [data-testid="stRadio"] label {
            border: 1px solid transparent;
            border-radius: 12px;
            margin: 4px 0;
            padding: 9px 10px;
        }
        [data-testid="stRadio"] label:hover {
            background: rgba(139,92,246,.16);
            border-color: rgba(167,139,250,.25);
        }
        [data-testid="stRadio"] label[data-selected="true"] {
            background: linear-gradient(135deg, #7c3aed, #4f46e5);
            box-shadow: 0 8px 22px rgba(79,70,229,.28);
        }
        [data-testid="stRadio"] label p { color: #eef0ff !important; }
        .hero {
            position: relative;
            overflow: hidden;
            border-radius: 24px;
            padding: 42px 44px;
            margin: 0 0 1.2rem;
            background: linear-gradient(135deg, #111827 0%, #312e81 55%, #6d28d9 100%);
            color: white;
            box-shadow: 0 18px 44px rgba(49,46,129,.18);
        }
        .hero:after {
            content: "";
            position: absolute;
            width: 190px;
            height: 190px;
            right: -60px;
            top: -80px;
            border-radius: 50%;
            background: linear-gradient(135deg, rgba(34,211,238,.85), rgba(244,114,182,.55));
            filter: blur(2px);
            pointer-events: none;
        }
        .hero h1 { color: white !important; font-size: clamp(30px, 3.2vw, 44px); line-height: 1.12; margin: 12px 0; }
        .hero p { color: #eef2ff !important; max-width: 740px; font-size: 16px; line-height: 1.65; margin: 0; }
        .eyebrow {
            display: inline-block;
            border: 1px solid rgba(255,255,255,.24);
            background: rgba(255,255,255,.10);
            border-radius: 999px;
            padding: 7px 12px;
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 1.2px;
        }
        .metric-card {
            min-height: 132px;
            border: 1px solid #edf0f7;
            border-radius: 18px;
            background: #ffffff;
            padding: 20px;
            box-shadow: 0 10px 26px rgba(40,47,75,.08);
        }
        .metric-icon {
            float: right;
            display: grid;
            place-items: center;
            width: 44px;
            height: 44px;
            border-radius: 14px;
            font-size: 22px;
            background: #ede9fe;
        }
        .metric-card small { color: #59647a; font-weight: 800; text-transform: uppercase; }
        .metric-card h3 { color: #172033; margin: 10px 0 3px; font-size: 23px; }
        .metric-card p { color: #536078; margin: 0; font-size: 13px; }
        .section-title { color: #172033; font-size: 20px; font-weight: 800; margin-bottom: 4px; }
        .section-copy { color: #536078; margin: 0 0 16px; }
        .tech-pill {
            display: inline-block;
            margin: 4px 5px 4px 0;
            padding: 8px 11px;
            border: 1px solid #ddd6fe;
            border-radius: 999px;
            background: #f5f3ff;
            color: #4c1d95;
            font-weight: 800;
            font-size: 12px;
        }
        .pipeline-step {
            min-height: 94px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 7px;
            padding: 12px 6px;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            background: #f8fafc;
            color: #172033;
            text-align: center;
            font-size: 13px;
            font-weight: 800;
        }
        .result-positive, .result-negative, .result-neutral {
            border-radius: 20px;
            padding: 26px;
            color: #fff;
        }
        .result-positive { background: linear-gradient(135deg,#10b981,#059669); }
        .result-negative { background: linear-gradient(135deg,#f43f5e,#be123c); }
        .result-neutral { background: linear-gradient(135deg,#f59e0b,#d97706); }
        .stTextArea textarea, .stSelectbox > div > div, .stDataFrame {
            border-radius: 12px !important;
        }
        .stTextArea textarea {
            color: #172033 !important;
            background: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
        }
        .stTextArea textarea::placeholder { color: #64748b !important; opacity: 1; }
        [data-baseweb="select"] > div, [data-baseweb="popover"] {
            background: #ffffff !important;
            color: #172033 !important;
        }
        [data-testid="stSegmentedControl"] button {
            background: #334155 !important;
            border-color: #334155 !important;
            color: #ffffff !important;
        }
        [data-testid="stSegmentedControl"] button[aria-pressed="true"],
        [data-testid="stSegmentedControl"] button[data-selected="true"] {
            color: #ffffff !important;
            background: #5b21b6 !important;
            border-color: #5b21b6 !important;
        }
        [data-testid="stFileUploader"] section {
            background: #ffffff;
            border: 1px dashed #94a3b8;
            border-radius: 14px;
        }
        [data-testid="stFileUploader"] * { color: #26334b !important; }
        [data-testid="stDataFrame"] { border: 1px solid #e2e8f0; border-radius: 12px; }
        .stButton button {
            border-radius: 12px;
            padding: 10px 18px;
            border: 0;
            font-weight: 800;
            background: linear-gradient(135deg,#7c3aed,#4f46e5);
            color: white;
        }
        .stButton button:hover { background: #4c1d95; color: white; }
        .streamlit-expanderHeader { border-radius: 12px; }
        @media (max-width: 760px) {
            [data-testid="stAppViewContainer"] .main .block-container { padding: 1rem 1rem 2rem; }
            .hero { padding: 28px 24px; }
        }
        @media (max-width: 1050px) {
            div[data-testid="stHorizontalBlock"] {
                flex-wrap: wrap !important;
                gap: 14px !important;
            }
            div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
                flex: 1 1 100% !important;
                width: 100% !important;
                min-width: 0 !important;
            }
            [data-testid="stAppViewContainer"] .main .block-container {
                padding-left: 1.25rem;
                padding-right: 1.25rem;
            }
        }
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            min-width: 0;
            box-sizing: border-box;
        }
        .stTextArea, .stTextArea > div, .stTextArea textarea,
        [data-testid="stFileUploader"], [data-testid="stDataFrame"] {
            max-width: 100% !important;
            box-sizing: border-box !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_bilstm():
    checkpoint_path = MODEL_DIR / "bilstm.pt"
    vocabulary_path = MODEL_DIR / "bilstm_vocab.json"
    if not checkpoint_path.exists() or not vocabulary_path.exists():
        return None, None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = BiLSTMClassifier(
        vocabulary_size=checkpoint["vocabulary_size"],
        num_classes=checkpoint["num_classes"],
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, Vocabulary.load(vocabulary_path)


@st.cache_resource
def load_distilbert_model():
    model_dir = MODEL_DIR / "distilbert"
    if not model_dir.exists():
        return None, None
    try:
        tokenizer, model = load_distilbert(str(model_dir))
        model.eval()
        return tokenizer, model
    except Exception as exc:
        st.error(f"Could not load DistilBERT: {exc}")
        return None, None


def predict_bilstm(text, model, vocabulary):
    sequence = torch.tensor([vocabulary.encode(text, 128)], dtype=torch.long)
    with torch.no_grad():
        scores = torch.softmax(model(sequence), dim=1)[0]
    predicted = int(scores.argmax().item())
    return predicted, scores.tolist()


def predict_distilbert(text, tokenizer, model):
    encoded = tokenizer(text, truncation=True, max_length=128, padding=True, return_tensors="pt")
    with torch.no_grad():
        scores = torch.softmax(model(**encoded).logits, dim=1)[0]
    predicted = int(scores.argmax().item())
    return predicted, scores.tolist()


def sidebar_brand():
    st.sidebar.markdown(
        """
        <div style="padding:8px 6px 18px">
          <div style="font-size:28px">◈</div>
          <b style="display:block;font-size:17px;margin-top:8px">Sentiment AI</b>
          <small style="color:#a5b4fc!important">Yelp Deep Learning Lab</small>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(kicker, title, copy):
    st.markdown(
        f'<span style="color:#7c3aed;font-weight:800;font-size:12px;letter-spacing:1.3px">{kicker.upper()}</span>'
        f'<h1 style="margin:4px 0 6px;color:#172033">{title}</h1>'
        f'<p style="color:#7b8194;margin-bottom:24px">{copy}</p>',
        unsafe_allow_html=True,
    )


def home_page():
    st.markdown(
        """
        <section class="hero">
          <span class="eyebrow">DEEP LEARNING PROJECT</span>
          <h1>Understand every customer voice.</h1>
          <p>Convert Yelp review text into actionable sentiment intelligence using a
          bidirectional LSTM and fine-tuned DistilBERT transformer.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="metric-card"><span class="metric-icon">🧠</span><small>BiLSTM</small><h3>Sequence AI</h3><p>Bidirectional context</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><span class="metric-icon">⚡</span><small>DistilBERT</small><h3>Transformer</h3><p>Pretrained language</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><span class="metric-icon">◉</span><small>Output</small><h3>3 Sentiments</h3><p>Negative • Neutral • Positive</p></div>', unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1.35, 1])
    with left:
        with st.container(border=True):
            st.markdown("### From review to insight")
            st.caption("A complete, reproducible NLP workflow.")
            pipeline = st.columns(5, gap="small")
            steps = [
                ("📝", "Raw review"),
                ("🧹", "Preprocess"),
                ("🔢", "Encode"),
                ("🤖", "Model"),
                ("📊", "Sentiment"),
            ]
            for col, (icon, label) in zip(pipeline, steps):
                col.markdown(
                    f'<div class="pipeline-step"><span style="font-size:20px">{icon}</span>{label}</div>',
                    unsafe_allow_html=True,
                )
    with right:
        with st.container(border=True):
            st.markdown("### Project stack")
            st.caption("Modern machine-learning and visualization tools.")
            technologies = ["Python 3.11", "PyTorch", "Transformers", "Hugging Face", "Plotly", "Streamlit"]
            pills = "".join(f'<span class="tech-pill">{item}</span>' for item in technologies)
            st.markdown(pills, unsafe_allow_html=True)


def single_prediction_page():
    page_header("Live inference", "Analyze a single review", "Choose a model, enter a customer review, and receive a confidence breakdown.")
    left, right = st.columns([1.45, 1])
    with left:
        with st.container(border=True):
            model_choice = st.segmented_control("Model", ["DistilBERT", "BiLSTM"], default="DistilBERT")
            text = st.text_area(
                "Customer review",
                placeholder="Example: The food was excellent, but the waiting time was much longer than expected.",
                height=170,
            )
            analyze = st.button("Analyze sentiment", type="primary", use_container_width=True)

    with right:
        with st.container(border=True):
            st.markdown("### Prediction")
            st.caption("Your sentiment result will appear here.")
            if analyze:
                if not text.strip():
                    st.warning("Please enter a customer review.")
                elif model_choice == "DistilBERT":
                    tokenizer, model = load_distilbert_model()
                    if model is None:
                        st.error("DistilBERT has not been trained yet.")
                    else:
                        predicted, scores = predict_distilbert(text, tokenizer, model)
                        show_prediction(predicted, scores)
                else:
                    model, vocabulary = load_bilstm()
                    if model is None:
                        st.error("BiLSTM has not been trained yet.")
                    else:
                        predicted, scores = predict_bilstm(text, model, vocabulary)
                        show_prediction(predicted, scores)


def show_prediction(predicted, scores):
    class_name = LABEL_NAMES[predicted]
    css_name = class_name.lower()
    emoji = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}[class_name]
    st.markdown(
        f'<div class="result-{css_name}"><div style="font-size:34px">{emoji}</div>'
        f'<small style="opacity:.82">PREDICTED SENTIMENT</small>'
        f'<h2 style="color:white;margin:5px 0 0">{class_name}</h2>'
        f'<p style="color:white;margin:4px 0">{scores[predicted]:.1%} confidence</p></div>',
        unsafe_allow_html=True,
    )
    confidence = pd.DataFrame({"Sentiment": LABEL_NAMES, "Confidence": scores}).sort_values("Confidence", ascending=False)
    fig = px.bar(
        confidence,
        x="Confidence",
        y="Sentiment",
        orientation="h",
        color="Sentiment",
        color_discrete_map={"Positive": "#10b981", "Negative": "#f43f5e", "Neutral": "#f59e0b"},
        text_auto=".1%",
    )
    fig.update_layout(showlegend=False, height=220, margin=dict(l=0,r=10,t=10,b=10), xaxis=dict(range=[0,1],showgrid=True,gridcolor="#edf0f7"), yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig, use_container_width=True)


def bulk_analysis_page():
    page_header("Batch processing", "Bulk review analysis", "Upload a CSV with a text column and generate sentiment predictions for every review.")
    uploaded = st.file_uploader("Upload customer review CSV", type=["csv"])
    model_choice = st.segmented_control("Model", ["DistilBERT", "BiLSTM"], default="DistilBERT", key="bulk_model")
    if uploaded is None:
        st.markdown(
            '<div class="panel" style="text-align:center;padding:54px"><div style="font-size:44px">📁</div>'
            '<div class="section-title">Drop in a CSV to begin</div>'
            '<p class="section-copy">Required column: <b>text</b></p></div>',
            unsafe_allow_html=True,
        )
        return
    try:
        frame = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return
    if "text" not in frame.columns:
        st.error("The CSV must contain a column named text.")
        return
    st.caption(f"{len(frame):,} reviews loaded")
    if st.button("Analyze all reviews", type="primary"):
        predictions, confidences = run_bulk_predictions(frame["text"].fillna("").astype(str).tolist(), model_choice)
        if predictions is None:
            return
        result = frame.copy()
        result["predicted_sentiment"] = predictions
        result["confidence"] = confidences
        st.success(f"Analyzed {len(result):,} reviews")
        st.dataframe(result, use_container_width=True)
        st.download_button(
            "Download predictions CSV",
            result.to_csv(index=False).encode("utf-8"),
            "sentiment_predictions.csv",
            "text/csv",
        )


def run_bulk_predictions(texts, model_choice):
    predictions, confidences = [], []
    if model_choice == "DistilBERT":
        tokenizer, model = load_distilbert_model()
        if model is None:
            st.error("DistilBERT has not been trained yet.")
            return None, None
        for text in texts:
            predicted, scores = predict_distilbert(text, tokenizer, model)
            predictions.append(LABEL_NAMES[predicted])
            confidences.append(scores[predicted])
    else:
        model, vocabulary = load_bilstm()
        if model is None:
            st.error("BiLSTM has not been trained yet.")
            return None, None
        for text in texts:
            predicted, scores = predict_bilstm(text, model, vocabulary)
            predictions.append(LABEL_NAMES[predicted])
            confidences.append(scores[predicted])
    return predictions, confidences


def analytics_dashboard_page():
    page_header("Test data insights", "Analytics dashboard", "Explore sentiment and review-length patterns in the processed test dataset.")
    test_path = PROJECT_ROOT / "data" / "processed" / "test.csv"
    if not test_path.exists():
        st.info("Run data preparation to populate this dashboard.")
        return
    frame = pd.read_csv(test_path)
    frame["sentiment"] = frame["label"].map(dict(enumerate(LABEL_NAMES)))
    frame["review_length"] = frame["text"].fillna("").astype(str).str.split().str.len()
    counts = frame["sentiment"].value_counts().reindex(LABEL_NAMES, fill_value=0)
    total = len(frame)

    c1, c2, c3 = st.columns(3)
    cards = [
        ("Negative", counts["Negative"], "#ef4444", "↓"),
        ("Neutral", counts["Neutral"], "#f59e0b", "•"),
        ("Positive", counts["Positive"], "#10b981", "↑"),
    ]
    for col, (name, count, color, symbol) in zip((c1, c2, c3), cards):
        col.markdown(
            f'<div class="panel"><small style="color:{color};font-weight:800">{symbol} {name.upper()}</small>'
            f'<h2 style="margin:8px 0 2px">{count / total:.1%}</h2>'
            f'<p style="color:#8b92a4;margin:0">{count:,} of {total:,} reviews</p></div>',
            unsafe_allow_html=True,
        )

    left, right = st.columns([1, 1.25])
    with left:
        fig = px.pie(
            names=counts.index,
            values=counts.values,
            hole=0.62,
            color=counts.index,
            color_discrete_map={"Negative":"#ef4444","Neutral":"#f59e0b","Positive":"#10b981"},
        )
        fig.update_layout(height=380, margin=dict(l=20,r=20,t=20,b=20), legend=dict(orientation="h",y=-0.12))
        fig.update_traces(textinfo="percent", textfont_size=14)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.histogram(frame, x="review_length", nbins=45, color_discrete_sequence=["#7c3aed"])
        fig.update_layout(height=380, title="Review length distribution", margin=dict(l=20,r=20,t=55,b=20), plot_bgcolor="white")
        fig.update_xaxes(title="Words", gridcolor="#edf0f7")
        fig.update_yaxes(title="Reviews", gridcolor="#edf0f7")
        st.plotly_chart(fig, use_container_width=True)


def model_comparison_page():
    page_header("Model laboratory", "Model comparison", "Compare real test metrics, training cost, inference cost, and confusion matrices.")
    metrics_path = OUTPUT_DIR / "evaluation_metrics.json"
    if not metrics_path.exists():
        st.info("Run python evaluation.py after training both models.")
        return
    results = json.loads(metrics_path.read_text(encoding="utf-8"))
    comparison = pd.DataFrame([
        {
            "Model": name,
            "Accuracy": result["accuracy"],
            "Macro Precision": result["macro_precision"],
            "Macro Recall": result["macro_recall"],
            "Macro F1": result["macro_f1"],
            "Training seconds": result.get("training_seconds"),
            "Inference seconds": result["inference_seconds"],
        }
        for name, result in results.items()
    ])

    c1, c2 = st.columns(2)
    for col, model_name in zip((c1, c2), comparison["Model"]):
        result = results[model_name]
        col.markdown(
            f'<div class="panel"><small style="color:#7c3aed;font-weight:800">{model_name.upper()}</small>'
            f'<h2 style="margin:8px 0">Macro F1 <span style="color:#7c3aed">{result["macro_f1"]:.1%}</span></h2>'
            f'<p style="color:#7b8194">Accuracy {result["accuracy"]:.1%} • '
            f'Training {result["training_seconds"]:.1f}s • Inference {result["inference_seconds"]:.1f}s</p></div>',
            unsafe_allow_html=True,
        )

    long_metrics = comparison.melt(id_vars="Model", value_vars=["Accuracy","Macro Precision","Macro Recall","Macro F1"])
    fig = px.bar(
        long_metrics,
        x="variable",
        y="value",
        color="Model",
        barmode="group",
        text_auto=".1%",
        color_discrete_map={"BiLSTM":"#06b6d4","DistilBERT":"#7c3aed"},
    )
    fig.update_layout(height=410, yaxis=dict(range=[0,1],gridcolor="#edf0f7"), plot_bgcolor="white", margin=dict(l=20,r=20,t=35,b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Confusion matrices</div>', unsafe_allow_html=True)
    matrix_columns = st.columns(len(results))
    for col, (name, result) in zip(matrix_columns, results.items()):
        fig = go.Figure(go.Heatmap(
            z=result["confusion_matrix"],
            x=["Neg","Neu","Pos"],
            y=["Neg","Neu","Pos"],
            colorscale="Purples",
            text=result["confusion_matrix"],
            texttemplate="%{text}",
            showscale=False,
        ))
        fig.update_layout(title=name, height=350, margin=dict(l=20,r=20,t=50,b=20), xaxis_title="Predicted", yaxis_title="Actual")
        col.plotly_chart(fig, use_container_width=True)


st.set_page_config(
    page_title="Sentiment AI",
    page_icon="◈",
    layout="wide",
)
inject_styles()
sidebar_brand()
page = st.sidebar.radio(
    "Workspace",
    ["Home", "Single Review Prediction", "Bulk Review Analysis", "Analytics Dashboard", "Model Comparison"],
)
st.sidebar.caption("PyTorch • Hugging Face • Streamlit")

if page == "Home":
    home_page()
elif page == "Single Review Prediction":
    single_prediction_page()
elif page == "Bulk Review Analysis":
    bulk_analysis_page()
elif page == "Analytics Dashboard":
    analytics_dashboard_page()
else:
    model_comparison_page()
