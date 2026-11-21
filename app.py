import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import hstack
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Email & SMS Spam Classifier | Research AI Benchmark",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with Dark & Light theme support
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        opacity: 0.85;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 15px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        background: rgba(128, 128, 128, 0.05);
    }
    .spam-badge {
        background-color: #dc2626;
        color: white;
        padding: 8px 18px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        font-size: 1.15rem;
        letter-spacing: 0.5px;
    }
    .ham-badge {
        background-color: #16a34a;
        color: white;
        padding: 8px 18px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        font-size: 1.15rem;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Quiet NLTK Resource Setup
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def setup_nltk():
    for corpus in ['stopwords', 'wordnet', 'omw-1.4']:
        try:
            nltk.data.find(f'corpora/{corpus}')
        except LookupError:
            nltk.download(corpus, quiet=True)

setup_nltk()

# ---------------------------------------------------------
# Session State Initialization for Persistent Input
# ---------------------------------------------------------
if "message_input" not in st.session_state:
    st.session_state.message_input = ""

if "trigger_classify" not in st.session_state:
    st.session_state.trigger_classify = False

def set_preset(text):
    st.session_state.message_input = text
    st.session_state.trigger_classify = True

# ---------------------------------------------------------
# Contraction & Preprocessing Pipelines (Matching Notebook)
# ---------------------------------------------------------
CONTRACTIONS = {
    "won't": "will not", "can't": "cannot", "n't": " not",
    "'re": " are", "'s": " is", "'d": " would",
    "'ll": " will", "'t": " not", "'ve": " have",
    "'m": " am", "u": "you", "ur": "your", "txt": "text",
    "pls": "please", "plz": "please", "msg": "message"
}

def clean_text_pipeline(text):
    text = str(text).lower()
    for contraction, expansion in CONTRACTIONS.items():
        text = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text, flags=re.IGNORECASE)
    
    # Entity masking
    text = re.sub(r'(https?://\S+|www\.\S+)', ' __url__ ', text)
    text = re.sub(r'\b\d{10,}\b|\b\d{3,5}[-\s]?\d{3,5}[-\s]?\d{3,5}\b', ' __phone__ ', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' __email__ ', text)
    text = re.sub(r'[$£€¥₹]|\b(?:usd|gbp|eur)\b', ' __currency__ ', text)
    
    # Strip non-alphanumeric noise while retaining entity markers
    text = re.sub(r'[^a-zA-Z0-9_\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_engineered_features(raw_text):
    text_len = len(raw_text)
    words = raw_text.split()
    word_cnt = len(words)
    
    capitals = sum(1 for c in raw_text if c.isupper())
    digits = sum(1 for c in raw_text if c.isdigit())
    specials = sum(1 for c in raw_text if not c.isalnum() and not c.isspace())
    exclamations = raw_text.count('!')
    
    cap_ratio = capitals / max(text_len, 1)
    digit_ratio = digits / max(text_len, 1)
    special_ratio = specials / max(text_len, 1)
    
    has_curr = int(bool(re.search(r'[$£€¥₹]|\b(?:usd|gbp|eur)\b', raw_text, re.IGNORECASE)))
    has_link = int(bool(re.search(r'https?://|www\.', raw_text, re.IGNORECASE)))
    has_ph = int(bool(re.search(r'\b\d{10,}\b|\b\d{3,5}[-\s]?\d{3,5}[-\s]?\d{3,5}\b', raw_text)))
    
    return np.array([[
        word_cnt, text_len, cap_ratio, digit_ratio, special_ratio,
        exclamations, has_curr, has_link, has_ph
    ]], dtype=np.float64), {
        'word_count': word_cnt,
        'char_count': text_len,
        'capital_ratio': cap_ratio,
        'digit_ratio': digit_ratio,
        'exclamations': exclamations,
        'has_currency': bool(has_curr),
        'has_url': bool(has_link),
        'has_phone': bool(has_ph)
    }

# ---------------------------------------------------------
# Resource Loading via @st.cache_resource
# ---------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

@st.cache_resource
def load_metadata():
    path = os.path.join(MODELS_DIR, 'model_metadata.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

@st.cache_resource
def load_vectorizer_and_scaler():
    vec_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib')
    scaler_path = os.path.join(MODELS_DIR, 'feature_scaler.joblib')
    vec = joblib.load(vec_path)
    scaler = joblib.load(scaler_path)
    return vec, scaler

@st.cache_resource
def load_model(model_filename):
    path = os.path.join(MODELS_DIR, model_filename)
    return joblib.load(path)

# Initialize models and artifacts
metadata = load_metadata()
tfidf_vectorizer, feature_scaler = load_vectorizer_and_scaler()

MODEL_MAP = {
    "ANN (Deep Multi-Layer Perceptron) [Champion]": ("champion_model.joblib", "fused"),
    "Multinomial Naive Bayes (Laplace)": ("naive_bayes.joblib", "tfidf"),
    "Support Vector Machine (Calibrated LinearSVC)": ("support_vector_machine.joblib", "fused"),
    "Random Forest (200 Trees)": ("random_forest.joblib", "fused"),
    "XGBoost Classifier": ("xgboost.joblib", "fused"),
    "Logistic Regression (Balanced)": ("logistic_regression.joblib", "fused")
}

# ---------------------------------------------------------
# Header & Navigation
# ---------------------------------------------------------
st.markdown('<div class="main-title">🛡️ Email & SMS Spam Classification</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Real-Time Threat Detection powered by Classical Machine Learning and Deep Neural Networks</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Sidebar: Project Details & Benchmark Comparison
# ---------------------------------------------------------
with st.sidebar:
    st.header("📊 Benchmark & Architecture")
    st.markdown("""
    **Project Scope:**
    * **Corpus:** 5,572 enriched SMS records (403 duplicates audited & dropped).
    * **Anti-Leakage Split:** 70% Train, 15% Validation, 15% Test (Stratified).
    * **Feature Fusion:** 4,000 TF-IDF n-grams + 9 standardized continuous telemetry features.
    """)
    
    st.subheader("Model Performance (Test Set)")
    if 'benchmark_metrics' in metadata:
        bench_df = pd.DataFrame(metadata['benchmark_metrics'])
        display_cols = ['Model', 'Accuracy', 'Precision (Spam)', 'Recall (Spam)', 'F1-Score (Spam)', 'ROC-AUC']
        display_df = bench_df[display_cols].copy()
        for col in display_cols[1:]:
            display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%" if col != 'ROC-AUC' else f"{x:.4f}")
        st.dataframe(display_df, hide_index=True, use_container_width=True)
        
        st.subheader("F1-Score vs Precision")
        chart_data = pd.DataFrame({
            'Model': bench_df['Model'],
            'F1-Score': bench_df['F1-Score (Spam)'],
            'Precision': bench_df['Precision (Spam)']
        }).set_index('Model')
        st.bar_chart(chart_data)
    
    st.divider()
    st.markdown("📖 **Research Artifact:** [Jupyter Notebook](notebooks/Email_Spam_Classification_Research.ipynb)")

# ---------------------------------------------------------
# Main Panel: Real-Time Prediction Interface
# ---------------------------------------------------------
col1, col2 = st.columns([1.6, 1.0], gap="large")

with col1:
    st.subheader("✉️ Message Classification")
    
    # Model Selection Dropdown
    selected_model_name = st.selectbox(
        "Select Classification Model Architecture:",
        options=list(MODEL_MAP.keys()),
        index=0
    )
    
    # Sample Presets with on_click callbacks for robust state preservation
    st.markdown("**Quick Test Presets (Click to autofill):**")
    sample_col1, sample_col2, sample_col3 = st.columns(3)
    sample_col4, sample_col5, sample_col6 = st.columns(3)
    
    sample_col1.button(
        "🚨 Spam: Lottery Prize",
        use_container_width=True,
        on_click=set_preset,
        args=("WINNER!! As a valued customer you have been selected to receive £900 prize reward! Call 09061701461 to claim now. T&Cs apply.",)
    )
    sample_col2.button(
        "🚨 Spam: Bank Alert",
        use_container_width=True,
        on_click=set_preset,
        args=("URGENT: Your bank card has been suspended due to suspicious activity. Visit http://secure-bank-login.cc/verify immediately or call 08000930707.",)
    )
    sample_col3.button(
        "🚨 Spam: Gift Offer",
        use_container_width=True,
        on_click=set_preset,
        args=("Congratulations! You have won a FREE entry to 2 wkly comps for cash prizes. Text WIN to 87121 now to claim your reward.",)
    )
    sample_col4.button(
        "✅ Ham: Meeting Sync",
        use_container_width=True,
        on_click=set_preset,
        args=("Hi Ayush, just checking if we are still meeting at 4 PM for the project sync? Let me know if you need to reschedule.",)
    )
    sample_col5.button(
        "✅ Ham: Casual Chat",
        use_container_width=True,
        on_click=set_preset,
        args=("Hey! Going to Jurong Point for dinner tonight. Are you free to come along? Got a discount voucher.",)
    )
    sample_col6.button(
        "✅ Ham: Flight / ETA",
        use_container_width=True,
        on_click=set_preset,
        args=("My flight landed safely. Will take a cab and reach home by 7:30 PM. See you soon!",)
    )
    
    # Text Area bound directly to st.session_state["message_input"]
    user_text = st.text_area(
        "Enter or paste the message text to analyze:",
        key="message_input",
        height=140,
        placeholder="Type or paste any SMS or email body here..."
    )
    
    classify_clicked = st.button("🔍 Classify Message", type="primary", use_container_width=True)
    
    # Trigger classification if either button clicked or preset activated
    should_run = classify_clicked or st.session_state.trigger_classify
    # Reset trigger flag for subsequent interactions
    st.session_state.trigger_classify = False

with col2:
    st.subheader("🎯 Prediction & Telemetry")
    
    if should_run:
        current_text = st.session_state.message_input.strip()
        
        if current_text:
            model_filename, input_type = MODEL_MAP[selected_model_name]
            clf = load_model(model_filename)
            
            # Preprocessing text
            cleaned = clean_text_pipeline(current_text)
            stop_words = set(stopwords.words('english'))
            stop_words.difference_update({"not", "no", "nor", "only", "free", "all", "win", "cash"})
            lemmatizer = WordNetLemmatizer()
            
            tokens = cleaned.split()
            tokens_no_stop = [t for t in tokens if t not in stop_words and len(t) > 1]
            lemmatized_tokens = [lemmatizer.lemmatize(t, pos='v') for t in tokens_no_stop]
            processed_text = ' '.join(lemmatized_tokens)
            
            # Feature transformation
            text_vec = tfidf_vectorizer.transform([processed_text])
            eng_raw, telemetry = extract_engineered_features(current_text)
            
            if input_type == 'tfidf':
                final_features = text_vec
            else:
                eng_scaled = feature_scaler.transform(eng_raw)
                final_features = hstack([text_vec, eng_scaled]).tocsr()
            
            # Inference
            pred_label = clf.predict(final_features)[0]
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(final_features)[0]
                spam_prob = float(probs[1])
            else:
                spam_prob = 1.0 if pred_label == 1 else 0.0
            
            # Display Result Card
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if pred_label == 1:
                st.markdown('<div class="spam-badge">🚨 SPAM DETECTED</div>', unsafe_allow_html=True)
                st.markdown(f"**Confidence:** `{spam_prob * 100:.2f}%`")
                st.progress(min(max(spam_prob, 0.0), 1.0))
            else:
                st.markdown('<div class="ham-badge">✅ LEGITIMATE (HAM)</div>', unsafe_allow_html=True)
                st.markdown(f"**Confidence:** `{(1.0 - spam_prob) * 100:.2f}%`")
                st.progress(min(max(1.0 - spam_prob, 0.0), 1.0))
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display Telemetry Signals
            st.markdown("##### 🔬 Extracted Telemetry Signals")
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.metric("Length", f"{telemetry['char_count']} chars")
                st.metric("Words", f"{telemetry['word_count']} words")
                st.metric("Capital Ratio", f"{telemetry['capital_ratio']*100:.1f}%")
            with col_t2:
                st.write(f"🌐 **URL Detected:** {'Yes' if telemetry['has_url'] else 'No'}")
                st.write(f"📞 **Phone Number:** {'Yes' if telemetry['has_phone'] else 'No'}")
                st.write(f"💰 **Currency Symbol:** {'Yes' if telemetry['has_currency'] else 'No'}")
                st.write(f"❗ **Exclamation Marks:** {telemetry['exclamations']}")
                
            with st.expander("Show Lemmatized Tokens"):
                st.code(processed_text if processed_text else "(No tokens after stopword filtering)")
        else:
            st.warning("⚠️ Please enter a message text before clicking Classify.")
    else:
        st.info("👈 Enter a message or select a test preset, choose a model, and click **Classify Message**.")
