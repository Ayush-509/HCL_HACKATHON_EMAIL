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

# Custom CSS for Dark & Light Theme Accessibility
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
    .threat-tag {
        background-color: rgba(220, 38, 38, 0.15);
        color: #ef4444;
        border: 1px solid rgba(220, 38, 38, 0.3);
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 3px 4px 3px 0;
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
# Cyber Threat Heuristics & Signature Dictionary
# ---------------------------------------------------------
THREAT_PATTERNS = {
    "Account Suspension / Lock Alert": r'\b(?:account|card|wallet|profile|access|apple\s*id|paypal|bank|amazon|netflix)\b.*?\b(?:suspended|blocked|locked|limited|compromised|frozen|disabled|terminated)\b',
    "Credential / Identity Phishing": r'\b(?:verify|confirm|validate|update|restore)\b.*?\b(?:account|identity|password|pin|security|details|billing|credential)\b',
    "Fraud / Suspicious Activity Warning": r'\b(?:unauthorized|suspicious|unusual|fraudulent)\b.*?\b(?:activity|transaction|transactions|access|charge|charges|login|alert)\b',
    "Lottery / Prize / Inheritance Scam": r'\b(?:congratulations|winner|selected|lottery|prize|reward|grant|jackpot|inheritance|next\s+of\s+kin|deceased\s+client)\b',
    "Urgent Action / Immediate Deadline": r'\b(?:urgent|immediate|immediately|action\s+required|within\s+24\s+hours?|time\s+sensitive|respond\s+now|act\s+now)\b',
    "Call-to-Action Link": r'\b(?:click\s+here|click\s+the\s+link|visit\s+http|login\s+immediately|verify\s+now|reset\s+now|open\s+link)\b',
    "High-Yield / Work-from-Home Scam": r'\b(?:get\s+rich|earn\s+money|make\s+money|work\s+from\s+home|guaranteed\s+profit|invest\s+in\s+crypto|bitcoin\s+investment)\b',
    "Phishing Invoice / Fake Billing": r'\b(?:invoice|receipt|order\s+confirmation|billed\s+you|payment\s+of)\b.*?\b(?:\$|\b\d+\s*usd\b|attached|call\s+support|dispute)\b',
    "Unsolicited Pharma / Spam Product": r'\b(?:prescription|medications?|viagra|cialis|weight\s+loss|discount\s+pills)\b',
    "Promotional Discount Blast": r'\b(?:free\s+delivery|discount|discounts|cheap|save\s+up\s+to|\d+%\s+off|best\s+price)\b',
    "Suspicious Shortened URL": r'(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|is\.gd|buff\.ly|ow\.ly|serveo\.net|ngrok\.io|https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})',
    "Monetary Currency Trigger": r'[$£€¥₹]|\b(?:usd|gbp|eur)\b'
}

def scan_threat_signatures(raw_text):
    hits = []
    for signature, regex in THREAT_PATTERNS.items():
        if re.search(regex, raw_text, re.IGNORECASE):
            hits.append(signature)
    return hits

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
def load_all_models():
    models = {
        "ANN (Deep Multi-Layer Perceptron) [Champion]": (joblib.load(os.path.join(MODELS_DIR, "champion_model.joblib")), "fused"),
        "Multinomial Naive Bayes (Laplace)": (joblib.load(os.path.join(MODELS_DIR, "naive_bayes.joblib")), "tfidf"),
        "Support Vector Machine (Calibrated LinearSVC)": (joblib.load(os.path.join(MODELS_DIR, "support_vector_machine.joblib")), "fused"),
        "Random Forest (200 Trees)": (joblib.load(os.path.join(MODELS_DIR, "random_forest.joblib")), "fused"),
        "XGBoost Classifier": (joblib.load(os.path.join(MODELS_DIR, "xgboost.joblib")), "fused"),
        "Logistic Regression (Balanced)": (joblib.load(os.path.join(MODELS_DIR, "logistic_regression.joblib")), "fused")
    }
    return models

# Initialize models and artifacts
metadata = load_metadata()
tfidf_vectorizer, feature_scaler = load_vectorizer_and_scaler()
all_models = load_all_models()

# ---------------------------------------------------------
# Header & Navigation
# ---------------------------------------------------------
st.markdown('<div class="main-title">🛡️ Email & SMS Spam Classification</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Hybrid Cyber-Defense System powered by NLP Machine Learning, Deep Neural Networks & Threat Intelligence</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Sidebar: Project Details, Sensitivity, & Benchmark
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Detection Sensitivity")
    detection_threshold = st.slider(
        "Decision Cutoff Threshold:",
        min_value=0.10,
        max_value=0.80,
        value=0.35,
        step=0.05,
        help="In imbalanced datasets (~13% spam baseline), a 0.35 cutoff provides optimal recall on modern phishing & email spam without false alarms."
    )
    
    st.header("📊 Benchmark & Architecture")
    st.markdown("""
    **Project Scope:**
    * **Corpus:** 5,572 enriched SMS records (403 duplicates audited & purged).
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
        options=list(all_models.keys()),
        index=0
    )
    
    # Sample Presets with on_click callbacks for robust state preservation
    st.markdown("**Quick Test Presets (Click to autofill):**")
    sample_col1, sample_col2, sample_col3 = st.columns(3)
    sample_col4, sample_col5, sample_col6 = st.columns(3)
    
    sample_col1.button(
        "🚨 Spam: PayPal Phish",
        use_container_width=True,
        on_click=set_preset,
        args=("URGENT: Your PayPal account has been limited due to suspicious transactions. Click here to verify your identity: http://secure-paypal-login.net",)
    )
    sample_col2.button(
        "🚨 Spam: Bank Security",
        use_container_width=True,
        on_click=set_preset,
        args=("Dear Customer, Your bank account has been locked. Verify your credentials immediately at http://verify-bank.com/login to restore access.",)
    )
    sample_col3.button(
        "🚨 Spam: $5M Lottery",
        use_container_width=True,
        on_click=set_preset,
        args=("Congratulations! You have won $5,000,000 USD in the International Lottery. Please contact claims@lottery-fund.org with your details to claim cash.",)
    )
    sample_col4.button(
        "✅ Ham: Meeting Sync",
        use_container_width=True,
        on_click=set_preset,
        args=("Hi Ayush, just checking if we are still meeting at 4 PM for the project sync? Let me know if you need to reschedule.",)
    )
    sample_col5.button(
        "✅ Ham: Dinner Plans",
        use_container_width=True,
        on_click=set_preset,
        args=("Hey! Going to Jurong Point for dinner tonight. Are you free to come along? Got a discount voucher.",)
    )
    sample_col6.button(
        "✅ Ham: Travel Update",
        use_container_width=True,
        on_click=set_preset,
        args=("My flight landed safely. Will take a cab and reach home by 7:30 PM. See you soon!",)
    )
    
    # Text Area bound directly to st.session_state["message_input"]
    user_text = st.text_area(
        "Enter or paste the email or SMS text to analyze:",
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
    st.subheader("🎯 Prediction & Threat Telemetry")
    
    if should_run:
        current_text = st.session_state.message_input.strip()
        
        if current_text:
            clf, input_type = all_models[selected_model_name]
            
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
            eng_scaled = feature_scaler.transform(eng_raw)
            final_fused = hstack([text_vec, eng_scaled]).tocsr()
            
            selected_features = text_vec if input_type == 'tfidf' else final_fused
            
            # Model inference
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(selected_features)[0]
                model_spam_prob = float(probs[1])
            else:
                raw_pred = clf.predict(selected_features)[0]
                model_spam_prob = 1.0 if raw_pred == 1 else 0.0
            
            # Threat heuristics scan
            detected_threats = scan_threat_signatures(current_text)
            threat_bonus = min(len(detected_threats) * 0.15, 0.45) if detected_threats else 0.0
            
            # Hybrid combined confidence
            effective_spam_prob = min(model_spam_prob + threat_bonus, 1.0) if detected_threats else model_spam_prob
            
            # Final decision based on threshold or multiple high-risk signatures
            is_spam = (effective_spam_prob >= detection_threshold) or (len(detected_threats) >= 2)
            
            # Display Result Card
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if is_spam:
                st.markdown('<div class="spam-badge">🚨 SPAM DETECTED</div>', unsafe_allow_html=True)
                st.markdown(f"**Confidence:** `{effective_spam_prob * 100:.2f}%`")
                st.progress(min(max(effective_spam_prob, 0.0), 1.0))
            else:
                st.markdown('<div class="ham-badge">✅ LEGITIMATE (HAM)</div>', unsafe_allow_html=True)
                st.markdown(f"**Confidence:** `{(1.0 - effective_spam_prob) * 100:.2f}%`")
                st.progress(min(max(1.0 - effective_spam_prob, 0.0), 1.0))
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display Detected Threats
            if detected_threats:
                st.markdown("##### ⚠️ Threat Indicators Identified")
                threat_tags = "".join([f'<span class="threat-tag">⚠️ {t}</span>' for t in detected_threats])
                st.markdown(threat_tags, unsafe_allow_html=True)
                st.write("")
            
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
            
            # Multi-Model Consensus Breakdown
            with st.expander("🗳️ Multi-Model Consensus Breakdown"):
                consensus_scores = []
                for m_name, (m_obj, m_type) in all_models.items():
                    m_feat = text_vec if m_type == 'tfidf' else final_fused
                    p = float(m_obj.predict_proba(m_feat)[0][1]) if hasattr(m_obj, "predict_proba") else 0.0
                    consensus_scores.append({"Model": m_name, "Spam Probability": f"{p*100:.1f}%", "Verdict": "SPAM" if p >= detection_threshold else "HAM"})
                st.dataframe(pd.DataFrame(consensus_scores), hide_index=True, use_container_width=True)
                
            with st.expander("Show Lemmatized Tokens"):
                st.code(processed_text if processed_text else "(No tokens after stopword filtering)")
        else:
            st.warning("⚠️ Please enter a message text before clicking Classify.")
    else:
        st.info("👈 Enter a message or select a test preset, choose a model, and click **Classify Message**.")
