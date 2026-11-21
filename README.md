# 🛡️ Real-Time Email & SMS Spam Classification

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5%2B-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1%2B-green.svg)](https://xgboost.ai/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An end-to-end, research-grade Email and SMS Spam Classification project developed for hackathons and production deployment. This project benchmarks Classical Machine Learning models against Deep Learning architectures under strict, leak-free experimental conditions, and deploys the champion model via an interactive **Streamlit Web Application** for real-time threat detection.

🌐 **Live Streamlit App:** `https://<your-username>-spam-classifier.streamlit.app` *(Deploy via share.streamlit.io)*

---

## 📌 Executive Summary & Research Highlights

* **Corpus:** 5,572 enriched SMS records mapped into an email-like schema.
* **Data Sanitization:** 403 duplicates ($7.23\%$) identified and purged to eliminate train/test contamination. 20 unpopulated email header placeholders identified and removed.
* **Severe Class Imbalance:** $4,516$ Ham ($87.37\%$) vs. $653$ Spam ($12.63\%$) — an imbalance ratio of $\approx 6.92 : 1$.
* **Leak-Proof Splitting:** Stratified $70\%$ Train, $15\%$ Validation, $15\%$ Test split. All vectorizers and scalers were fit strictly on training partitions.
* **Multimodal Representation:** Fused $4,000$ sublinear TF-IDF n-grams ($1$-gram and $2$-gram) with $9$ standardized continuous telemetry signals (capitalization ratio, digit density, punctuation counts, URL/phone/currency indicators).
* **Champion Model:** **Artificial Neural Network (Deep Multi-Layer Perceptron)** achieving **$99.36\%$ Accuracy**, **$98.95\%$ Precision**, **$95.92\%$ Recall**, and **$0.9986$ ROC-AUC**.

---

## 📊 Benchmark Results on Sealed Test Set ($N=776$)

| Rank | Model Architecture | Accuracy | Precision (Spam) | Recall (Spam) | F1-Score (Spam) | Macro F1 | ROC-AUC | PR-AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **ANN (Deep MLP)** | **99.36%** | **98.95%** | **95.92%** | **0.9741** | **0.9852** | **0.9986** | **0.9928** |
| 🥈 | **Multinomial Naive Bayes** | $99.10\%$ | $98.92\%$ | $93.88\%$ | $0.9634$ | $0.9791$ | $0.9962$ | $0.9856$ |
| 🥉 | **Random Forest (200 Trees)** | $98.84\%$ | $96.84\%$ | $93.88\%$ | $0.9534$ | $0.9734$ | $0.9971$ | $0.9861$ |
| 4 | **Support Vector Machine (Calibrated)** | $98.84\%$ | $96.84\%$ | $93.88\%$ | $0.9534$ | $0.9734$ | $0.9950$ | $0.9850$ |
| 5 | **XGBoost Classifier** | $98.45\%$ | $93.88\%$ | $93.88\%$ | $0.9388$ | $0.9650$ | $0.9951$ | $0.9813$ |
| 6 | **Logistic Regression (Balanced)** | $98.07\%$ | $91.09\%$ | $93.88\%$ | $0.9246$ | $0.9568$ | $0.9950$ | $0.9790$ |

> **Key Cybersecurity Insight:** Minimizing False Positives (legitimate messages flagged as spam) is paramount. The champion ANN model made only **$1$ False Positive** across 678 legitimate test messages.

---

## 📁 Repository Structure

```
hcl_tech/
│
├── .gitignore                                 # Git exclusions (venv, cache, checkpoints)
├── README.md                                  # Complete research report and documentation
├── requirements.txt                           # Pinned dependencies for Streamlit Cloud
├── runtime.txt                                # Python runtime configuration (python-3.11)
├── app.py                                     # Real-time Streamlit web application
│
├── data/
│   └── sms_spam_dataset_enriched_5572.csv    # Enriched SMS Spam Collection dataset
│
├── notebooks/
│   └── Email_Spam_Classification_Research.ipynb # Full interactive Jupyter research notebook
│
├── models/                                    # Serialized models & feature extractors (<100MB)
│   ├── champion_model.joblib                  # Winning Deep ANN model artifact (12.5 MB)
│   ├── naive_bayes.joblib                     # Multinomial Naive Bayes
│   ├── logistic_regression.joblib             # Balanced Logistic Regression
│   ├── support_vector_machine.joblib          # Calibrated Linear Support Vector Machine
│   ├── random_forest.joblib                   # 200-Tree Random Forest ensemble
│   ├── xgboost.joblib                         # Gradient-boosted trees
│   ├── tfidf_vectorizer.joblib                # Fitted TF-IDF feature extractor (4,000 n-grams)
│   ├── feature_scaler.joblib                  # Fitted continuous standard scaler (9 features)
│   └── model_metadata.json                    # Benchmark metrics and metadata
│
└── reports/
    └── figures/                               # Generated research plots & visualizations
        ├── 01_class_imbalance_distribution.png
        ├── 02_message_length_distribution.png
        ├── 03_lexical_features_distribution.png
        ├── 04_ann_loss_curve.png
        ├── 05_confusion_matrices_grid.png
        ├── 06_roc_and_pr_curves.png
        ├── 07_models_benchmark_barplot.png
        └── 08_feature_importance_coefficients.png
```

---

## 🚀 Quickstart & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/email-spam-classification.git
cd email-spam-classification
```

### 2. Create and Activate a Virtual Environment
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 5. Run the Research Notebook
```bash
jupyter notebook notebooks/Email_Spam_Classification_Research.ipynb
```

---

## 📸 Application Screenshots

*(Screenshots generated from local Streamlit run)*

### 1. Real-Time Spam Detection Interface
The user inputs an SMS/email message, selects any of the 6 trained architectures, and receives real-time classification, confidence gauge, and extracted telemetry.

### 2. Multi-Model Benchmark & Research Dashboard
The sidebar features a live leaderboard displaying accuracy, precision, recall, F1, and an interactive bar chart comparison.

---

## 🌐 Deployment to Streamlit Community Cloud

1. Push this repository to GitHub on branch `main`.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **"New App"**.
4. Select your repository, set **Branch** to `main`, and **Main file path** to `app.py`.
5. Click **"Deploy"**. The app will automatically build and launch in under 2 minutes.

---

## 📜 License
This project is open-source under the MIT License.
