"""
💳 Credit Card Fraud Detection — Streamlit App
================================================
Run:  streamlit run streamlit_app.py
Need: pip install streamlit pandas numpy scikit-learn plotly faiss-cpu openpyxl

Tabs:
  1. Dataset         — raw data preview + basic info
  2. Imbalance       — before/after class counts + chart
  3. Feature Eng.    — engineered dataset preview + descriptions
  4. Predictions     — predict fraud on a new transaction
  5. RAG Chatbot     — ask questions about the dataset
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ─────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="💳 Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
  .stTab [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
  .metric-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    color: white; border-radius: 10px; padding: 1rem;
    text-align: center; margin-bottom: 0.5rem;
  }
  .metric-card .val { font-size: 2rem; font-weight: 700; }
  .metric-card .lbl { font-size: 0.8rem; opacity: 0.8; }
  .info { background:#eef4ff; border-left:4px solid #3b82f6;
          padding:0.7rem 1rem; border-radius:0 6px 6px 0; font-size:0.9rem; }
  .fraud-badge {
    background:#fee2e2; border:2px solid #ef4444;
    border-radius:10px; padding:1rem; text-align:center;
  }
  .legit-badge {
    background:#dcfce7; border:2px solid #22c55e;
    border-radius:10px; padding:1rem; text-align:center;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# CONSTANTS — must match notebook exactly
# ─────────────────────────────────────────────────────
RANDOM_STATE  = 42
FEATURE_COLS  = [
    "log_amount","foreign_transaction","location_mismatch","device_trust_score",
    "velocity_last_24h","merchant_category","part_of_day","age_group",
    "is_high_value","is_night","low_trust_device","risk_score","amount_velocity_interact"
]
CAT_COLS      = ["merchant_category","part_of_day","age_group"]
NUM_COLS      = [c for c in FEATURE_COLS if c not in CAT_COLS]
MERCHANT_CATS = ["Clothing","Electronics","Food","Grocery","Travel"]


# ─────────────────────────────────────────────────────
# FEATURE ENGINEERING — identical to notebook
# ─────────────────────────────────────────────────────
def get_part_of_day(h):
    if 5 <= h <= 11: return "Morning"
    elif 12 <= h <= 17: return "Afternoon"
    elif 18 <= h <= 22: return "Evening"
    else: return "Night"


def engineer_features(df, threshold=None):
    df = df.copy()
    df["part_of_day"]              = df["transaction_hour"].apply(get_part_of_day)
    df["is_night"]                 = df["transaction_hour"].apply(lambda h: 1 if h >= 23 or h <= 4 else 0)
    df["age_group"]                = pd.cut(df["cardholder_age"],
                                            bins=[0,25,35,50,65,120],
                                            labels=["Young","Adult","Middle-aged","Senior","Elderly"]).astype(str)
    if threshold is None:
        threshold = df["amount"].quantile(0.90)
    df["is_high_value"]            = (df["amount"] > threshold).astype(int)
    df["low_trust_device"]         = (df["device_trust_score"] < 50).astype(int)
    max_vel                        = max(df["velocity_last_24h"].max(), 1)
    df["risk_score"]               = (
        df["foreign_transaction"] * 0.3 + df["location_mismatch"] * 0.3 +
        (1 - df["device_trust_score"] / 100) * 0.2 +
        df["velocity_last_24h"] / max_vel * 0.2
    ).round(4)
    df["amount_velocity_interact"] = (df["amount"] * df["velocity_last_24h"]).round(4)
    df["log_amount"]               = np.log1p(df["amount"])
    return df, float(threshold)


# ─────────────────────────────────────────────────────
# DATA LOADING (cached — runs once)
# ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load and clean the raw dataset."""
    try:
        df = pd.read_excel("credit_card_fraud_10k.xls")
    except Exception:
        try:
            df = pd.read_csv("credit_card_fraud_10k.csv")
        except Exception:
            # Demo data if file not found
            st.warning("Data file not found — using demo data.")
            rng = np.random.default_rng(42)
            n   = 2000
            df  = pd.DataFrame({
                "transaction_id":    range(1, n+1),
                "amount":            rng.exponential(200, n),
                "transaction_hour":  rng.integers(0, 24, n),
                "merchant_category": rng.choice(MERCHANT_CATS, n),
                "foreign_transaction": rng.choice([0,1], n, p=[0.9,0.1]),
                "location_mismatch": rng.choice([0,1], n, p=[0.91,0.09]),
                "device_trust_score": rng.integers(25, 100, n),
                "velocity_last_24h": rng.integers(0, 10, n),
                "cardholder_age":    rng.integers(18, 80, n),
                "is_fraud":          rng.choice([0,1], n, p=[0.985,0.015]),
            })
    df.columns = df.columns.str.strip().str.lower()
    return df


@st.cache_data
def get_engineered_data():
    """Clean + engineer features from raw data."""
    df     = load_data()
    df_c   = df.copy()
    df_c.drop_duplicates(inplace=True)
    df_c["amount"] = df_c["amount"].clip(lower=0)
    if "transaction_id" in df_c.columns:
        df_c.drop(columns=["transaction_id"], inplace=True)
    df_feat, thr = engineer_features(df_c)
    return df_feat, thr


# ─────────────────────────────────────────────────────
# TRAIN ML PIPELINE (cached — trains once)
# ─────────────────────────────────────────────────────
@st.cache_resource
def train_pipeline():
    """Train all 6 ML models and return everything needed for predictions."""
    df_feat, threshold = get_engineered_data()

    X = df_feat[FEATURE_COLS]
    y = df_feat["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Preprocessing: StandardScaler for numbers, OneHotEncoder for categories
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("sc", StandardScaler())]), NUM_COLS),
        ("cat", Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT_COLS),
    ])
    X_tr = preprocessor.fit_transform(X_train)   # fit on TRAIN only
    X_te = preprocessor.transform(X_test)         # transform test with train params

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5, random_state=RANDOM_STATE),
        "Decision Tree":       DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=RANDOM_STATE),
        "Naive Bayes":         GaussianNB(),
        "SVM":                 SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE),
    }

    results, trained = [], {}
    for name, m in models.items():
        m.fit(X_tr, y_train)
        trained[name] = m
        yp   = m.predict(X_te)
        yprb = m.predict_proba(X_te)[:,1]
        results.append({
            "Model":     name,
            "Accuracy":  round(accuracy_score(y_test, yp), 4),
            "Precision": round(precision_score(y_test, yp, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, yp, zero_division=0), 4),
            "F1-Score":  round(f1_score(y_test, yp, zero_division=0), 4),
            "ROC-AUC":   round(roc_auc_score(y_test, yprb), 4),
        })

    results_df      = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    best_name       = results_df.iloc[0]["Model"]

    return trained, preprocessor, results_df, best_name, threshold


# ─────────────────────────────────────────────────────
# RAG CHATBOT (cached — built once)
# ─────────────────────────────────────────────────────
@st.cache_resource
def get_chatbot():
    """Build and return the RAG chatbot. Returns (bot, error_message)."""
    try:
        from rag_chatbot import FraudRAGChatbot, engineer_features
        df_feat, _ = get_engineered_data()
        bot = FraudRAGChatbot(df_feat)
        return bot, None
    except ImportError as e:
        return None, f"Missing package: {e}. Run: pip install faiss-cpu scikit-learn"
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────────────
# LOAD EVERYTHING
# ─────────────────────────────────────────────────────
df_raw             = load_data()
df_feat, thr       = get_engineered_data()

with st.spinner("Training ML models (first run only)…"):
    trained, preprocessor, results_df, best_name, amt_thr = train_pipeline()


# ─────────────────────────────────────────────────────
# HEADER + KPI ROW
# ─────────────────────────────────────────────────────
st.title("💳 Credit Card Fraud Detection Dashboard")
st.caption("End-to-end pipeline: EDA → Feature Engineering → 6 ML Models → RAG Chatbot")

total   = len(df_raw)
n_fraud = int(df_raw["is_fraud"].sum())
fr_rate = n_fraud / total * 100
best_auc = float(results_df.iloc[0]["ROC-AUC"])

c1, c2, c3, c4 = st.columns(4)
for col, val, lbl in [
    (c1, f"{total:,}", "Total Transactions"),
    (c2, f"{n_fraud} ({fr_rate:.2f}%)", "Fraud Cases"),
    (c3, f"{best_auc:.4f}", f"Best AUC ({best_name})"),
    (c4, f"${df_raw['amount'].mean():.0f}", "Avg Amount"),
]:
    col.markdown(f'<div class="metric-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>',
                 unsafe_allow_html=True)

st.markdown("")

# ─────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs([
    "📋 Dataset", "⚖️ Imbalance", "🔧 Features", "🤖 Predict", "💬 RAG Chatbot"
])


# ═══════════════════════════════════════════════
# TAB 1 — DATASET
# ═══════════════════════════════════════════════
with t1:
    st.subheader("Raw Dataset")
    st.markdown('<div class="info">10,000 credit card transactions. '
                'Target column <code>is_fraud</code>: 1 = fraud, 0 = legitimate.</div>',
                unsafe_allow_html=True)
    st.dataframe(df_raw.head(20), use_container_width=True)

    l, r = st.columns(2)
    with l:
        st.markdown("**Column Info**")
        info = pd.DataFrame({
            "Type": [str(t) for t in df_raw.dtypes],
            "Non-Null": df_raw.notnull().sum(),
            "Unique":   df_raw.nunique(),
        })
        st.dataframe(info, use_container_width=True)
    with r:
        st.markdown("**Statistics**")
        st.dataframe(df_raw.describe().T.round(2), use_container_width=True)

    st.subheader("EDA Charts")
    c1_, c2_ = st.columns(2)
    with c1_:
        cnt = df_raw["is_fraud"].value_counts().rename({0:"Legitimate",1:"Fraud"})
        fig = px.pie(values=cnt.values, names=cnt.index,
                     color=cnt.index, color_discrete_map={"Legitimate":"#22c55e","Fraud":"#ef4444"},
                     title="Fraud vs Legitimate", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    with c2_:
        cat = df_raw.groupby("merchant_category")["is_fraud"].mean().reset_index()
        cat.columns = ["Category","Fraud Rate"]
        cat["Fraud Rate %"] = (cat["Fraud Rate"]*100).round(2)
        fig2 = px.bar(cat.sort_values("Fraud Rate %"), x="Fraud Rate %", y="Category",
                      orientation="h", color="Fraud Rate %", color_continuous_scale="Reds",
                      title="Fraud Rate by Merchant Category", text="Fraud Rate %")
        fig2.update_traces(texttemplate="%{text:.2f}%")
        fig2.update_layout(showlegend=False, plot_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

    hr = df_raw.groupby("transaction_hour")["is_fraud"].mean().reset_index()
    hr.columns = ["Hour","Fraud Rate %"]
    hr["Fraud Rate %"] = (hr["Fraud Rate %"]*100).round(3)
    fig3 = px.area(hr, x="Hour", y="Fraud Rate %", title="Fraud Rate by Hour of Day",
                   color_discrete_sequence=["#ef4444"])
    fig3.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════
# TAB 2 — IMBALANCE HANDLING
# ═══════════════════════════════════════════════
with t2:
    st.subheader("Class Imbalance Handling")
    st.markdown("""
    <div class="info">
    <b>Problem:</b> Only ~1.5% of transactions are fraud. A naive model that always predicts
    "not fraud" would be 98.5% accurate but detect zero frauds!<br>
    <b>Solution:</b> <code>class_weight='balanced'</code> — gives fraud cases ~66× more
    weight during training so the model pays proper attention to them.
    </div>
    """, unsafe_allow_html=True)

    n0     = int((df_raw["is_fraud"]==0).sum())
    n1     = int((df_raw["is_fraud"]==1).sum())
    w1     = total / (2 * n1)    # effective weight for fraud class
    w0     = total / (2 * n0)    # effective weight for legit class
    eff0   = int(n0 * w0)
    eff1   = int(n1 * w1)

    cl, cr = st.columns(2)
    with cl:
        st.markdown("**Before — Raw Data**")
        st.metric("Legitimate (0)", f"{n0:,}", f"{n0/total*100:.2f}%")
        st.metric("Fraud (1)",      f"{n1}",   f"{n1/total*100:.2f}%")
        st.metric("Imbalance ratio", f"{n0//n1}:1")
    with cr:
        st.markdown("**After — class_weight='balanced'**")
        st.metric("Legitimate weight", f"{w0:.4f}")
        st.metric("Fraud weight",      f"{w1:.4f}")
        st.metric("Effective ratio",   "≈ 1:1")

    fig = go.Figure()
    for label, before, after, color in [
        ("Legitimate", n0, eff0, "#22c55e"),
        ("Fraud",      n1, eff1, "#ef4444"),
    ]:
        fig.add_trace(go.Bar(name=f"{label} — Before", x=["Before"], y=[before],
                             marker_color=color, opacity=0.5))
        fig.add_trace(go.Bar(name=f"{label} — After (weighted)", x=["After"], y=[after],
                             marker_color=color))
    fig.update_layout(title="Class Counts: Before vs After Weighting",
                      barmode="group", plot_bgcolor="white", height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Why not SMOTE?**")
    st.table(pd.DataFrame({
        "Method":     ["class_weight='balanced' ✅", "SMOTE", "Undersampling"],
        "How it works": [
            "Multiplies loss for minority class during training",
            "Creates synthetic fraud samples",
            "Removes majority class rows",
        ],
        "Pro":  ["No data change, built-in, fast", "More training data",  "Simpler"],
        "Con":  ["Doesn't generate new samples", "Synthetic ≠ real",    "Loses real data"],
    }))


# ═══════════════════════════════════════════════
# TAB 3 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════
with t3:
    st.subheader("Feature Engineering")
    st.markdown('<div class="info">8 new features were created from the original 9 columns '
                'to give the ML models richer patterns to learn from.</div>', unsafe_allow_html=True)

    feat_desc = pd.DataFrame({
        "Feature":    ["log_amount","part_of_day","is_night","age_group",
                       "is_high_value","low_trust_device","risk_score","amount_velocity_interact"],
        "From":       ["amount","transaction_hour","transaction_hour","cardholder_age",
                       "amount","device_trust_score","4 columns","amount × velocity"],
        "Why useful": [
            "Reduces right skew — linear models prefer symmetric data",
            "Night/Evening show higher fraud rates than Morning/Afternoon",
            "Binary flag for 11pm–5am — strong fraud time window",
            "Age buckets capture non-linear demographic patterns",
            "Top 10% amounts are disproportionately targeted",
            "Score < 50 is a known indicator of risky devices",
            "#1 feature: combines foreign + mismatch + trust + velocity",
            "High spend + high velocity = classic fraud pattern",
        ],
    })
    st.dataframe(feat_desc, use_container_width=True, hide_index=True)

    st.subheader("Engineered Dataset Preview")
    new_cols = ["amount","transaction_hour","merchant_category","is_fraud",
                "log_amount","part_of_day","is_night","age_group",
                "is_high_value","low_trust_device","risk_score","amount_velocity_interact"]
    show = [c for c in new_cols if c in df_feat.columns]
    st.dataframe(df_feat[show].head(15), use_container_width=True)

    c1_, c2_ = st.columns(2)
    with c1_:
        fig = px.histogram(df_feat, x="risk_score", color="is_fraud",
                           barmode="overlay", nbins=30,
                           color_discrete_map={0:"#22c55e",1:"#ef4444"},
                           title="Risk Score: Fraud vs Legitimate")
        fig.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    with c2_:
        fig2 = px.histogram(df_feat, x="log_amount", color="is_fraud",
                            barmode="overlay", nbins=30,
                            color_discrete_map={0:"#22c55e",1:"#ef4444"},
                            title="Log-Amount: Fraud vs Legitimate")
        fig2.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("ML Model Performance")
    st.dataframe(results_df.style.background_gradient(
        cmap="RdYlGn", subset=["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]
    ).format({c:"{:.4f}" for c in ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]}),
        use_container_width=True, hide_index=True)
    st.success(f"🏆 Best model: **{best_name}** (ROC-AUC = {results_df.iloc[0]['ROC-AUC']:.4f})")


# ═══════════════════════════════════════════════
# TAB 4 — PREDICTIONS
# ═══════════════════════════════════════════════
with t4:
    st.subheader("Predict Fraud on a New Transaction")
    st.markdown('<div class="info">Enter transaction details. All feature engineering '
                'is applied automatically — same as the notebook.</div>', unsafe_allow_html=True)

    with st.form("pred_form"):
        r1a, r1b, r1c = st.columns(3)
        amount    = r1a.number_input("Amount ($)", 0.0, 50000.0, 150.0, 10.0)
        hour      = r1b.slider("Transaction Hour", 0, 23, 14)
        r1b.caption(f"→ {get_part_of_day(hour)}")
        category  = r1c.selectbox("Merchant Category", MERCHANT_CATS)

        r2a, r2b, r2c = st.columns(3)
        age       = r2a.slider("Cardholder Age", 18, 90, 35)
        foreign   = r2b.selectbox("Foreign Transaction", [0,1], format_func=lambda x: "Yes" if x else "No")
        mismatch  = r2c.selectbox("Location Mismatch", [0,1], format_func=lambda x: "Yes" if x else "No")

        r3a, r3b, r3c = st.columns(3)
        trust     = r3a.slider("Device Trust Score", 25, 100, 75)
        velocity  = r3b.slider("Velocity Last 24h", 0, 10, 1)
        sel_model = r3c.selectbox("Model", list(trained.keys()),
                                  index=list(trained.keys()).index(best_name))

        submitted = st.form_submit_button("🔍 Predict", type="primary", use_container_width=True)

    if submitted:
        # Build raw DataFrame → engineer features → preprocess → predict
        raw = pd.DataFrame([{
            "amount": amount, "transaction_hour": hour, "merchant_category": category,
            "foreign_transaction": foreign, "location_mismatch": mismatch,
            "device_trust_score": trust, "velocity_last_24h": velocity, "cardholder_age": age,
        }])
        feat_row, _ = engineer_features(raw, threshold=amt_thr)
        X_new       = feat_row[FEATURE_COLS]
        X_proc      = preprocessor.transform(X_new)
        model       = trained[sel_model]
        proba       = float(model.predict_proba(X_proc)[0,1])
        pred        = int(proba >= 0.5)

        risk  = "🔴 High Risk" if proba >= 0.6 else "🟡 Medium Risk" if proba >= 0.3 else "🟢 Low Risk"
        ba, bb, bc = st.columns(3)

        with ba:
            if pred == 1:
                st.markdown(f'<div class="fraud-badge"><h2>🚨 FRAUD</h2><h3>{risk}</h3></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="legit-badge"><h2>✅ LEGITIMATE</h2><h3>{risk}</h3></div>',
                            unsafe_allow_html=True)

        with bb:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba*100,
                number={"suffix":"%","font":{"size":36}},
                title={"text":"Fraud Probability"},
                gauge={
                    "axis": {"range":[0,100]},
                    "bar":  {"color":"#ef4444" if pred else "#22c55e"},
                    "steps": [{"range":[0,30],"color":"#dcfce7"},
                               {"range":[30,60],"color":"#fef9c3"},
                               {"range":[60,100],"color":"#fee2e2"}],
                    "threshold": {"line":{"color":"black","width":3},"value":50},
                }
            ))
            fig.update_layout(height=240, margin=dict(t=30,b=0,l=10,r=10))
            st.plotly_chart(fig, use_container_width=True)

        with bc:
            st.markdown("**Engineered features:**")
            for k in ["log_amount","part_of_day","is_night","age_group",
                      "is_high_value","low_trust_device","risk_score"]:
                st.markdown(f"- `{k}` = **{feat_row[k].iloc[0]}**")

        # All models comparison
        st.markdown("**All 6 models vote:**")
        votes = []
        for name, m in trained.items():
            p = float(m.predict_proba(X_proc)[0,1])
            votes.append({"Model": name, "Probability": f"{p:.4f}",
                          "Verdict": "🚨 Fraud" if p >= 0.5 else "✅ Legit"})
        st.dataframe(pd.DataFrame(votes), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════
# TAB 5 — RAG CHATBOT
# ═══════════════════════════════════════════════
with t5:
    st.subheader("RAG Chatbot — Ask About the Dataset")
    st.markdown("""
    <div class="info">
    <b>How it works:</b>
    1. Dataset statistics → converted into text documents (knowledge base)<br>
    2. TF-IDF embeddings → each document becomes a numerical vector<br>
    3. FAISS index → stores vectors for fast similarity search<br>
    4. Your question → matched against vectors → top documents retrieved<br>
    5. Retrieved facts → returned as the grounded answer (no hallucination)
    </div>
    """, unsafe_allow_html=True)

    # Initialise chatbot
    with st.spinner("Building RAG index…"):
        bot, err = get_chatbot()

    if err:
        st.error(f"❌ {err}")
        st.code("pip install faiss-cpu scikit-learn")
    else:
        st.success("✅ Chatbot ready — no API key required!")

        # Quick questions
        st.markdown("**Quick questions (click to ask):**")
        sample_qs = [
            "What percentage of transactions are fraudulent?",
            "Which merchant category has the highest fraud rate?",
            "What is the risk score formula?",
            "Which ML model performed best?",
            "How does device trust score affect fraud?",
            "Do fraud transactions happen more at night?",
        ]
        cols = st.columns(3)
        quick = None
        for i, q in enumerate(sample_qs):
            if cols[i%3].button(q, key=f"qb_{i}"):
                user_q = q  # directly set the question
                answer = bot.chat(user_q)
                st.session_state.msgs.append({"q": user_q, "a": answer})
                # st.experimental_rerun()  # rerun so chat appears immediately

        st.markdown("---")

        # Chat history
        if "msgs" not in st.session_state:
            st.session_state.msgs = []

        user_q = st.text_input("Your question:", value=quick or "",
                               placeholder="Ask anything about the fraud dataset…")

        ca, cb = st.columns([1,1])
        ask     = ca.button("Send", type="primary", use_container_width=True)
        clear   = cb.button("Clear chat", use_container_width=True)

        if clear:
            st.session_state.msgs = []
            # st.rerun()

        if ask and user_q.strip():
            answer = bot.chat(user_q)
            st.session_state.msgs.append({"q": user_q, "a": answer})

        # Display messages newest-first
        for turn in reversed(st.session_state.msgs):
            st.markdown(f"**You:** {turn['q']}")
            st.info(f"🤖 **Bot:** {turn['a']}")
            st.markdown("---")
