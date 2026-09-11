"""
RAG Chatbot for Credit Card Fraud Dataset
==========================================
What is RAG?
  Retrieval-Augmented Generation = Search your data first, then answer.
  Step 1 → Convert dataset statistics into text "documents"
  Step 2 → Embed them as vectors (numbers) so we can search by meaning
  Step 3 → When user asks, find the most relevant documents
  Step 4 → Return the retrieved facts as the grounded answer

Why FAISS?
  • Free, local, no server needed — just: pip install faiss-cpu
  • Very fast similarity search, works completely offline
  • No API key required (unlike OpenAI/Anthropic embeddings)

Why TF-IDF embeddings (not a neural model)?
  • Works with zero setup, zero cost, zero API keys
  • TF-IDF: words frequent in ONE doc but rare elsewhere get high weight
  • Accurate enough for factual dataset Q&A
  • If you want a neural model later, just swap the vectorizer

How to run:
  pip install faiss-cpu scikit-learn pandas numpy openpyxl
  python rag_chatbot.py
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# ──────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (identical to notebook — do NOT change)
# ──────────────────────────────────────────────────────────────────

def get_part_of_day(hour):
    if 5 <= hour <= 11:    return "Morning"
    elif 12 <= hour <= 17: return "Afternoon"
    elif 18 <= hour <= 22: return "Evening"
    else:                  return "Night"


def engineer_features(df):
    """Apply all feature engineering from the notebook."""
    df = df.copy()
    df["part_of_day"]              = df["transaction_hour"].apply(get_part_of_day)
    df["is_night"]                 = df["transaction_hour"].apply(lambda h: 1 if h >= 23 or h <= 4 else 0)
    df["age_group"]                = pd.cut(df["cardholder_age"],
                                            bins=[0,25,35,50,65,120],
                                            labels=["Young","Adult","Middle-aged","Senior","Elderly"]).astype(str)
    threshold                      = df["amount"].quantile(0.90)
    df["is_high_value"]            = (df["amount"] > threshold).astype(int)
    df["low_trust_device"]         = (df["device_trust_score"] < 50).astype(int)
    max_vel                        = max(df["velocity_last_24h"].max(), 1)
    df["risk_score"]               = (
        df["foreign_transaction"] * 0.3 +
        df["location_mismatch"]   * 0.3 +
        (1 - df["device_trust_score"] / 100) * 0.2 +
        df["velocity_last_24h"] / max_vel * 0.2
    ).round(4)
    df["amount_velocity_interact"] = (df["amount"] * df["velocity_last_24h"]).round(4)
    df["log_amount"]               = np.log1p(df["amount"])
    return df, float(threshold)


# ──────────────────────────────────────────────────────────────────
# STEP 1: BUILD KNOWLEDGE DOCUMENTS
# Convert dataset statistics into human-readable text paragraphs.
# Each paragraph = one "document" in the knowledge base.
# ──────────────────────────────────────────────────────────────────

def build_documents(df):
    """
    Convert dataset statistics into a list of knowledge documents.
    Returns a list of dicts: {'topic', 'keywords', 'text'}
    """
    docs  = []
    fraud = df[df["is_fraud"] == 1]
    legit = df[df["is_fraud"] == 0]
    total = len(df)
    n_fr  = int(df["is_fraud"].sum())
    pct   = df["is_fraud"].mean() * 100

    # 1. Overview
    docs.append({"topic": "overview",
        "keywords": "total dataset overview size transactions fraud rate percent imbalance how many",
        "text": (
            f"The dataset has {total:,} transactions: {n_fr} fraudulent ({pct:.2f}%) "
            f"and {total-n_fr:,} legitimate ({100-pct:.2f}%). "
            f"Imbalance ratio is {(total-n_fr)//n_fr}:1 (legitimate:fraud). "
            f"Handled with class_weight='balanced' in all ML models."
        )})

    # 2. Amount
    docs.append({"topic": "amount",
        "keywords": "amount money dollar value high value spend transaction cost price average",
        "text": (
            f"Overall amount: mean=${df['amount'].mean():.2f}, median=${df['amount'].median():.2f}, "
            f"min=${df['amount'].min():.2f}, max=${df['amount'].max():.2f}. "
            f"Fraud mean=${fraud['amount'].mean():.2f}, legitimate mean=${legit['amount'].mean():.2f}. "
            f"High-value (top 10%, >${df['amount'].quantile(0.9):.2f}) fraud rate: "
            f"{df[df['is_high_value']==1]['is_fraud'].mean()*100:.2f}%."
        )})

    # 3. Merchant category
    cs = df.groupby("merchant_category")["is_fraud"].agg(["sum","mean"]).reset_index()
    hi = cs.loc[cs["mean"].idxmax(), "merchant_category"]
    lo = cs.loc[cs["mean"].idxmin(), "merchant_category"]
    cl = " | ".join(f"{r['merchant_category']}: {int(r['sum'])} fraud ({r['mean']*100:.2f}%)" for _,r in cs.iterrows())
    docs.append({"topic": "merchant_category",
        "keywords": "merchant category store electronics travel grocery food clothing shop type",
        "text": f"Fraud by merchant category: {cl}. Highest fraud: {hi}. Lowest fraud: {lo}."
    })

    # 4. Foreign transactions
    fr = df[df["foreign_transaction"]==1]["is_fraud"].mean()*100
    dr = df[df["foreign_transaction"]==0]["is_fraud"].mean()*100
    docs.append({"topic": "foreign_transaction",
        "keywords": "foreign international abroad domestic country overseas outside",
        "text": (
            f"Foreign transaction fraud rate: {fr:.2f}%. "
            f"Domestic fraud rate: {dr:.2f}%. "
            f"Foreign is {fr/max(dr,0.001):.1f}x more likely to be fraud."
        )})

    # 5. Location mismatch
    mr  = df[df["location_mismatch"]==1]["is_fraud"].mean()*100
    nmr = df[df["location_mismatch"]==0]["is_fraud"].mean()*100
    docs.append({"topic": "location_mismatch",
        "keywords": "location mismatch place address geography where usual",
        "text": (
            f"Location mismatch fraud rate: {mr:.2f}%. "
            f"No mismatch fraud rate: {nmr:.2f}%. "
            f"Location mismatch is a strong fraud signal."
        )})

    # 6. Device trust score
    ft = fraud["device_trust_score"].mean()
    lt = legit["device_trust_score"].mean()
    ltr = df[df["device_trust_score"]<50]["is_fraud"].mean()*100
    docs.append({"topic": "device_trust",
        "keywords": "device trust score trusted untrusted phone browser low trust second important",
        "text": (
            f"Device trust score (25-100): fraud avg={ft:.1f}, legitimate avg={lt:.1f}. "
            f"Low trust (<50) fraud rate: {ltr:.2f}%. "
            f"2nd most important feature in the Random Forest model."
        )})

    # 7. Time patterns
    hr    = df.groupby("transaction_hour")["is_fraud"].mean()
    ph    = int(hr.idxmax())
    pr    = hr.max()*100
    nm    = df["transaction_hour"].isin(list(range(23,24))+list(range(0,5)))
    nr    = df[nm]["is_fraud"].mean()*100
    pod   = df.groupby("part_of_day")["is_fraud"].mean()*100
    pods  = " | ".join(f"{p}: {r:.2f}%" for p,r in pod.items())
    docs.append({"topic": "time_patterns",
        "keywords": "hour time night morning afternoon evening when late peak part day",
        "text": (
            f"Peak fraud hour: {ph}:00 ({pr:.2f}% rate). "
            f"Night (11pm-5am) fraud rate: {nr:.2f}%. "
            f"By part of day: {pods}."
        )})

    # 8. Velocity
    fv = fraud["velocity_last_24h"].mean()
    lv = legit["velocity_last_24h"].mean()
    hv = df[df["velocity_last_24h"]>=5]["is_fraud"].mean()*100
    docs.append({"topic": "velocity",
        "keywords": "velocity frequency 24h transactions per often multiple repeated rapid speed",
        "text": (
            f"Avg transactions in last 24h: fraud={fv:.2f}, legitimate={lv:.2f}. "
            f"High velocity (5+ in 24h) fraud rate: {hv:.2f}%. "
            f"Rapid repeated transactions is a classic fraud pattern."
        )})

    # 9. Feature engineering
    docs.append({"topic": "feature_engineering",
        "keywords": "feature engineering log risk score night age group high value interaction created new",
        "text": (
            "8 engineered features added to the original columns: "
            "log_amount (log-transform), part_of_day (time label), is_night (binary), "
            "age_group (age bucket), is_high_value (top 10% amount), "
            "low_trust_device (trust<50), risk_score (composite of 4 signals), "
            "amount_velocity_interact (amount × velocity)."
        )})

    # 10. ML models
    docs.append({"topic": "ml_models",
        "keywords": "model accuracy recall precision f1 auc roc best performance logistic "
                    "random forest gradient boosting svm naive bayes decision tree machine learning",
        "text": (
            "6 models trained: Logistic Regression, Decision Tree, Random Forest, "
            "Gradient Boosting, Naive Bayes, SVM — all with class_weight='balanced'. "
            "Gradient Boosting: best ROC-AUC ~0.997 (5-fold cross-validation). "
            "Top features: risk_score, device_trust_score, part_of_day_Night, is_night, low_trust_device. "
            "Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC."
        )})

    # 11. Risk score
    docs.append({"topic": "risk_score",
        "keywords": "risk score formula composite calculate weight combine top important number one",
        "text": (
            f"risk_score = foreign_transaction×0.3 + location_mismatch×0.3 "
            f"+ (1 - device_trust_score/100)×0.2 + velocity/max_velocity×0.2. "
            f"Range: {df['risk_score'].min():.2f} to {df['risk_score'].max():.2f}. "
            f"#1 feature in Random Forest (importance ~0.35)."
        )})

    # 12. Age
    ag  = df.groupby("age_group")["is_fraud"].mean()*100
    ags = " | ".join(f"{g}: {r:.2f}%" for g,r in ag.items())
    docs.append({"topic": "cardholder_age",
        "keywords": "age cardholder young adult senior elderly old group demographic",
        "text": (
            f"Ages {int(df['cardholder_age'].min())}–{int(df['cardholder_age'].max())} "
            f"(mean={df['cardholder_age'].mean():.1f}). "
            f"Fraud rate by age group: {ags}."
        )})

    return docs


# ──────────────────────────────────────────────────────────────────
# STEP 2 & 3: FAISS INDEX + RETRIEVAL
# ──────────────────────────────────────────────────────────────────

class FraudRAGChatbot:
    """
    Simple RAG chatbot for the credit card fraud dataset.
    No external API keys required — runs fully offline.

    Usage:
        bot = FraudRAGChatbot(df_feat)
        print(bot.chat("What is the fraud rate?"))
    """

    # Keywords that clearly signal an out-of-scope question
    OUT_OF_SCOPE = [
        r"\bweather\b", r"\bsports\b", r"\bcooking\b", r"\brecipe\b",
        r"\bhistory\b", r"\bgeograph\b", r"\bphysic\b", r"\bchemist\b",
        r"\bbiolog\b", r"\bmusic\b", r"\bmovie\b", r"\bnews\b",
        r"\bstock\b", r"\bcrypto\b", r"\bpet\b", r"\banimal\b",
        r"\bsoccer\b", r"\bbaseball\b", r"\bfootball\b",
    ]

    # Keywords that signal the question is about the dataset
    IN_SCOPE = [
        "fraud", "transaction", "amount", "merchant", "device", "trust",
        "velocity", "location", "mismatch", "foreign", "cardholder", "age",
        "feature", "model", "accuracy", "recall", "precision", "auc",
        "dataset", "risk", "score", "legitimate", "predict", "category",
        "hour", "night", "imbalance", "class", "ml", "logistic",
        "forest", "gradient", "svm", "naive", "rate", "percent",
        "average", "mean", "count", "data", "engineered", "night",
    ]

    def __init__(self, df):
        """
        Build the RAG index from a cleaned + feature-engineered DataFrame.
        """
        try:
            import faiss
            self._faiss = faiss
        except ImportError:
            raise ImportError("Run: pip install faiss-cpu")

        # Build knowledge documents from the dataset
        self.docs = build_documents(df)

        # TF-IDF vectoriser: converts text → weighted word-frequency vectors
        # ngram_range=(1,2) captures single words AND 2-word phrases (e.g. "fraud rate")
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=3000,
            sublinear_tf=True       # log(tf) — prevents very common words dominating
        )

        # Combine keywords + text into the corpus for richer matching
        corpus = [d["keywords"] + " " + d["text"] for d in self.docs]
        matrix = self.vectorizer.fit_transform(corpus).toarray().astype("float32")

        # Normalise rows to unit length → dot product = cosine similarity
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        matrix = matrix / norms

        # Build FAISS flat inner-product index
        self.index = self._faiss.IndexFlatIP(matrix.shape[1])
        self.index.add(matrix)
        print(f"✅ Chatbot ready — {len(self.docs)} documents indexed.")

    def _scope_check(self, question):
        """Return True if clearly out of scope."""
        q = question.lower()
        if any(kw in q for kw in self.IN_SCOPE):
            return False                           # definitely in scope
        if any(re.search(p, q) for p in self.OUT_OF_SCOPE):
            return True                            # clearly out of scope
        return False                               # default: try to answer

    def chat(self, question):
        """
        Answer a question about the fraud dataset.
        Returns a string answer (grounded in dataset facts) or an out-of-scope message.
        """
        if not question.strip():
            return "Please enter a question."

        if self._scope_check(question):
            return (
                "⚠️ Out of context: This chatbot only answers questions about the "
                "credit card fraud dataset. Ask about fraud rates, transaction amounts, "
                "merchant categories, ML models, feature engineering, etc."
            )

        # Vectorise the question with the same vocabulary as the docs
        q_vec = self.vectorizer.transform([question]).toarray().astype("float32")
        norm  = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec /= norm

        # Search FAISS for top-2 most similar documents
        scores, indices = self.index.search(q_vec, k=2)

        if not indices[0].any() or scores[0][0] < 0.05:
            return (
                "I couldn't find relevant information for that question. "
                "Try asking about fraud rates, amounts, device trust, ML models, "
                "or feature engineering."
            )

        # Build answer from top retrieved document(s)
        top_doc   = self.docs[indices[0][0]]
        answer    = top_doc["text"]

        # Add the second doc if its score is also high (>0.1)
        if scores[0][1] > 0.10 and indices[0][1] >= 0:
            second = self.docs[indices[0][1]]["text"]
            answer += " | " + second

        topic = top_doc["topic"].replace("_", " ").title()
        return f"📊 [{topic}] {answer}"


# ──────────────────────────────────────────────────────────────────
# STANDALONE DEMO
# ──────────────────────────────────────────────────────────────────

def load_data(path="credit_card_fraud_10k.xls"):
    try:
        df = pd.read_excel(path)
    except Exception:
        df = pd.read_csv(path.replace(".xls", ".csv"))
    df.columns = df.columns.str.strip().str.lower()
    return df


if __name__ == "__main__":
    df_raw       = load_data()
    df_feat, _   = engineer_features(df_raw)
    bot          = FraudRAGChatbot(df_feat)

    questions = [
        "What percentage of transactions are fraudulent?",
        "Which merchant category has the highest fraud rate?",
        "How does device trust score relate to fraud?",
        "What is the risk score formula?",
        "What is the best ML model?",
        "Tell me about the weather.",     # Out of scope
        "What is 2+2?",                   # Out of scope
    ]
    print("\n" + "="*60)
    for q in questions:
        print(f"\nQ: {q}\nA: {bot.chat(q)}\n" + "-"*50)
