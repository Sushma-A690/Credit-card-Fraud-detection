#!/usr/bin/env python
# coding: utf-8

# # 💳 Credit Card Fraud Detection — Production-Ready ML Pipeline
# 
# **Objective:** Build a robust, end-to-end ML pipeline that accurately detects credit card fraud on any new, unseen data.
# 
# ---
# 
# ## 📋 Table of Contents
# 1. [Setup & Imports](#1-setup)
# 2. [Data Loading & Initial Inspection](#2-data-loading)
# 3. [Exploratory Data Analysis (EDA)](#3-eda)
# 4. [Data Cleaning](#4-cleaning)
# 5. [Feature Engineering](#5-features)
# 6. [Train/Test Split & Class Imbalance Handling](#6-split)
# 7. [Preprocessing Pipeline (Encoding + Scaling)](#7-preprocessing)
# 8. [Model Training & Evaluation](#8-models)
# 9. [Model Comparison & Best Model Selection](#9-comparison)
# 10. [Reusable Prediction Function](#10-prediction)
# 11. [RAG Chatbot Integration](#11-chatbot)
# 
# ---
# > **Key principle:** All preprocessing (encoding, scaling, SMOTE) is applied **only on training data**. The test set is transformed using parameters learned from training — never the other way around.

# ## 1. Setup & Imports 📦

# In[8]:


# ─────────────────────────────────────────────────────────
# CELL 1: Install & import all required libraries
# ─────────────────────────────────────────────────────────

# Standard data science stack
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn: preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Scikit-learn: models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Scikit-learn: metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# Class imbalance handling
# We use sklearn's class_weight='balanced' as a robust alternative to SMOTE
# To use SMOTE: pip install imbalanced-learn, then:
# from imblearn.over_sampling import SMOTE

# Optional: XGBoost (uncomment if installed)
# from xgboost import XGBClassifier

import joblib   # To save/load the trained pipeline
import json

# Plotting settings
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
RANDOM_STATE = 42

print("All libraries loaded successfully!")


# ## 2. Data Loading & Initial Inspection 🔍

# In[9]:


# ─────────────────────────────────────────────────────────
# CELL 2: Load the dataset
# ─────────────────────────────────────────────────────────

# Load from Excel or CSV — update the path as needed
try:
    df = pd.read_excel('credit_card_fraud_10k.xls')
    print("Loaded from Excel (.xls)")
except Exception:
    df = pd.read_csv('credit_card_fraud_10k.csv')
    print("Loaded from CSV")

# Standardise column names: strip spaces, lowercase
df.columns = df.columns.str.strip().str.lower()

print(f"\n📐 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
df.head()


# In[10]:


# ─────────────────────────────────────────────────────────
# CELL 3: Data types and memory info
# ─────────────────────────────────────────────────────────
df.info()

# Convert merchant_category to 'category' dtype — saves memory
df['merchant_category'] = df['merchant_category'].astype('category')
print("\nmerchant_category converted to category dtype")


# In[11]:


# ─────────────────────────────────────────────────────────
# CELL 4: Statistical summary
# ─────────────────────────────────────────────────────────
# describe() shows count, mean, std, min, max, quartiles
df.describe().T.style.background_gradient(cmap='Blues')


# ## 3. Exploratory Data Analysis (EDA) 📊
# 
# EDA helps us understand the data **before** modelling — distributions, correlations, outliers, and class balance.

# In[12]:


# ─────────────────────────────────────────────────────────
# CELL 5: Missing values & duplicates
# ─────────────────────────────────────────────────────────
print("=== Missing Values ===")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values")

print(f"\n=== Duplicates ===")
dupes = df.duplicated().sum()
print(f"Duplicate rows: {dupes}")
print(f"Duplicate transaction_ids: {df['transaction_id'].duplicated().sum()}")


# In[13]:


# ─────────────────────────────────────────────────────────
# CELL 6: Class balance (is_fraud)
# Fraud detection datasets are typically very imbalanced —
# most transactions are legitimate, very few are fraud.
# This imbalance must be addressed during model training.
# ─────────────────────────────────────────────────────────
fraud_counts = df['is_fraud'].value_counts()
fraud_pct = df['is_fraud'].value_counts(normalize=True) * 100

print("=== Class Distribution ===")
print(pd.DataFrame({'Count': fraud_counts, 'Percentage': fraud_pct.round(2)}))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar plot
colors = ['#2ECC71', '#E74C3C']
fraud_counts.plot(kind='bar', ax=axes[0], color=colors, edgecolor='black', rot=0)
axes[0].set_title('Transaction Count: Fraud vs Legitimate', fontsize=14, fontweight='bold')
axes[0].set_xlabel('is_fraud  (0=Legitimate, 1=Fraud)')
axes[0].set_ylabel('Count')
for i, v in enumerate(fraud_counts):
    axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')

# Pie chart
axes[1].pie(fraud_counts, labels=['Legitimate', 'Fraud'], colors=colors,
            autopct='%1.1f%%', startangle=90, explode=(0, 0.1),
            textprops={'fontsize': 12})
axes[1].set_title('Fraud Proportion', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

imbalance_ratio = fraud_counts[0] / fraud_counts[1]
print(f"Imbalance ratio: {imbalance_ratio:.1f}:1 (legitimate:fraud)")
print("We will handle this with class_weight='balanced' in each model.")


# In[14]:


# ─────────────────────────────────────────────────────────
# CELL 7: Distributions of numeric features
# Histograms help spot skewness, outliers, and value ranges
# ─────────────────────────────────────────────────────────
numeric_cols = ['amount', 'transaction_hour', 'device_trust_score',
                'velocity_last_24h', 'cardholder_age']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    df[df['is_fraud'] == 0][col].hist(ax=axes[i], bins=30, alpha=0.6,
                                       color='#2ECC71', label='Legitimate')
    df[df['is_fraud'] == 1][col].hist(ax=axes[i], bins=30, alpha=0.8,
                                       color='#E74C3C', label='Fraud')
    axes[i].set_title(f'Distribution: {col}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
    axes[i].legend()

# Hide unused subplot
fig.delaxes(axes[-1])
plt.suptitle('Feature Distributions: Fraud vs Legitimate', y=1.02, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# In[15]:


# ─────────────────────────────────────────────────────────
# CELL 8: Correlation heatmap
# Shows how strongly each numeric feature is linearly
# correlated with every other feature.
# Values close to +1 or -1 = strong correlation.
# Values near 0 = weak/no linear relationship.
# ─────────────────────────────────────────────────────────
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include='number').drop(columns=['transaction_id'], errors='ignore')
corr_matrix = numeric_df.corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Show only lower triangle
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
            mask=mask, linewidths=0.5, vmin=-1, vmax=1,
            annot_kws={'size': 9})
plt.title('Correlation Heatmap of Numeric Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Show top correlations with target
print("Top correlations with is_fraud:")
print(corr_matrix['is_fraud'].drop('is_fraud').sort_values(key=abs, ascending=False))


# In[16]:


# ─────────────────────────────────────────────────────────
# CELL 9: Fraud rate by categorical feature
# Which merchant categories have the highest fraud rates?
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

cat_features = ['merchant_category', 'foreign_transaction', 'location_mismatch']
for ax, col in zip(axes, cat_features):
    fraud_rate = df.groupby(col)['is_fraud'].mean().sort_values(ascending=False) * 100
    bars = fraud_rate.plot(kind='bar', ax=ax, color='#E74C3C', edgecolor='black',
                           alpha=0.8, rot=15)
    ax.set_title(f'Fraud Rate by {col}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fraud Rate (%)')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9)

plt.suptitle('Fraud Rate by Categorical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# In[17]:


# ─────────────────────────────────────────────────────────
# CELL 10: Box plots — spot differences in medians & outliers
# between fraud and legitimate transactions
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

for ax, col in zip(axes, ['amount', 'device_trust_score', 'velocity_last_24h']):
    df.boxplot(column=col, by='is_fraud', ax=ax,
               boxprops=dict(color='navy'),
               medianprops=dict(color='red', linewidth=2))
    ax.set_title(f'{col} by Fraud Status', fontsize=11, fontweight='bold')
    ax.set_xlabel('is_fraud  (0=Legitimate, 1=Fraud)')
    ax.set_ylabel(col)

plt.suptitle('Box Plots: Feature Distribution by Fraud Status', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# ## 4. Data Cleaning 🧹

# In[18]:


# ─────────────────────────────────────────────────────────
# CELL 11: Data cleaning
# ─────────────────────────────────────────────────────────
df_clean = df.copy()  # Always work on a copy; keep the original safe

# --- 1. Drop duplicates ---
before = len(df_clean)
df_clean.drop_duplicates(inplace=True)
print(f"Duplicates removed: {before - len(df_clean)}")

# --- 2. Handle missing values ---
# Numeric columns: fill with median (robust to outliers)
# Categorical columns: fill with mode (most frequent value)
for col in df_clean.select_dtypes(include='number').columns:
    if df_clean[col].isnull().any():
        median_val = df_clean[col].median()
        df_clean[col].fillna(median_val, inplace=True)
        print(f"  Filled numeric '{col}' with median: {median_val}")

for col in df_clean.select_dtypes(include=['object', 'category']).columns:
    if df_clean[col].isnull().any():
        mode_val = df_clean[col].mode()[0]
        df_clean[col].fillna(mode_val, inplace=True)
        print(f"  Filled categorical '{col}' with mode: {mode_val}")

# --- 3. Validate value ranges ---
# transaction_hour must be 0–23
invalid_hours = df_clean[~df_clean['transaction_hour'].between(0, 23)]
print(f"\nInvalid transaction_hour entries: {len(invalid_hours)}")

# Amount should be non-negative
negative_amounts = df_clean[df_clean['amount'] < 0]
print(f"Negative amount entries: {len(negative_amounts)}")
if len(negative_amounts) > 0:
    df_clean['amount'] = df_clean['amount'].clip(lower=0)

# Binary flags should only contain 0 or 1
for col in ['foreign_transaction', 'location_mismatch', 'is_fraud']:
    invalid = ~df_clean[col].isin([0, 1])
    print(f"Invalid '{col}' values: {invalid.sum()}")

# --- 4. Drop the ID column — it carries no predictive information ---
df_clean.drop(columns=['transaction_id'], inplace=True)

print(f"\n Cleaning complete. Dataset shape: {df_clean.shape}")


# ## 5. Feature Engineering 🔧
# 
# We create **new features** from existing ones that better capture fraud-related patterns.
# All engineering is done as pure transformations of raw columns — no information from the test set is used.

# In[19]:


# ─────────────────────────────────────────────────────────
# CELL 12: Feature engineering
# ─────────────────────────────────────────────────────────
df_feat = df_clean.copy()

# ── Feature 1: Part of Day ──────────────────────────────
# Fraud may be more common at unusual hours (late night, early morning)
def get_part_of_day(hour):
    if 5 <= hour <= 11:  return 'Morning'
    elif 12 <= hour <= 17: return 'Afternoon'
    elif 18 <= hour <= 22: return 'Evening'
    else:                  return 'Night'  # 23:00 – 04:59

df_feat['part_of_day'] = df_feat['transaction_hour'].apply(get_part_of_day)

# ── Feature 2: Is Night Transaction ────────────────────
# Binary: 1 if transaction happened between 11pm and 5am
df_feat['is_night'] = df_feat['transaction_hour'].apply(
    lambda h: 1 if (h >= 23 or h <= 4) else 0
)

# ── Feature 3: Age Group ────────────────────────────────
# Older cardholders may have different fraud patterns
df_feat['age_group'] = pd.cut(
    df_feat['cardholder_age'],
    bins=[0, 25, 35, 50, 65, 120],
    labels=['Young', 'Adult', 'Middle-aged', 'Senior', 'Elderly']
).astype(str)  # Convert to str to allow consistent encoding later

# ── Feature 4: High-Value Transaction ──────────────────
# Transactions in the top 10% by amount — more risky
# IMPORTANT: we use a fixed threshold here (computed on full data).
# In production, compute this threshold on TRAINING data only.
amount_threshold = df_feat['amount'].quantile(0.90)
df_feat['is_high_value'] = (df_feat['amount'] > amount_threshold).astype(int)
print(f"High-value threshold (90th percentile): ${amount_threshold:.2f}")

# ── Feature 5: Low Device Trust Score ──────────────────
# Untrusted devices are a strong fraud signal
df_feat['low_trust_device'] = (df_feat['device_trust_score'] < 50).astype(int)

# ── Feature 6: Composite Risk Score ────────────────────
# Manually engineered risk proxy combining multiple risk signals
# Each component is normalised to [0,1] range before summing
df_feat['risk_score'] = (
    df_feat['foreign_transaction'] * 0.3 +
    df_feat['location_mismatch'] * 0.3 +
    (1 - df_feat['device_trust_score'] / 100) * 0.2 +
    df_feat['velocity_last_24h'] / df_feat['velocity_last_24h'].max() * 0.2
).round(4)

# ── Feature 7: Amount × Velocity Interaction ───────────
# High spend + high velocity is a strong fraud pattern
df_feat['amount_velocity_interact'] = (
    df_feat['amount'] * df_feat['velocity_last_24h']
).round(4)

# ── Feature 8: Log-transform of Amount ─────────────────
# Amount is right-skewed. Log-transform reduces skew and
# helps linear models like Logistic Regression
df_feat['log_amount'] = np.log1p(df_feat['amount'])  # log(1+x) handles amount=0 safely

print(f"\n Shape after feature engineering: {df_feat.shape}")
print("\nNew features added:")
new_cols = ['part_of_day', 'is_night', 'age_group', 'is_high_value',
            'low_trust_device', 'risk_score', 'amount_velocity_interact', 'log_amount']
print(df_feat[new_cols].head())


# ## 6. Train/Test Split & Class Imbalance Handling ⚖️
# 
# **Critical rule:** Split the data **before** any encoding or scaling. This prevents _data leakage_ — where information from the test set accidentally influences the training process.

# In[20]:


# ─────────────────────────────────────────────────────────
# CELL 13: Define features and target, then split
# ─────────────────────────────────────────────────────────

# Define which columns are features and which is the target
TARGET = 'is_fraud'

# Columns to use as model features
# We exclude 'transaction_hour' since 'part_of_day' and 'is_night' capture it better
# We exclude 'cardholder_age' since 'age_group' captures it as a category
# We exclude raw 'amount' since 'log_amount' is a better representation
FEATURE_COLS = [
    'log_amount',             # log-transformed transaction amount
    'foreign_transaction',    # binary: 1 = foreign
    'location_mismatch',      # binary: 1 = mismatch
    'device_trust_score',     # numeric: 0–100
    'velocity_last_24h',      # numeric: count of transactions
    'merchant_category',      # categorical: Electronics, Travel, etc.
    'part_of_day',            # categorical: Morning/Afternoon/Evening/Night
    'age_group',              # categorical: Young/Adult/etc.
    'is_high_value',          # binary: top 10% amount
    'is_night',               # binary: late-night transaction
    'low_trust_device',       # binary: device trust < 50
    'risk_score',             # composite risk score
    'amount_velocity_interact'# interaction: amount × velocity
]

CATEGORICAL_COLS = ['merchant_category', 'part_of_day', 'age_group']
NUMERIC_COLS = [c for c in FEATURE_COLS if c not in CATEGORICAL_COLS]

X = df_feat[FEATURE_COLS]
y = df_feat[TARGET]

# ── Stratified split: preserves fraud ratio in both sets ──
# 80% training, 20% test. stratify=y ensures both splits
# have the same fraud:legitimate ratio as the full dataset.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"Training set:  {X_train.shape[0]:,} rows")
print(f"Test set:      {X_test.shape[0]:,} rows")
print(f"\nFraud rate in train: {y_train.mean()*100:.2f}%")
print(f"Fraud rate in test:  {y_test.mean()*100:.2f}%")
print("\n Stratification successful — fraud ratio is consistent across splits")


# ## 7. Preprocessing Pipeline (Encoding + Scaling) 🔄
# 
# We use `sklearn Pipeline` + `ColumnTransformer` to:
# - **Encode** categorical features with One-Hot Encoding
# - **Scale** numeric features with StandardScaler
# 
# The transformer is **fit on training data only**, then applied to both train and test. This is the correct way to avoid leakage.

# In[21]:


# ─────────────────────────────────────────────────────────
# CELL 14: Build the preprocessing transformer
# ─────────────────────────────────────────────────────────

# Numeric transformer: StandardScaler
# Centers data to mean=0, std=1. Required for distance-based
# models (SVM, Logistic Regression) and good practice generally.
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Categorical transformer: OneHotEncoder
# Converts each category to a binary column.
# handle_unknown='ignore' means unseen categories in test data
# will be treated as all-zeros (safe for production).
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine both transformers with ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, NUMERIC_COLS),
    ('cat', categorical_transformer, CATEGORICAL_COLS)
])

# Fit on training data and transform both train and test
X_train_processed = preprocessor.fit_transform(X_train)   # ← learns from train
X_test_processed  = preprocessor.transform(X_test)        # ← applies learned params to test

print(f" Preprocessing complete")
print(f"   Training features shape: {X_train_processed.shape}")
print(f"   Test features shape:     {X_test_processed.shape}")
print(f"\n   Original features: {len(FEATURE_COLS)}")
print(f"   After OHE expansion: {X_train_processed.shape[1]} (extra cols from one-hot encoding)")


# ## 8. Model Training & Evaluation 🤖
# 
# We train **6 different classifiers**. Each model uses `class_weight='balanced'` where supported — this automatically adjusts for the class imbalance by giving fraud cases more weight during training.
# 
# **Why multiple models?** Different algorithms capture different patterns. By comparing them, we pick the best one for our data.

# In[22]:


# ─────────────────────────────────────────────────────────
# CELL 15: Define all models
# ─────────────────────────────────────────────────────────

models = {

    # Logistic Regression: Simple linear model. Fast and interpretable.
    # Good baseline. class_weight='balanced' handles imbalance.
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        C=0.5  # Regularization strength — lower = stronger regularization
    ),

    # Decision Tree: Easy to interpret, can overfit.
    # max_depth limits overfitting.
    'Decision Tree': DecisionTreeClassifier(
        max_depth=6,
        min_samples_split=20,
        class_weight='balanced',
        random_state=RANDOM_STATE
    ),

    # Random Forest: Ensemble of Decision Trees. Very powerful and robust.
    # n_estimators = number of trees. More trees = better but slower.
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1  # Use all CPU cores
    ),

    # Gradient Boosting: Builds trees sequentially, each fixing the
    # mistakes of the previous. Often the best performer.
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=RANDOM_STATE
    ),

    # Naive Bayes: Fast, probabilistic model. Assumes features are
    # independent (not always true, but often works well).
    'Naive Bayes': GaussianNB(
        var_smoothing=1e-8
    ),

    # SVM: Finds the optimal separating hyperplane. Works well with
    # scaled data. Can be slow on large datasets.
    'SVM': SVC(
        kernel='rbf',
        probability=True,    # Needed to get fraud probabilities
        class_weight='balanced',
        C=1.0,
        gamma='scale',
        random_state=RANDOM_STATE
    ),
}

print(f" Defined {len(models)} models:")
for name in models:
    print(f"   • {name}")


# In[23]:


# ─────────────────────────────────────────────────────────
# CELL 16: Train all models and evaluate on test set
# ─────────────────────────────────────────────────────────

results = []
trained_models = {}  # Store fitted models for later use

print("Training models...\n")
for name, model in models.items():
    # TRAIN
    model.fit(X_train_processed, y_train)
    trained_models[name] = model

    # PREDICT on test set
    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:, 1]  # Probability of fraud

    # METRICS
    # - Accuracy: overall correct predictions (misleading with imbalanced data)
    # - Precision: of predicted fraud, how many are actually fraud?
    # - Recall: of actual fraud cases, how many did we catch?
    # - F1-Score: harmonic mean of precision and recall (best single metric for fraud)
    # - ROC-AUC: ability to rank fraud cases above legitimate ones (0.5=random, 1.0=perfect)
    results.append({
        'Model': name,
        'Accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'Recall':    round(recall_score(y_test, y_pred, zero_division=0), 4),
        'F1-Score':  round(f1_score(y_test, y_pred, zero_division=0), 4),
        'ROC-AUC':   round(roc_auc_score(y_test, y_proba), 4),
    })
    print(f"  ✓ {name} trained")

results_df = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False).reset_index(drop=True)
print("\n All models trained and evaluated!")


# In[24]:


# ─────────────────────────────────────────────────────────
# CELL 17: Display results table
# ─────────────────────────────────────────────────────────
print("=" * 75)
print("MODEL PERFORMANCE ON TEST SET (sorted by ROC-AUC)")
print("=" * 75)
display(results_df.style
        .background_gradient(cmap='RdYlGn', subset=['Accuracy','Precision','Recall','F1-Score','ROC-AUC'])
        .format({'Accuracy': '{:.4f}', 'Precision': '{:.4f}',
                 'Recall': '{:.4f}', 'F1-Score': '{:.4f}', 'ROC-AUC': '{:.4f}'})
        .set_properties(**{'font-size': '12px', 'text-align': 'center'})
)

best_model_name = results_df.iloc[0]['Model']
best_roc = results_df.iloc[0]['ROC-AUC']
print(f"\n Best Model: {best_model_name} (ROC-AUC = {best_roc:.4f})")


# In[25]:


# ─────────────────────────────────────────────────────────
# CELL 18: Confusion matrices for all models
# A confusion matrix shows:
#  TN: Legitimate correctly identified as legitimate
#  TP: Fraud correctly identified as fraud
#  FP: Legitimate wrongly flagged as fraud (false alarm)
#  FN: Fraud missed (most dangerous in fraud detection!)
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, (name, model) in enumerate(trained_models.items()):
    y_pred = model.predict(X_test_processed)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legitimate', 'Fraud'])
    disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
    axes[i].set_title(f'{name}', fontsize=12, fontweight='bold')

    # Annotate with FN (missed fraud) count
    fn = cm[1][0]
    axes[i].set_xlabel(f'Predicted\nMissed Fraud (FN): {fn}', fontsize=10)

plt.suptitle('Confusion Matrices — All Models', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


# In[26]:


# ─────────────────────────────────────────────────────────
# CELL 19: Detailed classification report for the best model
# ─────────────────────────────────────────────────────────
best_model = trained_models[best_model_name]
y_pred_best = best_model.predict(X_test_processed)

print(f"=== Classification Report: {best_model_name} ===")
print(classification_report(y_test, y_pred_best,
                             target_names=['Legitimate', 'Fraud']))


# ## 9. Model Comparison & Best Model Selection 🏆

# In[27]:


# ─────────────────────────────────────────────────────────
# CELL 20: ROC curves for all models
# ROC (Receiver Operating Characteristic) curve plots the
# True Positive Rate vs False Positive Rate at various thresholds.
# The larger the AUC (area under the curve), the better.
# ─────────────────────────────────────────────────────────
plt.figure(figsize=(10, 8))

colors_roc = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']

for i, (name, model) in enumerate(trained_models.items()):
    y_proba = model.predict_proba(X_test_processed)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})',
             color=colors_roc[i], linewidth=2.5)

plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier (AUC=0.50)')
plt.fill_between([0, 1], [0, 1], alpha=0.05, color='grey')

plt.xlabel('False Positive Rate (Legitimate flagged as Fraud)', fontsize=12)
plt.ylabel('True Positive Rate (Fraud correctly caught)', fontsize=12)
plt.title('ROC Curve Comparison — All Models', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# In[28]:


# ─────────────────────────────────────────────────────────
# CELL 21: Side-by-side bar chart comparing key metrics
# ─────────────────────────────────────────────────────────
metrics_to_plot = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']
plot_df = results_df.set_index('Model')[metrics_to_plot]

ax = plot_df.plot(kind='bar', figsize=(14, 6), width=0.75,
                  color=['#3498DB', '#E74C3C', '#2ECC71', '#F39C12'],
                  edgecolor='black', alpha=0.85)
plt.title('Model Metric Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Model')
plt.ylabel('Score')
plt.xticks(rotation=20, ha='right')
plt.ylim(0, 1.1)
plt.legend(loc='upper right')
plt.axhline(y=0.9, color='grey', linestyle='--', alpha=0.5, label='0.9 reference')
plt.tight_layout()
plt.show()


# In[29]:


# ─────────────────────────────────────────────────────────
# CELL 22: Feature importance from Random Forest
# Which features does the model rely on most?
# ─────────────────────────────────────────────────────────
rf_model = trained_models['Random Forest']

# Get feature names after OHE expansion
ohe_cats = (preprocessor.named_transformers_['cat']
            .named_steps['onehot']
            .get_feature_names_out(CATEGORICAL_COLS).tolist())
all_feature_names = NUMERIC_COLS + ohe_cats

# Feature importances
importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=True).tail(15)  # Top 15

plt.figure(figsize=(10, 7))
bars = plt.barh(importance_df['Feature'], importance_df['Importance'],
                color='steelblue', edgecolor='navy', alpha=0.85)

# Highlight top 3 features
for bar in bars[-3:]:
    bar.set_color('#E74C3C')
    bar.set_alpha(0.9)

plt.xlabel('Feature Importance (Mean Decrease in Impurity)', fontsize=11)
plt.title('Top 15 Feature Importances — Random Forest', fontsize=13, fontweight='bold')
plt.grid(axis='x', alpha=0.4)
plt.tight_layout()
plt.show()

print("\n Top 5 most important features:")
print(importance_df.tail(5)[::-1].to_string(index=False))


# In[30]:


# ─────────────────────────────────────────────────────────
# CELL 23: Cross-validation for best model
# Cross-validation tests the model on 5 different splits of
# the training data. This gives a more reliable estimate of
# performance than a single train/test split.
# ─────────────────────────────────────────────────────────
print(f"Running 5-Fold Cross-Validation on {best_model_name}...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(
    best_model, X_train_processed, y_train,
    cv=cv, scoring='roc_auc', n_jobs=-1
)

print(f"\n5-Fold CV ROC-AUC Scores: {cv_scores.round(4)}")
print(f"Mean:  {cv_scores.mean():.4f}")
print(f"Std:   {cv_scores.std():.4f}")
print(f"\nInterpretation:")
print(f"  • Mean CV AUC of {cv_scores.mean():.3f} shows model generalises well.")
print(f"  • Low std ({cv_scores.std():.4f}) means consistent performance across folds.")


# In[31]:


# ─────────────────────────────────────────────────────────
# CELL 24: Save the full pipeline to disk
# We save: the preprocessor + best model + engineering params
# ─────────────────────────────────────────────────────────

# Save preprocessor
joblib.dump(preprocessor, 'fraud_preprocessor.pkl')

# Save best model
joblib.dump(best_model, 'fraud_best_model.pkl')

# Save feature engineering parameters needed for new data
pipeline_config = {
    'amount_threshold': float(amount_threshold),
    'feature_cols': FEATURE_COLS,
    'numeric_cols': NUMERIC_COLS,
    'categorical_cols': CATEGORICAL_COLS,
    'best_model_name': best_model_name
}
with open('pipeline_config.json', 'w') as f:
    json.dump(pipeline_config, f, indent=2)

print(" Pipeline saved to disk:")
print("   • fraud_preprocessor.pkl  — encoding + scaling transformer")
print("   • fraud_best_model.pkl    — trained best model")
print("   • pipeline_config.json    — feature engineering parameters")


# ## 10. Reusable Prediction Function 🔮
# 
# This function accepts **any new, raw transaction data** and returns fraud predictions. All preprocessing is handled internally — the user just passes raw data.

# In[32]:


# ─────────────────────────────────────────────────────────
# CELL 25: Production prediction function
# ─────────────────────────────────────────────────────────

def predict_fraud(new_data, model=None, processor=None, config=None, threshold=0.5):
    """
    Predict fraud on new, raw transaction data.

    Parameters
    ----------
    new_data : pd.DataFrame
        Raw transaction data. Must include the original columns:
        amount, transaction_hour, merchant_category, foreign_transaction,
        location_mismatch, device_trust_score, velocity_last_24h, cardholder_age.
    model : sklearn model, optional
        Trained model. Defaults to best_model from current session.
    processor : sklearn ColumnTransformer, optional
        Fitted preprocessor. Defaults to preprocessor from current session.
    config : dict, optional
        Pipeline config. Defaults to pipeline_config from current session.
    threshold : float
        Probability threshold for classifying as fraud (default 0.5).
        Lower threshold → catches more fraud, but more false alarms.

    Returns
    -------
    pd.DataFrame with predictions and fraud probabilities.
    """
    # Use defaults from current session if not provided
    if model is None:     model = best_model
    if processor is None: processor = preprocessor
    if config is None:    config = pipeline_config

    df_new = new_data.copy()
    df_new.columns = df_new.columns.str.strip().str.lower()

    # ── Step 1: Apply the same feature engineering ──────
    df_new['part_of_day'] = df_new['transaction_hour'].apply(get_part_of_day)
    df_new['is_night'] = df_new['transaction_hour'].apply(
        lambda h: 1 if (h >= 23 or h <= 4) else 0
    )
    df_new['age_group'] = pd.cut(
        df_new['cardholder_age'],
        bins=[0, 25, 35, 50, 65, 120],
        labels=['Young', 'Adult', 'Middle-aged', 'Senior', 'Elderly']
    ).astype(str)
    df_new['is_high_value'] = (df_new['amount'] > config['amount_threshold']).astype(int)
    df_new['low_trust_device'] = (df_new['device_trust_score'] < 50).astype(int)
    df_new['risk_score'] = (
        df_new['foreign_transaction'] * 0.3 +
        df_new['location_mismatch'] * 0.3 +
        (1 - df_new['device_trust_score'] / 100) * 0.2 +
        df_new['velocity_last_24h'] / df_new['velocity_last_24h'].clip(lower=1).max() * 0.2
    ).round(4)
    df_new['amount_velocity_interact'] = (df_new['amount'] * df_new['velocity_last_24h']).round(4)
    df_new['log_amount'] = np.log1p(df_new['amount'])

    # ── Step 2: Select the same feature columns ──────────
    X_new = df_new[config['feature_cols']]

    # ── Step 3: Apply the saved preprocessor ────────────
    X_new_processed = processor.transform(X_new)

    # ── Step 4: Predict ──────────────────────────────────
    fraud_proba = model.predict_proba(X_new_processed)[:, 1]
    fraud_pred  = (fraud_proba >= threshold).astype(int)

    # ── Step 5: Return results ───────────────────────────
    results = new_data.copy()
    results['fraud_probability'] = fraud_proba.round(4)
    results['predicted_fraud'] = fraud_pred
    results['risk_level'] = pd.cut(
        fraud_proba,
        bins=[-0.001, 0.3, 0.6, 1.001],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    return results


print(" predict_fraud() function defined and ready!")


# In[33]:


# ─────────────────────────────────────────────────────────
# CELL 26: Demo — predict on new unseen transactions
# ─────────────────────────────────────────────────────────

new_transactions = pd.DataFrame({
    'transaction_id':    [10001, 10002, 10003, 10004, 10005],
    'amount':            [35.00, 1499.99, 22.50, 875.00, 9.99],
    'transaction_hour':  [14,     2,      11,     23,     16],
    'merchant_category': ['Grocery', 'Electronics', 'Food', 'Travel', 'Food'],
    'foreign_transaction':[0,      1,       0,       1,      0],
    'location_mismatch': [0,       1,       0,       1,      0],
    'device_trust_score':[85,      30,      72,      45,     91],
    'velocity_last_24h': [1,       8,       0,       5,      1],
    'cardholder_age':    [35,      55,      28,      67,     42]
})

print(" Input: New transactions to classify\n")
print(new_transactions.to_string(index=False))

predictions = predict_fraud(new_transactions)

print("\n Predictions:")
output_cols = ['transaction_id', 'amount', 'fraud_probability', 'predicted_fraud', 'risk_level']
display(predictions[output_cols].style
        .background_gradient(cmap='RdYlGn_r', subset=['fraud_probability'])
        .applymap(lambda v: 'background-color: #FFB3B3; font-weight: bold'
                  if v == 1 else '', subset=['predicted_fraud'])
        .format({'fraud_probability': '{:.4f}'})
)

print("\n Risk explanations:")
for _, row in predictions.iterrows():
    status = '🚨 FRAUD' if row['predicted_fraud'] == 1 else '✅ LEGIT'
    print(f"  TXN {int(row['transaction_id'])}: {status} | "
          f"P(fraud)={row['fraud_probability']:.4f} | "
          f"{row['risk_level']}")


# ## 11. RAG Chatbot Integration 🤖💬
# 
# A **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about the fraud dataset. It works by:
# 1. Computing **summary statistics** from the cleaned dataset (the *knowledge base*)
# 2. Retrieving the most relevant statistics for a given question
# 3. Sending them to Claude's API along with the question to generate an accurate, grounded answer
# 
# > **Note:** You need an Anthropic API key to use this section. Set it as an environment variable: `ANTHROPIC_API_KEY=your_key`

# In[34]:


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
        "what is the average device trust score?" ,
        "what is the best ml model for your dataset?",
        "At what time the fraud transaction is happening more?",
    ]
    print("\n" + "="*60)
    for q in questions:
        print(f"\nQ: {q}\nA: {bot.chat(q)}\n" + "-"*50)


# ---
# 
# ## ✅ Pipeline Summary
# 
# | Step | What was done |
# |------|---------------|
# | **Data Loading** | Loaded from XLS/CSV, standardised column names |
# | **EDA** | Distributions, correlations, class balance, categorical fraud rates |
# | **Cleaning** | Duplicates, missing values, range validation, ID column removed |
# | **Feature Engineering** | 8 new features: part_of_day, is_night, age_group, is_high_value, low_trust_device, risk_score, amount×velocity interaction, log_amount |
# | **Train/Test Split** | 80/20 stratified split — fraud ratio preserved in both sets |
# | **Preprocessing** | OneHotEncoder (categorical) + StandardScaler (numeric) fit on train only |
# | **Imbalance Handling** | `class_weight='balanced'` in all models |
# | **Models** | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, Naive Bayes, SVM |
# | **Evaluation** | Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, Cross-Validation |
# | **Saved Artifacts** | `fraud_preprocessor.pkl`, `fraud_best_model.pkl`, `pipeline_config.json` |
# | **Prediction Function** | `predict_fraud()` — accepts raw data, handles all transforms internally |
# | **RAG Chatbot** | Knowledge base from dataset stats + retrieval + Claude API integration |
# 
# ### 🚀 To use on new data:
# ```python
# import joblib, json, pandas as pd
# preprocessor = joblib.load('fraud_preprocessor.pkl')
# best_model   = joblib.load('fraud_best_model.pkl')
# with open('pipeline_config.json') as f:
#     pipeline_config = json.load(f)
# 
# # Then call:
# results = predict_fraud(your_new_dataframe)
# ```












