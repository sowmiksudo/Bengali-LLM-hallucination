import pandas as pd
import numpy as np
import json
import re
import difflib
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# Load data
with open("dataset samples.json", encoding="utf-8") as f:
    train_records = json.load(f)
train_df = pd.DataFrame(train_records)

NO_CONTEXT_VALUES = {"", "nan", "NaN", "[NULL]", None}
def clean_context(value):
    if pd.isna(value) or str(value).strip() in NO_CONTEXT_VALUES:
        return ""
    return str(value).strip()

train_df["context"] = train_df["context"].apply(clean_context)
train_df["has_context"] = train_df["context"].str.len() > 0
train_df["prompt_bn"] = train_df["prompt_bn"].astype(str)
train_df["response_bn"] = train_df["response_bn"].astype(str)

# Features
def get_jaccard_similarity(c, r):
    words_c = set(c.split())
    words_r = set(r.split())
    if not words_c or not words_r:
        return 0.0
    return len(words_c.intersection(words_r)) / len(words_c.union(words_r))

def get_char_overlap(c, r):
    chars_c = set(c)
    chars_r = set(r)
    if not chars_r:
        return 0.0
    return len(chars_c.intersection(chars_r)) / len(chars_r)

def get_exact_match(c, r):
    return 1.0 if r in c else 0.0

def get_lcs_ratio(c, r):
    seq = difflib.SequenceMatcher(None, c, r)
    return seq.ratio()

def extract_numbers(text):
    return set(re.findall(r'[0-9০-৯]+', text))

def get_number_mismatch(c, r):
    nums_c = extract_numbers(c)
    nums_r = extract_numbers(r)
    if not nums_r:
        return 0.0
    return 1.0 if not nums_r.issubset(nums_c) else 0.0

def extract_lexical_features(df):
    features = []
    for row in df.itertuples():
        c, r = row.context, row.response_bn
        features.append({
            "jaccard": get_jaccard_similarity(c, r),
            "char_overlap": get_char_overlap(c, r),
            "exact_match": get_exact_match(c, r),
            "lcs_ratio": get_lcs_ratio(c, r),
            "num_mismatch": get_number_mismatch(c, r)
        })
    return pd.DataFrame(features)

train_ob = train_df[train_df["has_context"]].copy().reset_index(drop=True)
X_train_ob = extract_lexical_features(train_ob)
y_train_ob = train_ob["label"].values

# Stratified K-Fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_probas = np.zeros(len(train_ob))

for train_idx, val_idx in cv.split(X_train_ob, y_train_ob):
    X_tr, y_tr = X_train_ob.iloc[train_idx], y_train_ob[train_idx]
    X_val, y_val = X_train_ob.iloc[val_idx], y_train_ob[val_idx]
    
    clf = LogisticRegression(class_weight="balanced", random_state=42)
    clf.fit(X_tr, y_tr)
    cv_probas[val_idx] = clf.predict_proba(X_val)[:, 1]

# Grid search threshold
best_thresh = 0.5
best_f1 = 0.0

for thresh in np.linspace(0.1, 0.9, 81):
    preds = (cv_probas >= thresh).astype(int)
    score = f1_score(y_train_ob, preds, average="macro")
    if score > best_f1:
        best_f1 = score
        best_thresh = thresh

print(f"Best Open-Book Threshold: {best_thresh:.4f}")
print(f"Best Open-Book Macro-F1: {best_f1:.4f}")
