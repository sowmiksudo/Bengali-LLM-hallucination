import os
import json
import re
import difflib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from tqdm.auto import tqdm

IS_KAGGLE = os.path.exists("/kaggle/input")
print(f"Running on Kaggle: {IS_KAGGLE}")

TRAIN_PATH = "dataset samples.json"
TEST_PATH = "test set.csv"

# Load training data
with open(TRAIN_PATH, encoding="utf-8") as f:
    train_records = json.load(f)
train_df = pd.DataFrame(train_records)

# Load test data
test_df = pd.read_csv(TEST_PATH)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Clean context helper
NO_CONTEXT_VALUES = {"", "nan", "NaN", "[NULL]", None}

def clean_context(value):
    if pd.isna(value) or str(value).strip() in NO_CONTEXT_VALUES:
        return ""
    return str(value).strip()

for df in [train_df, test_df]:
    df["context"] = df["context"].apply(clean_context)
    df["has_context"] = df["context"].str.len() > 0
    df["prompt_bn"] = df["prompt_bn"].astype(str)
    df["response_bn"] = df["response_bn"].astype(str)

# Lexical features
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

# Since we don't have NLI installed or GPU locally, we use a dummy NLI extractor for local test
def extract_nli_features(df):
    # Local fallback
    return pd.DataFrame({
        "nli_entailment": [0.5] * len(df),
        "nli_contradiction": [0.5] * len(df)
    })

# Open-book train
train_ob = train_df[train_df["has_context"]].copy().reset_index(drop=True)
lex_feats_train = extract_lexical_features(train_ob)
nli_feats_train = extract_nli_features(train_ob)
X_train_ob = pd.concat([lex_feats_train, nli_feats_train], axis=1)
y_train_ob = train_ob["label"].values

print("\n--- Open-Book Local Cross Validation (Lexical Features Only) ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_preds = np.zeros(len(train_ob))
for train_idx, val_idx in cv.split(X_train_ob, y_train_ob):
    X_tr, y_tr = X_train_ob.iloc[train_idx], y_train_ob[train_idx]
    X_val, y_val = X_train_ob.iloc[val_idx], y_train_ob[val_idx]
    clf = LogisticRegression(class_weight="balanced", random_state=42)
    clf.fit(X_tr, y_tr)
    cv_preds[val_idx] = clf.predict(X_val)

print(classification_report(y_train_ob, cv_preds, target_names=["Hallucinated", "Faithful"]))

ob_classifier = LogisticRegression(class_weight="balanced", random_state=42)
ob_classifier.fit(X_train_ob, y_train_ob)

# Closed-book fallback classifier (TF-IDF)
train_cb = train_df[~train_df["has_context"]].copy().reset_index(drop=True)
print(f"Closed-book training samples: {len(train_cb)}")

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000, analyzer='char_wb', ngram_range=(2,4))
X_train_cb = vectorizer.fit_transform(train_cb["prompt_bn"] + " " + train_cb["response_bn"])
y_train_cb = train_cb["label"].values

cb_clf = LogisticRegression(class_weight="balanced", random_state=42)
cb_clf.fit(X_train_cb, y_train_cb)
cb_preds = cb_clf.predict(X_train_cb)
print("\n--- Closed-Book Local Fit (TF-IDF Baseline) ---")
print(classification_report(y_train_cb, cb_preds, target_names=["Hallucinated", "Faithful"]))

# Split test
test_ob = test_df[test_df["has_context"]].copy().reset_index(drop=True)
test_cb = test_df[~test_df["has_context"]].copy().reset_index(drop=True)

# Predict open-book test
lex_feats_test = extract_lexical_features(test_ob)
nli_feats_test = extract_nli_features(test_ob)
X_test_ob = pd.concat([lex_feats_test, nli_feats_test], axis=1)
test_ob["label"] = ob_classifier.predict(X_test_ob)

# Predict closed-book test
X_test_cb = vectorizer.transform(test_cb["prompt_bn"] + " " + test_cb["response_bn"])
test_cb["label"] = cb_clf.predict(X_test_cb)

# Combine
submission_ob = test_ob[["id", "label"]]
submission_cb = test_cb[["id", "label"]]
submission_all = pd.concat([submission_ob, submission_cb], axis=0).sort_values("id").reset_index(drop=True)

submission_all["id"] = submission_all["id"].astype(int)
submission_all["label"] = submission_all["label"].astype(int)

print("\n--- Test Set Submission Preview ---")
print(submission_all["label"].value_counts())
submission_all.to_csv("scratch/test_submission.csv", index=False)
print("Saved temporary submission to scratch/test_submission.csv successfully!")
