import json
import pandas as pd
import numpy as np
import ollama
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

# ==========================================
# CONFIGURATION
# ==========================================
OLLAMA_MODEL = "qwen2.5-coder:7b"  # Change to "deepseek-r1:7b" or others
OLLAMA_HOST = "http://127.0.0.1:11434"
USE_OLLAMA_FOR_VAL = True  # Set to False to use TF-IDF fallback for closed-book

client = ollama.Client(host=OLLAMA_HOST)

# ==========================================
# 1. LOAD AND PREPROCESS DATA
# ==========================================
print("Loading data...")
with open("dataset samples.json", encoding="utf-8") as f:
    train_records = json.load(f)
df = pd.DataFrame(train_records)

NO_CONTEXT_VALUES = {"", "nan", "NaN", "[NULL]", None}
def clean_context(value):
    if pd.isna(value) or str(value).strip() in NO_CONTEXT_VALUES:
        return ""
    return str(value).strip()

df["context"] = df["context"].apply(clean_context)
df["has_context"] = df["context"].str.len() > 0
df["prompt_bn"] = df["prompt_bn"].astype(str)
df["response_bn"] = df["response_bn"].astype(str)

# ==========================================
# 2. OPEN-BOOK PIPELINE (Lexical Overlap)
# ==========================================
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
    import difflib
    seq = difflib.SequenceMatcher(None, c, r)
    return seq.ratio()

def extract_numbers(text):
    import re
    return set(re.findall(r'[0-9০-৯]+', text))

def get_number_mismatch(c, r):
    nums_c = extract_numbers(c)
    nums_r = extract_numbers(r)
    if not nums_r:
        return 0.0
    return 1.0 if not nums_r.issubset(nums_c) else 0.0

def extract_lexical_features(data_df):
    features = []
    for row in data_df.itertuples():
        c, r = row.context, row.response_bn
        features.append({
            "jaccard": get_jaccard_similarity(c, r),
            "char_overlap": get_char_overlap(c, r),
            "exact_match": get_exact_match(c, r),
            "lcs_ratio": get_lcs_ratio(c, r),
            "num_mismatch": get_number_mismatch(c, r)
        })
    return pd.DataFrame(features)

# Run CV for Open-Book
df_ob = df[df["has_context"]].copy().reset_index(drop=True)
X_ob = extract_lexical_features(df_ob)
y_ob = df_ob["label"].values

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
ob_preds = np.zeros(len(df_ob))
for train_idx, val_idx in cv.split(X_ob, y_ob):
    X_tr, y_tr = X_ob.iloc[train_idx], y_ob[train_idx]
    X_val, y_val = X_ob.iloc[val_idx], y_ob[val_idx]
    clf = LogisticRegression(class_weight="balanced", random_state=42)
    clf.fit(X_tr, y_tr)
    ob_preds[val_idx] = clf.predict(X_val)

# ==========================================
# 3. CLOSED-BOOK PIPELINE (Ollama / LLM)
# ==========================================
df_cb = df[~df["has_context"]].copy().reset_index(drop=True)
y_cb = df_cb["label"].values

def check_fact_ollama(prompt, response):
    # Chain-of-thought prompt format
    user_content = (
        f"প্রশ্ন: {prompt}\n"
        f"উত্তর: {response}\n\n"
        "উপরের উত্তরটি কি সঠিক? অনুগ্রহ করে সংক্ষেপে আপনার যুক্তি দিন। "
        "যুক্তির শেষে একটি নতুন লাইনে শুধুমাত্র 'হ্যাঁ' (যদি সঠিক হয়) অথবা 'না' (যদি ভুল হয়) লিখুন।"
    )
    
    try:
        res = client.chat(model=OLLAMA_MODEL, messages=[
            {'role': 'user', 'content': user_content},
        ], options={'temperature': 0.0})
        
        output = res['message']['content'].strip()
        
        # Parse the last line of output
        lines = [line.strip() for line in output.split('\n') if line.strip()]
        if not lines:
            return 0
        last_line = lines[-1].replace("**", "").replace(".", "").strip()
        
        if "হ্যাঁ" in last_line or "সঠিক" in last_line:
            return 1
        else:
            return 0
    except Exception as e:
        # Fallback
        return 0

if USE_OLLAMA_FOR_VAL:
    print(f"\nFact-checking {len(df_cb)} closed-book samples using Ollama ({OLLAMA_MODEL})...")
    cb_preds = []
    for row in tqdm(df_cb.itertuples(), total=len(df_cb)):
        cb_preds.append(check_fact_ollama(row.prompt_bn, row.response_bn))
    cb_preds = np.array(cb_preds)
else:
    print("\nUSE_OLLAMA_FOR_VAL is False. Running local TF-IDF closed-book baseline...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=5000, analyzer='char_wb', ngram_range=(2,4))
    X_cb = vectorizer.fit_transform(df_cb["prompt_bn"] + " " + df_cb["response_bn"])
    cb_preds = np.zeros(len(df_cb))
    for train_idx, val_idx in cv.split(X_cb, y_cb):
        X_tr, y_tr = X_cb[train_idx], y_cb[train_idx]
        X_val, y_val = X_cb[val_idx], y_cb[val_idx]
        clf = LogisticRegression(class_weight="balanced", random_state=42)
        clf.fit(X_tr, y_tr)
        cb_preds[val_idx] = clf.predict(X_val)

# ==========================================
# 4. OVERALL SCORE EVALUATION
# ==========================================
# Combine predictions
# Reconstruct overall target and prediction vectors in correct alignment
df_ob["pred"] = ob_preds
df_cb["pred"] = cb_preds

df_combined = pd.concat([df_ob, df_cb], axis=0)

y_true_all = df_combined["label"].values
y_pred_all = df_combined["pred"].values

print("\n" + "="*45)
print("             LOCAL VALIDATION REPORT")
print("="*45)
print(f"Open-Book Samples  : {len(df_ob)}")
print(f"Closed-Book Samples: {len(df_cb)}")
print(f"Total Samples      : {len(df_combined)}")
print("-"*45)

print("\n--- Open-Book Classification Report ---")
print(classification_report(df_ob["label"].values, df_ob["pred"].values, target_names=["Hallucinated", "Faithful"]))
print(f"Open-Book Macro-F1: {f1_score(df_ob['label'].values, df_ob['pred'].values, average='macro'):.4f}\n")

print("--- Closed-Book Classification Report ---")
print(classification_report(df_cb["label"].values, df_cb["pred"].values, target_names=["Hallucinated", "Faithful"]))
print(f"Closed-Book Macro-F1: {f1_score(df_cb['label'].values, df_cb['pred'].values, average='macro'):.4f}\n")

print("--- Overall Combined Classification Report ---")
print(classification_report(y_true_all, y_pred_all, target_names=["Hallucinated", "Faithful"]))
print(f"OVERALL COMBINED MACRO-F1: {f1_score(y_true_all, y_pred_all, average='macro'):.4f}")
print("="*45)
