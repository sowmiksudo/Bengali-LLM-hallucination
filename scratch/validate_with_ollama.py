import json
import pandas as pd
from ollama import Client
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score

# Load training data
with open("dataset samples.json", encoding="utf-8") as f:
    train_records = json.load(f)
train_df = pd.DataFrame(train_records)

# Clean context
NO_CONTEXT_VALUES = {"", "nan", "NaN", "[NULL]", None}
def clean_context(value):
    if pd.isna(value) or str(value).strip() in NO_CONTEXT_VALUES:
        return ""
    return str(value).strip()

train_df["context"] = train_df["context"].apply(clean_context)
train_df["has_context"] = train_df["context"].str.len() > 0

# Get closed-book rows
train_cb = train_df[~train_df["has_context"]].copy().reset_index(drop=True)
print(f"Loaded {len(train_cb)} closed-book training samples.")

MODEL_NAME = "qwen2.5-coder:7b"
client = Client(host='http://127.0.0.1:11434')

def check_fact_ollama(prompt, response):
    system_prompt = "আপনি একটি প্রশ্নের উত্তর সঠিক নাকি ভুল তা যাচাই করছেন। অতিরিক্ত কোনো কথা না বলে শুধুমাত্র একটি শব্দে উত্তর দিন: 'হ্যাঁ' (যদি সঠিক হয়) অথবা 'না' (যদি ভুল হয়)।"
    user_content = f"প্রশ্ন: {prompt}\nউত্তর: {response}\nউপরের উত্তরটি কি সঠিক? হ্যাঁ অথবা না বলুন।"
    
    try:
        res = client.chat(model=MODEL_NAME, messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_content},
        ], options={
            'temperature': 0.0  # Greedy decoding for deterministic results
        })
        output = res['message']['content'].strip()
        # Parse output
        if "হ্যাঁ" in output or "সঠিক" in output:
            return 1
        else:
            return 0
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return 0

# Run on the first 20 samples to verify it runs successfully and quickly
train_cb_sample = train_cb.head(20).copy()
preds = []
true_labels = []

print(f"\nFact-checking closed-book samples using Ollama ({MODEL_NAME})...")
for row in tqdm(train_cb_sample.itertuples(), total=len(train_cb_sample)):
    pred = check_fact_ollama(row.prompt_bn, row.response_bn)
    preds.append(pred)
    true_labels.append(row.label)

# Report
print("\n--- Ollama Fact-Checking Validation Report (20 Samples) ---")
print(classification_report(true_labels, preds, target_names=["Hallucinated (0)", "Faithful (1)"]))
print(f"Macro-F1 Score: {f1_score(true_labels, preds, average='macro'):.4f}")
