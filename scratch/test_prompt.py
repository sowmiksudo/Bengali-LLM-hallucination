import json
import pandas as pd
from ollama import Client

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

train_cb = train_df[~train_df["has_context"]].copy().reset_index(drop=True)
# Let's pick sample 2, 5, 9, 10 (which failed or were interesting)
samples_to_test = [1, 4, 8, 9] # corresponding to 0-indexed indices:
# index 1: ‘কাঁঠালপাড়া’য় (True Label: 0)
# index 4: ইংরেজি ভাষায় কম্পিটেন্ট (True Label: 1)
# index 8: আয়তনে পৃথিবীর সবচেয়ে ছোট দেশ (True Label: 0)
# index 9: ডক্টর ইউনূসের নেতৃত্বে (True Label: 0)

client = Client(host='http://127.0.0.1:11434')
MODEL_NAME = "qwen2.5-coder:7b"

with open("scratch/test_prompt_output.txt", "w", encoding="utf-8") as f:
    f.write("Testing Chain-of-Thought Prompt with Qwen:\n")
    for idx in samples_to_test:
        row = train_cb.iloc[idx]
        user_content = (
            f"প্রশ্ন: {row.prompt_bn}\n"
            f"উত্তর: {row.response_bn}\n\n"
            "উপরের উত্তরটি কি সঠিক? অনুগ্রহ করে সংক্ষেপে আপনার যুক্তি দিন। "
            "যুক্তির শেষে একটি নতুন লাইনে শুধুমাত্র 'হ্যাঁ' (যদি সঠিক হয়) অথবা 'না' (যদি ভুল হয়) লিখুন।"
        )
        
        try:
            res = client.chat(model=MODEL_NAME, messages=[
                {'role': 'user', 'content': user_content},
            ], options={'temperature': 0.0})
            output = res['message']['content'].strip()
        except Exception as e:
            output = f"Error: {e}"
            
        lines = [line.strip() for line in output.split('\n') if line.strip()]
        last_line = lines[-1].replace("**", "").replace(".", "").strip() if lines else ""
        parsed = 1 if ("হ্যাঁ" in last_line or "সঠিক" in last_line) else 0
        
        f.write(f"\n--- Index {idx} ---\n")
        f.write(f"Prompt: {row.prompt_bn}\n")
        f.write(f"Response: {row.response_bn}\n")
        f.write(f"True Label: {row.label}\n")
        f.write(f"Output:\n{output}\n")
        f.write(f"Parsed: {parsed} (Matches Ground Truth: {parsed == row.label})\n")
