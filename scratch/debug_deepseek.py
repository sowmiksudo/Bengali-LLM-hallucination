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
train_cb_sample = train_cb.head(10)

MODEL_NAME = "deepseek-r1:7b"
client = Client(host='http://127.0.0.1:11434')

with open("scratch/debug_deepseek.txt", "w", encoding="utf-8") as f:
    f.write("Debugging raw DeepSeek-R1 responses:\n")
    for idx, row in enumerate(train_cb_sample.itertuples()):
        system_prompt = "আপনি একটি প্রশ্নের উত্তর সঠিক নাকি ভুল তা যাচাই করছেন। অতিরিক্ত কোনো কথা না বলে শুধুমাত্র একটি শব্দে উত্তর দিন: 'হ্যাঁ' (যদি সঠিক হয়) অথবা 'না' (যদি ভুল হয়)।"
        user_content = f"প্রশ্ন: {row.prompt_bn}\nউত্তর: {row.response_bn}\nউপরের উত্তরটি কি সঠিক? হ্যাঁ অথবা না বলুন।"
        
        try:
            res = client.chat(model=MODEL_NAME, messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_content},
            ], options={'temperature': 0.0})
            
            output = res['message']['content'].strip()
        except Exception as e:
            output = f"Error: {e}"
        
        # Parse final answer by extracting text after </think> if present
        answer_part = output
        if "</think>" in output:
            answer_part = output.split("</think>")[-1].strip()
            
        parsed = 1 if ("হ্যাঁ" in answer_part or "সঠিক" in answer_part) else 0
        
        f.write(f"\n--- Sample {idx+1} ---\n")
        f.write(f"Prompt: {row.prompt_bn}\n")
        f.write(f"Response: {row.response_bn}\n")
        f.write(f"True Label: {row.label}\n")
        f.write(f"Raw Output:\n{output}\n")
        f.write(f"Parsed: {parsed}\n")
