import pandas as pd

test_df = pd.read_csv("test set.csv")

# Normalize context values
NO_CONTEXT_VALUES = {"", "nan", "NaN", "[NULL]"}
def clean_context(value):
    if pd.isna(value) or str(value).strip() in NO_CONTEXT_VALUES:
        return ""
    return str(value).strip()

test_df["context"] = test_df["context"].apply(clean_context)
test_df["has_context"] = test_df["context"].str.len() > 0

with open("scratch/analysis_output.txt", "w", encoding="utf-8") as f:
    f.write(f"Test shape: {test_df.shape}\n")
    f.write(f"Columns: {test_df.columns.tolist()}\n\n")
    
    f.write("Context stats:\n")
    f.write(f"{test_df['has_context'].value_counts().to_string()}\n\n")
    
    f.write("Null counts:\n")
    f.write(f"{test_df.isnull().sum().to_string()}\n\n")
    
    f.write("Sample rows with context:\n")
    f.write(f"{test_df[test_df['has_context']].head(5).to_string()}\n\n")
    
    f.write("Sample rows without context:\n")
    f.write(f"{test_df[~test_df['has_context']].head(5).to_string()}\n\n")
