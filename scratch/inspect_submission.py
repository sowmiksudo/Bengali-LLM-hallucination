import pandas as pd

sub_df = pd.read_csv("submission_v1.csv")
test_df = pd.read_csv("test set.csv")

print("Submission shape:", sub_df.shape)
print("Test shape:", test_df.shape)

# Normalize context values
NO_CONTEXT_VALUES = {"", "nan", "NaN", "[NULL]"}
def clean_context(value):
    if pd.isna(value) or str(value).strip() in NO_CONTEXT_VALUES:
        return ""
    return str(value).strip()

test_df["context"] = test_df["context"].apply(clean_context)
test_df["has_context"] = test_df["context"].str.len() > 0

# Merge submission and test
merged = pd.merge(test_df, sub_df, on="id")

# Analyze prediction distribution for open-book vs closed-book
print("\nOverall Label Distribution:")
print(merged["label"].value_counts(normalize=True))

print("\nOpen-Book Label Distribution (has context):")
print(merged[merged["has_context"]]["label"].value_counts(normalize=True))

print("\nClosed-Book Label Distribution (no context):")
print(merged[~merged["has_context"]]["label"].value_counts(normalize=True))
