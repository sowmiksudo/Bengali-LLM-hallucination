import pandas as pd

sub1 = pd.read_csv("submission_v1.csv")
sub2 = pd.read_csv("scratch/test_submission.csv")

print("sub1 (user) label counts:")
print(sub1["label"].value_counts())

print("\nsub2 (local lexical) label counts:")
print(sub2["label"].value_counts())

# Merge and calculate overlap
merged = pd.merge(sub1, sub2, on="id", suffixes=("_user", "_local"))
overlap = (merged["label_user"] == merged["label_local"]).mean()
print(f"\nOverlap ratio between user and local predictions: {overlap:.4f}")

# Open-book vs Closed-book overlap
test_df = pd.read_csv("test set.csv")
NO_CONTEXT_VALUES = {"", "nan", "NaN", "[NULL]"}
def clean_context(value):
    if pd.isna(value) or str(value).strip() in NO_CONTEXT_VALUES:
        return ""
    return str(value).strip()
test_df["context"] = test_df["context"].apply(clean_context)
test_df["has_context"] = test_df["context"].str.len() > 0

merged_with_context = pd.merge(merged, test_df[["id", "has_context"]], on="id")

ob_overlap = (merged_with_context[merged_with_context["has_context"]]["label_user"] == 
              merged_with_context[merged_with_context["has_context"]]["label_local"]).mean()
print(f"Open-Book overlap: {ob_overlap:.4f}")

cb_overlap = (merged_with_context[~merged_with_context["has_context"]]["label_user"] == 
              merged_with_context[~merged_with_context["has_context"]]["label_local"]).mean()
print(f"Closed-Book overlap: {cb_overlap:.4f}")
