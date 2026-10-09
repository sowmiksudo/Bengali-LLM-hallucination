import re

with open('build_improved_notebook.py', 'r') as f:
    content = f.read()

backfill_code = """
def backfill_context(df):
    count = 0
    for i, row in df.iterrows():
        if not row.get("has_context", False):
            ctx = retrieve_context(str(row["prompt_bn"]))
            if ctx:
                df.at[i, "context"] = ctx
                df.at[i, "has_context"] = True
                count += 1
    return count

if "dev" in globals():
    c = backfill_context(dev)
    print(f"Backfilled RAG context for {c} dev rows.")
if "test" in globals():
    c = backfill_context(test)
    print(f"Backfilled RAG context for {c} test rows.")
"""

# Remove from Cell 2
content = content.replace(backfill_code + "\nprint(\"dev :\", DEV_PATH, \"\\ntest:\", TEST_PATH)\nimport time", 
                          "print(\"dev :\", DEV_PATH, \"\\ntest:\", TEST_PATH)\nimport time")

# Add to Cell 6
content = content.replace("cell6_code = '''\n_BN2EN = str.maketrans(\"০১২৩৪৫৬৭৮৯\", \"0123456789\")",
                          "cell6_code = '''\n" + backfill_code + "\n_BN2EN = str.maketrans(\"০১২৩৪৫৬৭৮৯\", \"0123456789\")")

with open('build_improved_notebook.py', 'w') as f:
    f.write(content)
