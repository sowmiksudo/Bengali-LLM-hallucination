import re

with open('build_improved_notebook.py', 'r') as f:
    content = f.read()

# --- 2. Global Time Budget Fix ---
# Add GLOBAL_START_TIME to Cell 2 (at the very end of the cell)
old_c2_end = "print(\"dev :\", DEV_PATH, \"\\ntest:\", TEST_PATH)"
new_c2_end = "print(\"dev :\", DEV_PATH, \"\\ntest:\", TEST_PATH)\nimport time\nGLOBAL_START_TIME = time.time()"
content = content.replace(old_c2_end, new_c2_end)

# Update Cell 18 budget
old_budget = "import time\nSTART_TIME = time.time()\nBUDGET_S = 8.5 * 3600\ndef time_left(): return BUDGET_S - (time.time() - START_TIME)"
new_budget = "import time\nSTART_TIME = time.time()\nBUDGET_S = 9 * 3600 - (time.time() - GLOBAL_START_TIME) - 600\ndef time_left(): return BUDGET_S - (time.time() - START_TIME)"
content = content.replace(old_budget, new_budget)


# --- 3A. GPU Memory Accounting in Cell 10 ---
old_gpu_mem = "TOTAL_GB = GPU_GB * max(N_GPU, 1)"
new_gpu_mem = "TOTAL_GB = GPU_GB  # Pinned to 1 GPU"
content = content.replace(old_gpu_mem, new_gpu_mem)


# --- 3B & 3C. Math Parser and Idiom Loop in fast_route (Cell 12) ---
# We need to replace the body of fast_route.
old_fast_route = """def fast_route(df):
    \"\"\"Two-stage routing: auto-label easy rows, send hard ones to LLM.\"\"\"
    rows = df.reset_index(drop=True)
    labels = np.full(len(rows), -1, int)
    conf = np.full(len(rows), 0.5)
    hard_mask = np.ones(len(rows), dtype=bool)
    
    for i, (_, row) in enumerate(rows.iterrows()):
        g = row.get("ground", 0.0)
        p_text = str(row.get("prompt_bn", ""))
        r_text = str(row.get("response_bn", ""))
        
        # 1. Grounding Heuristic
        if g <= -0.55:
            labels[i] = 0; conf[i] = 0.15; hard_mask[i] = False
            continue
            
        # 2. Math Parser
        math_match = re.search(r'(\d+)\s*([\+\-\*\/])\s*(\d+)', str(p_text).translate(_BN2EN))
        if math_match:
            try:
                a, op, b = int(math_match.group(1)), math_match.group(2), int(math_match.group(3))
                ans = None
                if op == '+': ans = a + b
                elif op == '-': ans = a - b
                elif op == '*': ans = a * b
                elif op == '/': ans = a // b
                
                if ans is not None and str(ans) not in r_text.translate(_BN2EN):
                    labels[i] = 0; conf[i] = 0.15; hard_mask[i] = False
                    continue
            except Exception: pass

        # 3. Idiom Check
        idiom_found = False
        for idiom, meaning in IDIOM_DICT.items():
            if idiom in p_text:
                idiom_found = True
                m_tokens = set(meaning.split())
                r_tokens = set(r_text.split())
                if len(m_tokens & r_tokens) > len(m_tokens) * 0.3:
                    labels[i] = 1; conf[i] = 0.85; hard_mask[i] = False
                    break
        if idiom_found and not hard_mask[i]:
            continue
            
    auto_count = (~hard_mask).sum()
    print(f"Two-stage routing: {auto_count}/{len(rows)} auto-labeled, {hard_mask.sum()} sent to LLM")
    return labels, conf, hard_mask"""

new_fast_route = """# Precompile idiom regex for O(1) matching
IDIOM_REGEX = None

def fast_route(df):
    \"\"\"Two-stage routing: auto-label easy rows, send hard ones to LLM.\"\"\"
    global IDIOM_REGEX
    if IDIOM_REGEX is None and IDIOM_DICT:
        IDIOM_REGEX = re.compile(r'|'.join(map(re.escape, IDIOM_DICT.keys())))

    rows = df.reset_index(drop=True)
    labels = np.full(len(rows), -1, int)
    conf = np.full(len(rows), 0.5)
    hard_mask = np.ones(len(rows), dtype=bool)
    
    for i, (_, row) in enumerate(rows.iterrows()):
        g = row.get("ground", 0.0)
        p_text = str(row.get("prompt_bn", ""))
        r_text = str(row.get("response_bn", ""))
        
        # 1. Grounding Heuristic
        if g <= -0.55:
            labels[i] = 0; conf[i] = 0.15; hard_mask[i] = False
            continue
            
        # 2. Math Parser
        math_match = re.search(r'(\d+)\s*([\+\-\*\/])\s*(\d+)', str(p_text).translate(_BN2EN))
        if math_match:
            try:
                a, op, b = int(math_match.group(1)), math_match.group(2), int(math_match.group(3))
                ans = None
                if op == '+': ans = a + b
                elif op == '-': ans = a - b
                elif op == '*': ans = a * b
                elif op == '/': ans = a / b
                
                if ans is not None:
                    # Format ans: if it's an integer, strip .0
                    ans_str = f"{ans:.1f}".rstrip('0').rstrip('.')
                    if ans_str not in r_text.translate(_BN2EN):
                        labels[i] = 0; conf[i] = 0.15; hard_mask[i] = False
                        continue
            except Exception: pass

        # 3. Idiom Check
        if IDIOM_REGEX:
            idiom_match = IDIOM_REGEX.search(p_text)
            if idiom_match:
                idiom = idiom_match.group(0)
                meaning = IDIOM_DICT[idiom]
                m_tokens = set(meaning.split())
                r_tokens = set(r_text.split())
                if len(m_tokens & r_tokens) > len(m_tokens) * 0.3:
                    labels[i] = 1; conf[i] = 0.85; hard_mask[i] = False
                    continue
                # If idiom found but meaning not present, mark as hallucinated? No, fallback to LLM.
                
    auto_count = (~hard_mask).sum()
    print(f"Two-stage routing: {auto_count}/{len(rows)} auto-labeled, {hard_mask.sum()} sent to LLM")
    return labels, conf, hard_mask"""
content = content.replace(old_fast_route, new_fast_route)

# --- 4. RAG Context Backfilling in Cell 2 ---
# Right before print("dev :", DEV_PATH...)
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
content = content.replace("print(\"dev :\", DEV_PATH, \"\\ntest:\", TEST_PATH)\nimport time", 
                          backfill_code + "\nprint(\"dev :\", DEV_PATH, \"\\ntest:\", TEST_PATH)\nimport time")

with open('build_improved_notebook.py', 'w') as f:
    f.write(content)
