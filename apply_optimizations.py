import re

with open('build_improved_notebook.py', 'r') as f:
    content = f.read()

# 1. Update FAST_BATCH and FAST_MAX_NEW in Cell 12
content = content.replace("FAST_MAX_NEW = 64     \nFAST_BATCH   = 2", "FAST_MAX_NEW = 48     \nFAST_BATCH   = 24")

# 2. Update SC_BATCH in Cell 12
content = content.replace("SC_BATCH = 1", "SC_BATCH = 8")

# 3. Add Circuit Breaker to fast_judge and SC in Cell 12
old_fast_judge = "    for s in range(0, len(rows), FAST_BATCH):\n        batch = rows.iloc[s:s+FAST_BATCH]"
new_fast_judge = """    for s in range(0, len(rows), FAST_BATCH):
        if 'time_left' in globals() and time_left() < 600:
            print("\\nWARNING: Time budget almost exhausted! Breaking fast_judge early.")
            break
        batch = rows.iloc[s:s+FAST_BATCH]"""
content = content.replace(old_fast_judge, new_fast_judge)

old_sc = "    for s in range(0, len(uncertain_rows), SC_BATCH):\n        batch = uncertain_rows.iloc[s:s+SC_BATCH]"
new_sc = """    for s in range(0, len(uncertain_rows), SC_BATCH):
        if 'time_left' in globals() and time_left() < 600:
            print("\\nWARNING: Time budget almost exhausted! Breaking SC early.")
            break
        batch = uncertain_rows.iloc[s:s+SC_BATCH]"""
content = content.replace(old_sc, new_sc)

# 4. Enhance fast_route in Cell 12
old_fast_route = """def fast_route(df):
    \"\"\"Two-stage routing: auto-label easy rows, send hard ones to LLM.
    Returns (easy_labels, easy_conf, hard_mask) where hard_mask marks rows needing LLM.\"\"\"
    rows = df.reset_index(drop=True)
    labels = np.full(len(rows), -1, int)
    conf = np.full(len(rows), 0.5)
    hard_mask = np.ones(len(rows), dtype=bool)  # assume all hard initially
    
    for i, (_, row) in enumerate(rows.iterrows()):
        g = row.get("ground", 0.0)
        # Stage 1: auto-label rows with very strong grounding signal
        if g <= -0.55:  # strong number mismatch -> very likely hallucinated
            labels[i] = 0
            conf[i] = 0.15
            hard_mask[i] = False
    
    auto_count = (~hard_mask).sum()
    print(f"Two-stage routing: {auto_count}/{len(rows)} auto-labeled, {hard_mask.sum()} sent to LLM")
    return labels, conf, hard_mask"""

new_fast_route = """def fast_route(df):
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
content = content.replace(old_fast_route, new_fast_route)

# 5. Cell 16 wiring fast_route
old_c16_llm = """if LLM_OK:
    print(f"\\nScoring dev with the judge (mode={JUDGE_MODE}, samples={N_SAMPLES})...")
    dl, dc = fast_judge(dev)"""
new_c16_llm = """if LLM_OK:
    print(f"\\nScoring dev with the judge (mode={JUDGE_MODE}, samples={N_SAMPLES})...")
    dl_auto, dc_auto, hard_mask = fast_route(dev)
    dl, dc = dl_auto.copy(), dc_auto.copy()
    sub = dev[hard_mask].reset_index(drop=True)
    sub_l, sub_c = fast_judge(sub)
    dl[hard_mask] = sub_l
    dc[hard_mask] = sub_c"""
content = content.replace(old_c16_llm, new_c16_llm)

# 6. Cell 18 timer and fast_route + delete redefinitions
c18_regex = r"FAST_MAX_NEW = 64\s+FAST_BATCH   = 8\s+CHECK_EVERY  = 8\s+"
content = re.sub(c18_regex, "import time\nSTART_TIME = time.time()\nBUDGET_S = 8.5 * 3600\ndef time_left(): return BUDGET_S - (time.time() - START_TIME)\n\nCHECK_EVERY = 8\n", content)

old_c18_llm = """if LLM_OK:
    print("\\nJudging test set (fast CoT: early-stop + greedy)...")
    tl, tc = (fast_judge(test) if JUDGE_MODE == "cot" else judge(test))"""
new_c18_llm = """if LLM_OK:
    print("\\nJudging test set (fast CoT: early-stop + greedy)...")
    tl_auto, tc_auto, hard_mask = fast_route(test)
    tl, tc = tl_auto.copy(), tc_auto.copy()
    sub = test[hard_mask].reset_index(drop=True)
    sub_l, sub_c = (fast_judge(sub) if JUDGE_MODE == "cot" else judge(sub))
    tl[hard_mask] = sub_l
    tc[hard_mask] = sub_c"""
content = content.replace(old_c18_llm, new_c18_llm)

# 7. Add Cell 10 replacement for device_map
cell10_injection = """
# ============================================================
# CELL 10: Model Loading (pin to device_map={"": 0})
# ============================================================
import json
with open('backup_last_10-16-pm.ipynb') as f:
    backup_nb = json.load(f)
orig_cell10 = ''.join(backup_nb['cells'][10]['source'])
new_cell10 = orig_cell10.replace('device_map="auto"', 'device_map={"": 0}')
set_cell(10, new_cell10.split('\\n'))
"""

parts = content.split("# ============================================================\n# CELL 12")
if len(parts) == 2:
    content = parts[0] + cell10_injection + "\n# ============================================================\n# CELL 12" + parts[1]

with open('build_improved_notebook.py', 'w') as f:
    f.write(content)
