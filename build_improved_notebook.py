#!/usr/bin/env python3
"""Build improved notebook from backup with 6 accuracy improvements.
Uses JSON cell definitions to avoid Python string escaping issues."""
import json, copy

with open('backup_last_10-16-pm.ipynb') as f:
    nb = json.load(f)

nb_new = copy.deepcopy(nb)

def set_cell(idx, lines):
    """Set cell source from a list of lines (each line gets \\n appended except last)."""
    result = [line + '\n' for line in lines[:-1]]
    result.append(lines[-1])
    nb_new['cells'][idx]['source'] = result
    nb_new['cells'][idx]['outputs'] = []
    nb_new['cells'][idx]['execution_count'] = None

# ============================================================
# CELL 0: Updated title
# ============================================================
set_cell(0, [
    "# অলীকবচন — Bengali Hallucination Detection (Improved)",
    "**Chain-of-Thought LLM-as-Judge · NLI cross-encoder · dynamic few-shot · self-consistency · grounding override.**"
])


# ============================================================
# CELL 2: Multi-Dataset RAG and Setup
# ============================================================
import json
with open('backup_last_10-16-pm.ipynb') as f:
    backup_nb = json.load(f)
orig_cell2 = ''.join(backup_nb['cells'][2]['source'])

# Replace the OFFLINE RAG IMPLEMENTATION block in Cell 2
old_rag_block = """# --- OFFLINE RAG IMPLEMENTATION ---
OFFLINE_RAG_CORPUS = []
rag_vectorizer = None
rag_tfidf_matrix = None

rag_file = _find("BanglaNLP_Wikipedia_Corpus.csv")
if rag_file:
    try:
        print("Loading offline RAG corpus from:", rag_file)
        rag_df = pd.read_csv(rag_file).dropna(subset=["text", "title"])
        OFFLINE_RAG_CORPUS = rag_df.to_dict("records")
        print(f"Loaded {len(OFFLINE_RAG_CORPUS)} RAG documents.")
        
        # Build TF-IDF index for fast retrieval
        corpus_texts = [f"{row['title']} {row['text']}" for row in OFFLINE_RAG_CORPUS]
        rag_vectorizer = TfidfVectorizer(max_features=25000, stop_words=None)
        rag_tfidf_matrix = rag_vectorizer.fit_transform(corpus_texts)
        print("RAG TF-IDF index built successfully.")
    except Exception as e:
        print("Error loading RAG corpus:", e)

def retrieve_context(query, k=1):
    if not OFFLINE_RAG_CORPUS or rag_vectorizer is None or rag_tfidf_matrix is None:
        return ""
    try:
        query_vec = rag_vectorizer.transform([query])
        sims = cosine_similarity(query_vec, rag_tfidf_matrix)[0]
        best_idx = sims.argmax()
        if sims[best_idx] > 0.12:  # Similarity threshold
            doc = OFFLINE_RAG_CORPUS[best_idx]
            return f"Source: {doc['title']}\\nCategory: {doc['category']}\\nContext: {doc['text']}"
    except Exception as e:
        print("Retrieval error:", e)
    return ""
"""

new_rag_block = """# --- OFFLINE MULTI-DOMAIN RAG IMPLEMENTATION ---
OFFLINE_RAG_CORPUS = []
rag_vectorizer = None
rag_tfidf_matrix = None

dataset_files = [
    ("BanglaNLP_Wikipedia_Corpus.csv", "Wikipedia"),
    ("bengali-math-cot-dataset.csv", "Math"),
    ("Math_Bangla_Augmented.csv", "Math"),
    ("Bangladesh-Legal-Acts-Dataset.csv", "BDLaws"),
    ("Textbook_Dataset_from_NCTB.csv", "NCTB"),
    ("bangla_newspaper_dataset.csv", "News"),
    ("bangladesh_geography.csv", "Geography")
]

for fname, domain in dataset_files:
    path = _find(fname)
    if path:
        try:
            print(f"Loading {domain} corpus from:", path)
            df = pd.read_csv(path)
            text_col = 'text' if 'text' in df.columns else df.columns[0]
            title_col = 'title' if 'title' in df.columns else (df.columns[1] if len(df.columns) > 1 else df.columns[0])
            df = df.dropna(subset=[text_col])
            
            for _, row in df.iterrows():
                OFFLINE_RAG_CORPUS.append({
                    "title": str(row[title_col]),
                    "text": str(row[text_col]),
                    "domain": domain
                })
        except Exception as e:
            print(f"Error loading {fname}:", e)

if OFFLINE_RAG_CORPUS:
    print(f"Loaded {len(OFFLINE_RAG_CORPUS)} RAG documents from multiple domains.")
    corpus_texts = [f"{doc['title']} {doc['text']}" for doc in OFFLINE_RAG_CORPUS]
    rag_vectorizer = TfidfVectorizer(max_features=40000, stop_words=None)
    rag_tfidf_matrix = rag_vectorizer.fit_transform(corpus_texts)
    print("Multi-domain RAG TF-IDF index built successfully.")

def retrieve_context(query, k=1):
    if not OFFLINE_RAG_CORPUS or rag_vectorizer is None or rag_tfidf_matrix is None:
        return ""
    try:
        query_vec = rag_vectorizer.transform([query])
        sims = cosine_similarity(query_vec, rag_tfidf_matrix)[0]
        
        # Domain-Aware Routing Heuristics
        math_keywords = ['+', '-', '*', '/', '=', 'সংখ্যা', 'কত', 'যোগ', 'বিয়োগ', 'গুণ', 'ভাগ', 'মান', 'নির্ণয়']
        legal_keywords = ['আইন', 'ধারা', 'দণ্ডবিধি', 'অপরাধ', 'শাস্তি', 'আদালত', 'মামলা', 'সংবিধান']
        geo_keywords = ['জেলা', 'নদী', 'অবস্থিত', 'আয়তন', 'উপজেলা', 'রাজধানী', 'দেশ', 'সীমানা']
        
        q_lower = query.lower()
        is_math = any(kw in q_lower for kw in math_keywords)
        is_legal = any(kw in q_lower for kw in legal_keywords)
        is_geo = any(kw in q_lower for kw in geo_keywords)
        
        for i, doc in enumerate(OFFLINE_RAG_CORPUS):
            dom = doc.get("domain", "")
            if is_math and dom == "Math":
                sims[i] *= 1.5
            elif is_legal and dom == "BDLaws":
                sims[i] *= 1.5
            elif is_geo and dom in ["Geography", "Wikipedia"]:
                sims[i] *= 1.2

        best_idx = sims.argmax()
        if sims[best_idx] > 0.12:  # Similarity threshold
            doc = OFFLINE_RAG_CORPUS[best_idx]
            return f"Source: {doc['title']} (Domain: {doc.get('domain', 'Unknown')})\\nContext: {doc['text']}"
    except Exception as e:
        print("Retrieval error:", e)
    return ""
"""

new_cell2 = orig_cell2.replace(old_rag_block, new_rag_block)
set_cell(2, new_cell2.split('\n'))

# ============================================================
# CELL 6: Grounding signal — re-enable English leak detection
# ============================================================
# Load from external file to avoid escaping hell
cell6_code = '''

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

_BN2EN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
def extract_numbers(s): return set(re.findall(r"\\d+", str(s).translate(_BN2EN)))
def token_overlap(ctx, resp):
    c = set(str(ctx).translate(_BN2EN).split()); r = set(str(resp).translate(_BN2EN).split())
    return (len(c & r) / len(r)) if r else 0.0

def grounding_signal(row):
    """Returns a CONTINUOUS score in [-1, +1] range. Used as soft signal, never as hard override."""
    if not row["has_context"]: return 0.0
    score = 0.0
    rn, cn = extract_numbers(row["response_bn"]), extract_numbers(row["context"])
    if rn:
        # Numbers in response not in context -> strong negative signal
        mismatched = rn - cn
        if mismatched:
            score -= 0.6 * (len(mismatched) / len(rn))  # proportional, not binary
    
    # Token overlap gives positive signal
    overlap = token_overlap(row["context"], row["response_bn"])
    score += 0.3 * overlap  # continuous, not threshold
    
    return np.clip(score, -1.0, 1.0)

dev["ground"]  = dev.apply(grounding_signal, axis=1)
test["ground"] = test.apply(grounding_signal, axis=1)
print("grounding | dev non-zero:", int((dev['ground']!=0).sum()), "| test non-zero:", int((test['ground']!=0).sum()))
print("grounding | dev range:", f"[{dev['ground'].min():.2f}, {dev['ground'].max():.2f}]")
'''
set_cell(6, cell6_code.strip().split('\n'))


# ============================================================
# CELL 10: Model Loading (pin to device_map={"": 0})
# ============================================================
import json
with open('backup_last_10-16-pm.ipynb') as f:
    backup_nb = json.load(f)
orig_cell10 = ''.join(backup_nb['cells'][10]['source'])
new_cell10 = orig_cell10.replace('device_map="auto"', 'device_map={"": 0}')
set_cell(10, new_cell10.split('\n'))

# ============================================================
# CELL 12: Judge — bilingual prompt, dynamic few-shot, NLI, SC, threshold
# ============================================================
cell12_code = r'''import numpy as np
import torch
import re
from transformers import StoppingCriteria, StoppingCriteriaList
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- FAST INFERENCE PACING PARAMETERS ---
FAST_MAX_NEW = 48     
FAST_BATCH   = 24       
CHECK_EVERY  = 8      

# --- GLOBAL FEW-SHOT MATRIX INDEXING (for dynamic selection) ---
fs_prompts_global = [p for _, p, _, _, _ in FEWSHOT]
global_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4)).fit(fs_prompts_global)
global_tfidf_matrix = global_vectorizer.transform(fs_prompts_global)

# --- NLI MODEL LOADING ---
NLI_MODEL = None
NLI_TOKENIZER = None
NLI_OK = False
try:
    from transformers import AutoTokenizer as AT2, AutoModelForSequenceClassification
    nli_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    # Try local first
    import glob as _g
    nli_local = [d for d in _g.glob("/kaggle/input/**/config.json", recursive=True)
                 if "deberta" in os.path.dirname(d).lower() or "xnli" in os.path.dirname(d).lower()]
    nli_src = os.path.dirname(nli_local[0]) if nli_local else nli_name
    print(f"Loading NLI model from: {nli_src}")
    NLI_TOKENIZER = AT2.from_pretrained(nli_src)
    NLI_MODEL = AutoModelForSequenceClassification.from_pretrained(nli_src).eval()
    if HAS_GPU:
        NLI_MODEL = NLI_MODEL.half().cuda()
    NLI_OK = True
    print("NLI cross-encoder loaded successfully")
except Exception as e:
    print(f"NLI model not available ({repr(e)[:100]}), continuing without it")

def nli_score_batch(contexts, responses, batch_size=32):
    """Score (context, response) pairs for entailment/contradiction using NLI model."""
    if not NLI_OK:
        return np.full(len(contexts), 0.5), np.full(len(contexts), 0.0)
    
    entail_scores = np.full(len(contexts), 0.5)
    contra_scores = np.full(len(contexts), 0.0)
    
    for s in range(0, len(contexts), batch_size):
        batch_ctx = contexts[s:s+batch_size]
        batch_resp = responses[s:s+batch_size]
        
        pairs = [(str(c)[:512], str(r)[:512]) for c, r in zip(batch_ctx, batch_resp)]
        enc = NLI_TOKENIZER(
            [p[0] for p in pairs], [p[1] for p in pairs],
            return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        if HAS_GPU:
            enc = {k: v.cuda() for k, v in enc.items()}
        
        with torch.no_grad():
            logits = NLI_MODEL(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        
        # mDeBERTa-xnli: 0=contradiction, 1=neutral, 2=entailment
        for i in range(len(pairs)):
            entail_scores[s+i] = float(probs[i, 2])
            contra_scores[s+i] = float(probs[i, 0])
    
    return entail_scores, contra_scores

class StopWhenAllVerdicts(StoppingCriteria):
    """Stop the batch as soon as every row has emitted a parseable VERDICT."""
    def __init__(self, tok, plen): self.tok, self.plen, self.step = tok, plen, 0
    def __call__(self, input_ids, scores, **kw):
        self.step += 1
        if self.step % CHECK_EVERY: return False
        gen = input_ids[:, self.plen:]
        return all(_parse_verdict(t) is not None for t in self.tok.batch_decode(gen, skip_special_tokens=True))

# --- SYSTEM PROMPT (English-consistent to avoid 4-bit alignment drift) ---
# Bengali domain terms are kept but structure/commands stay in English
# to prevent language-switching confusion in quantized models.
SYS_COT = (
    "You are a meticulous Bengali fact-checking expert. Evaluate the RESPONSE for these hallucination types:\n"
    "1. Factual Inconsistency: Do dates, names, numbers, or entities contradict the CONTEXT or world knowledge?\n"
    "2. Reasoning/Math Errors: Calculate step-by-step for mathematical questions before determining the verdict. Is the logic, arithmetic, or calculation flawed?\n"
    "3. Temporal Errors: Extract and compare dates/years carefully for temporal questions.\n"
    "4. Idiomatic Errors (বাগধারা): Does the response give a LITERAL meaning of a Bengali idiom instead of its figurative meaning?\n"
    "5. Code-Mixing Errors: Are English/Bengali translations or transliterations incorrect?\n"
    "6. World Knowledge: Is the response factually wrong based on common knowledge?\n\n"
    "Reason step-by-step in 2-3 sentences, then output exactly: 'VERDICT: 1' (faithful) or 'VERDICT: 0' (hallucinated)."
)

SYS_LOGIT = (
    "You are a Bengali fact-checking expert. Output 1 if the RESPONSE is faithful/correct (and, when "
    "CONTEXT exists, supported by it); output 0 if it is false/fabricated. Answer with ONE character."
)
FINAL_COT   = "Reason briefly, then output 'VERDICT: 1' or 'VERDICT: 0'."
FINAL_LOGIT = "Is the RESPONSE faithful? Answer 1 or 0."

def _user_text(ctx, prompt, resp, final_q):
    parts = []
    if ctx: parts.append(f"CONTEXT:\n{ctx[:MAX_CTX_CHARS]}")
    parts += [f"PROMPT:\n{prompt}", f"RESPONSE:\n{resp}", final_q]
    return "\n\n".join(parts)

def _select_fewshot(query_prompt, k=3):
    """Dynamically select the top-k most similar few-shot exemplars."""
    try:
        qvec = global_vectorizer.transform([query_prompt])
        sims = cosine_similarity(qvec, global_tfidf_matrix)[0]
        top_idx = sims.argsort()[-k:][::-1]
        return [FEWSHOT[i] for i in top_idx]
    except Exception:
        return FEWSHOT[:k]

def _messages(row):
    prompt_text = str(row["prompt_bn"])
    context_text = str(row["context"])
    response_text = str(row["response_bn"])
    
    # Idiom Injection
    for idiom, meaning in IDIOM_DICT.items():
        if idiom in prompt_text or idiom in response_text:
            prompt_text += f"\n\nNote: '{idiom}' is a Bengali idiom meaning '{meaning}'. Evaluate figurative meaning, not literal."
            break
            
    # Offline RAG for Closed-Book
    if not row["has_context"]:
        retrieved = retrieve_context(prompt_text)
        if retrieved:
            context_text = retrieved
            
    if JUDGE_MODE == "cot":
        m = [{"role": "system", "content": SYS_COT}]
        
        # Dynamic few-shot selection based on query similarity
        selected_fs = _select_fewshot(prompt_text, k=3)
        for c, p, r, reason, v in selected_fs:
            m.append({"role": "user", "content": _user_text(c, p, r, FINAL_COT)})
            m.append({"role": "assistant", "content": f"{reason}\nVERDICT: {v}"})
                
        m.append({"role": "user", "content": _user_text(context_text, prompt_text, response_text, FINAL_COT)})
    else:
        m = [{"role": "system", "content": SYS_LOGIT}]
        for c, p, r, reason, v in FEWSHOT[:N_FEWSHOT]:
            m.append({"role": "user", "content": _user_text(c, p, r, FINAL_LOGIT)})
            m.append({"role": "assistant", "content": str(v)})
        m.append({"role": "user", "content": _user_text(context_text, prompt_text, response_text, FINAL_LOGIT)})
    return m

_VERDICT_RE = re.compile(r"VERDICT\s*[:：]?\s*([01])")
def _parse_verdict(text):
    hits = _VERDICT_RE.findall(text)
    if hits: return int(hits[-1])
    tail = text.strip()[-40:]
    d = re.findall(r"[01]", tail)
    return int(d[-1]) if d else None

def judge(df):
    rows = df.reset_index(drop=True)
    labels = np.full(len(rows), -1, dtype=int)
    conf   = np.full(len(rows), 0.5, dtype=float)
    for s in range(0, len(rows), BATCH_SIZE):
        batch = rows.iloc[s:s+BATCH_SIZE]
        texts = [tokenizer.apply_chat_template(_messages(r), tokenize=False, add_generation_prompt=True) for _, r in batch.iterrows()]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(model.device)
        with torch.no_grad():
            gkw = dict(max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.pad_token_id, num_return_sequences=N_SAMPLES)
            gkw.update(do_sample=False)
            out = model.generate(**enc, **gkw)
            gen = out[:, enc.input_ids.shape[1]:]
            dec = tokenizer.batch_decode(gen, skip_special_tokens=True)
            for i in range(len(batch)):
                comp = dec[i*N_SAMPLES:(i+1)*N_SAMPLES]
                votes = [v for v in (_parse_verdict(t) for t in comp) if v is not None]
                if votes:
                    mean_v = float(np.mean(votes))
                    labels[s+i] = int(round(mean_v))
                    conf[s+i] = mean_v
        del enc
        torch.cuda.empty_cache()
        print(f"  judged {s+len(batch)}/{len(rows)}", end="\r", flush=True)
    print()
    return labels, conf

def fast_judge(df):
    rows = df.reset_index(drop=True)
    labels = np.full(len(rows), -1, int)
    conf = np.full(len(rows), 0.5)
    for s in range(0, len(rows), FAST_BATCH):
        if 'time_left' in globals() and time_left() < 600:
            print("\nWARNING: Time budget almost exhausted! Breaking fast_judge early.")
            break
        batch = rows.iloc[s:s+FAST_BATCH]
        texts = [tokenizer.apply_chat_template(_messages(r), tokenize=False, add_generation_prompt=True) for _, r in batch.iterrows()]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(model.device)
        plen = enc.input_ids.shape[1]
        sc = StoppingCriteriaList([StopWhenAllVerdicts(tokenizer, plen)])
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=FAST_MAX_NEW, do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id, stopping_criteria=sc)
        for i, t in enumerate(tokenizer.batch_decode(out[:, plen:], skip_special_tokens=True)):
            v = _parse_verdict(t)
            if v is not None: 
                labels[s+i] = v
                conf[s+i] = float(v)
        del enc, out
        torch.cuda.empty_cache()
        print(f"  judged {s+len(batch)}/{len(rows)}", end="\r", flush=True)
    print()
    return labels, conf

def selective_self_consistency(df, labels, conf):
    """Re-run uncertain rows with sampling for self-consistency."""
    if not LLM_OK:
        return labels, conf
    
    # Identify uncertain rows: unresolved by greedy pass
    uncertain = (labels < 0)
    n_uncertain = uncertain.sum()
    
    if n_uncertain == 0:
        print("No uncertain rows to re-judge.")
        return labels, conf
    
    print(f"\nSelective self-consistency: re-judging {n_uncertain} uncertain rows with N=3 sampling...")
    
    rows = df.reset_index(drop=True)
    uncertain_rows = rows[uncertain].reset_index(drop=True)
    uncertain_indices = np.where(uncertain)[0]
    
    SC_SAMPLES = 3
    SC_BATCH = 8
    
    for s in range(0, len(uncertain_rows), SC_BATCH):
        if 'time_left' in globals() and time_left() < 600:
            print("\nWARNING: Time budget almost exhausted! Breaking SC early.")
            break
        batch = uncertain_rows.iloc[s:s+SC_BATCH]
        texts = [tokenizer.apply_chat_template(_messages(r), tokenize=False, add_generation_prompt=True) for _, r in batch.iterrows()]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(model.device)
        
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                num_return_sequences=SC_SAMPLES,
                pad_token_id=tokenizer.pad_token_id
            )
            gen = out[:, enc.input_ids.shape[1]:]
            dec = tokenizer.batch_decode(gen, skip_special_tokens=True)
            
            for i in range(len(batch)):
                orig_idx = uncertain_indices[s + i]
                comp = dec[i*SC_SAMPLES:(i+1)*SC_SAMPLES]
                votes = [v for v in (_parse_verdict(t) for t in comp) if v is not None]
                if votes:
                    mean_v = float(np.mean(votes))
                    labels[orig_idx] = int(round(mean_v))
                    conf[orig_idx] = mean_v
        
        del enc, out
        torch.cuda.empty_cache()
        print(f"  SC judged {s+len(batch)}/{len(uncertain_rows)}", end="\r", flush=True)
    
    print(f"\n  Self-consistency resolved {(labels[uncertain_indices] >= 0).sum()}/{n_uncertain} rows")
    return labels, conf

# --- CALIBRATED THRESHOLD ---
CALIBRATED_THRESHOLD = 0.5  # will be updated after dev evaluation

def finalize(labels, conf, ground, nli_contra=None, nli_entail=None, threshold=None):
    """Combine all signals into final predictions. Uses soft blending, not hard overrides."""
    if threshold is None:
        threshold = CALIBRATED_THRESHOLD
    
    # Start with LLM confidence as the primary signal
    combined = conf.copy()
    
    # Blend in grounding signal (soft, weighted)
    GROUND_WEIGHT = 0.15
    combined = combined + GROUND_WEIGHT * ground
    
    # Blend in NLI signal (soft, weighted) for rows that have it
    if nli_contra is not None and nli_entail is not None:
        NLI_WEIGHT = 0.20
        nli_signal = nli_entail - nli_contra  # range [-1, +1]
        combined = combined + NLI_WEIGHT * nli_signal
    
    # For unresolved LLM rows (label < 0), fall back to heuristic signals
    und = labels < 0
    combined = np.where(und, 0.5 + 0.4 * ground, combined)
    if nli_contra is not None:
        combined = np.where(und, combined + 0.3 * (nli_entail - nli_contra), combined)
    
    # Apply calibrated threshold
    out = (combined >= threshold).astype(int)
    return out

# Precompile idiom regex for O(1) matching
IDIOM_REGEX = None

def fast_route(df):
    """Two-stage routing: auto-label easy rows, send hard ones to LLM."""
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
    return labels, conf, hard_mask

print("Judge functions defined (soft ensemble, two-stage routing, dynamic few-shot, selective SC)")
'''
set_cell(12, cell12_code.strip().split('\n'))

# ============================================================
# CELL 16: Dev evaluation — NLI + threshold calibration
# ============================================================
cell16_code = r'''
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, classification_report
y = dev["label"].values
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
print(f"TF-IDF LR floor   macro-F1 = {f1_score(y, cross_val_predict(build_tfidf_lr(), dev, y, cv=cv), average='macro'):.4f}")

# --- NLI scoring on dev open-book rows ---
dev_nli_contra = np.zeros(len(dev))
dev_nli_entail = np.zeros(len(dev))
if NLI_OK:
    ob_mask = dev["has_context"].values
    if ob_mask.sum() > 0:
        print(f"\nScoring {ob_mask.sum()} open-book dev rows with NLI cross-encoder...")
        ob_entail, ob_contra = nli_score_batch(
            dev.loc[ob_mask, "context"].tolist(),
            dev.loc[ob_mask, "response_bn"].tolist()
        )
        dev_nli_contra[ob_mask] = ob_contra
        dev_nli_entail[ob_mask] = ob_entail
        print(f"  NLI scored | mean entailment={ob_entail.mean():.3f} | mean contradiction={ob_contra.mean():.3f}")

if LLM_OK:
    print(f"\nScoring dev with the judge (mode={JUDGE_MODE}, samples={N_SAMPLES})...")
    dl_auto, dc_auto, hard_mask = fast_route(dev)
    dl, dc = dl_auto.copy(), dc_auto.copy()
    sub = dev[hard_mask].reset_index(drop=True)
    sub_l, sub_c = fast_judge(sub)
    dl[hard_mask] = sub_l
    dc[hard_mask] = sub_c
    
    # Selective self-consistency on uncertain dev rows
    dl, dc = selective_self_consistency(dev, dl, dc)
    
    # --- Threshold calibration ---
    print("\nCalibrating threshold on dev set...")
    m = ~dev["prompt_bn"].isin(_FS_PROMPTS)
    ye = y[m.values]
    
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.20, 0.80, 0.01):
        preds_t = finalize(dl[m.values], dc[m.values], dev["ground"].values[m.values],
                          dev_nli_contra[m.values], dev_nli_entail[m.values], threshold=t)
        f = f1_score(ye, preds_t, average='macro')
        if f > best_f1:
            best_t, best_f1 = t, f
    
    CALIBRATED_THRESHOLD = best_t
    print(f"  Optimal threshold: {best_t:.2f} -> macro-F1 = {best_f1:.4f}")
    
    dfin = finalize(dl, dc, dev["ground"].values, dev_nli_contra, dev_nli_entail, threshold=best_t)
    pe = dfin[m.values]
    print(f"\nresolved by judge: {(dl>=0).mean():.1%}")
    print(f"JUDGE + NLI + grounding macro-F1 = {f1_score(ye, pe, average='macro'):.4f}   <-- IMPROVED (dev)")
    print("\n" + classification_report(ye, pe, target_names=["hallucinated(0)","faithful(1)"]))
else:
    print("Judge unavailable -> N3 uses the TF-IDF + grounding fallback (see section 5 banner).")
'''
set_cell(16, cell16_code.strip().split('\n'))

# ============================================================
# CELL 18: Test prediction — NLI + SC + calibrated threshold
# ============================================================
cell18_code = r'''
import numpy as np, torch
from transformers import StoppingCriteria, StoppingCriteriaList

import time
START_TIME = time.time()
BUDGET_S = 9 * 3600 - (time.time() - GLOBAL_START_TIME) - 600
def time_left(): return BUDGET_S - (time.time() - START_TIME)

CHECK_EVERY = 8
class StopWhenAllVerdicts(StoppingCriteria):
    """Stop the batch as soon as every row has emitted a parseable VERDICT."""
    def __init__(self, tok, plen): self.tok, self.plen, self.step = tok, plen, 0
    def __call__(self, input_ids, scores, **kw):
        self.step += 1
        if self.step % CHECK_EVERY: return False
        gen = input_ids[:, self.plen:]
        return all(_parse_verdict(t) is not None
                   for t in self.tok.batch_decode(gen, skip_special_tokens=True))

# --- NLI scoring on test open-book rows ---
test_nli_contra = np.zeros(len(test))
test_nli_entail = np.zeros(len(test))
if NLI_OK:
    ob_mask_test = test["has_context"].values
    if ob_mask_test.sum() > 0:
        print(f"Scoring {ob_mask_test.sum()} open-book test rows with NLI cross-encoder...")
        t_entail, t_contra = nli_score_batch(
            test.loc[ob_mask_test, "context"].tolist(),
            test.loc[ob_mask_test, "response_bn"].tolist()
        )
        test_nli_contra[ob_mask_test] = t_contra
        test_nli_entail[ob_mask_test] = t_entail
        print(f"  NLI scored | mean entailment={t_entail.mean():.3f} | mean contradiction={t_contra.mean():.3f}")

if LLM_OK:
    print("\nJudging test set (fast CoT: early-stop + greedy)...")
    tl_auto, tc_auto, hard_mask = fast_route(test)
    tl, tc = tl_auto.copy(), tc_auto.copy()
    sub = test[hard_mask].reset_index(drop=True)
    sub_l, sub_c = (fast_judge(sub) if JUDGE_MODE == "cot" else judge(sub))
    tl[hard_mask] = sub_l
    tc[hard_mask] = sub_c
    
    # Selective self-consistency on uncertain test rows
    tl, tc = selective_self_consistency(test, tl, tc)
    
    test_pred = finalize(tl, tc, test["ground"].values, test_nli_contra, test_nli_entail, threshold=CALIBRATED_THRESHOLD)
    path_used = f"PRIMARY LLM JUDGE ({ACTIVE_MODEL}, mode={JUDGE_MODE}) + NLI + SC + threshold={CALIBRATED_THRESHOLD:.2f}"
else:
    mdl = build_tfidf_lr(); mdl.fit(dev, y)
    fb = np.clip(mdl.predict_proba(test)[:, 1] + 0.25 * test["ground"].values, 0, 1)
    fb = np.where(test["ground"].values <= -0.8, np.minimum(fb, 0.25), fb)
    test_pred = (fb >= 0.50).astype(int)
    path_used = "FALLBACK (TF-IDF + grounding)"

submission = pd.DataFrame({"id": test["id"].values, "label": np.asarray(test_pred).astype(int)})
assert submission["label"].isin([0, 1]).all() and len(submission) == len(test)
submission.to_csv("N3.csv", index=False)
print(f"\nPATH USED : {path_used}")
print(f"Wrote N3.csv | {len(submission)} rows | faithful share {submission['label'].mean():.3f}")
submission.head()
'''
set_cell(18, cell18_code.strip().split('\n'))

# Write the improved notebook
with open('n3-improved.ipynb', 'w') as f:
    json.dump(nb_new, f, ensure_ascii=False)

print("Created n3-improved.ipynb with all 6 improvements applied")

# Validate
with open('n3-improved.ipynb') as f:
    check = json.load(f)
print(f"Validation: {len(check['cells'])} cells, notebook JSON is valid")
