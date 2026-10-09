
import os, re, json, glob, socket, warnings, numpy as np, random
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore")


CAND_MODELS = [
    ("Qwen/Qwen2.5-32B-Instruct", 32),
    ("Qwen/Qwen2.5-14B-Instruct", 14),
    ("Qwen/Qwen2.5-7B-Instruct",   7),
    ("Qwen/Qwen2.5-3B-Instruct",   3),
]
MAX_PARAMS_B  = 14      
USE_4BIT      = True    

JUDGE_MODE    = "cot"   
N_SAMPLES     = 3       
MAX_NEW_TOKENS= 192     
N_FEWSHOT     = 6
MAX_CTX_CHARS = 1400
MAX_LEN       = 3072
BATCH_SIZE    = 4      
GROUND_WEIGHT = 0.15
SEED          = 42
random.seed(SEED); np.random.seed(SEED)
try:
    import torch; torch.manual_seed(SEED)
except Exception:
    pass


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# --- OFFLINE RAG IMPLEMENTATION ---
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
            return f"Source: {doc['title']}\nCategory: {doc['category']}\nContext: {doc['text']}"
    except Exception as e:
        print("Retrieval error:", e)
    return ""

# --- IDIOM LOOKUP IMPLEMENTATION ---
IDIOM_DICT = {}
idiom_file = _find("bagdhara_export_done_2026-03-01_010356.json", "bagdhara_export_done_2026-03-01_010405.jsonl")
if idiom_file:
    try:
        print("Loading idiom dictionary from:", idiom_file)
        if idiom_file.endswith(".json"):
            with open(idiom_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                entries = data if isinstance(data, list) else data.get("idioms", [])
                for entry in entries:
                    if isinstance(entry, dict) and "idiom" in entry:
                        idiom_text = str(entry["idiom"]).strip()
                        fig_meaning = str(entry.get("figurative_meaning_bn", "") or entry.get("figurative_meaning_en", "")).strip()
                        if idiom_text and fig_meaning:
                            IDIOM_DICT[idiom_text] = fig_meaning
        else: # JSONL
            with open(idiom_file, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    if isinstance(entry, dict) and "idiom" in entry:
                        idiom_text = str(entry["idiom"]).strip()
                        fig_meaning = str(entry.get("figurative_meaning_bn", "") or entry.get("figurative_meaning_en", "")).strip()
                        if idiom_text and fig_meaning:
                            IDIOM_DICT[idiom_text] = fig_meaning
        print(f"Loaded {len(IDIOM_DICT)} idioms into dictionary.")
    except Exception as e:
        print("Error loading idioms:", e)

def _find(*names):
    roots = ["/kaggle/input/competitions/bengali-hallucination",
             "/kaggle/input/bengali-hallucination", "bengali-hallucination", "."] \
            + glob.glob("/kaggle/input/*")
    for r in roots:
        for n in names:
            p = os.path.join(r, n)
            if os.path.exists(p): return p
    return None

DEV_PATH  = _find("dataset samples.json", "dataset_samples.json")
TEST_PATH = _find("test set.csv", "test.csv", "test_set.csv")
print("dev :", DEV_PATH, "\ntest:", TEST_PATH)


import pandas as pd
with open(DEV_PATH, encoding="utf-8") as f:
    dev = pd.DataFrame(json.load(f))
test = pd.read_csv(TEST_PATH)
if "id" not in test.columns:
    test = test.reset_index().rename(columns={"index": "id"})
NO_CONTEXT = {"", "nan", "NaN", "[NULL]", "null", "None", "none"}
def clean_context(v):
    if v is None: return ""
    s = str(v).strip(); return "" if s in NO_CONTEXT else s
for d in (dev, test):
    for c in ("prompt_bn", "response_bn"):
        d[c] = d[c].astype(str).fillna("")
    d["context"] = d["context"].apply(clean_context)
    d["has_context"] = d["context"].str.len() > 0
print(f"dev : {len(dev)} rows | context {dev['has_context'].mean():.0%} | faithful {dev['label'].mean():.0%}")
print(f"test: {len(test)} rows | context {test['has_context'].mean():.0%}")


_BN2EN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
def extract_numbers(s): return set(re.findall(r"\d+", str(s).translate(_BN2EN)))
def token_overlap(ctx, resp):
    c = set(str(ctx).translate(_BN2EN).split()); r = set(str(resp).translate(_BN2EN).split())
    return (len(c & r) / len(r)) if r else 0.0
def grounding_signal(row):
    if not row["has_context"]: return 0.0
    rn, cn = extract_numbers(row["response_bn"]), extract_numbers(row["context"])
    if rn: 
        if not rn.issubset(cn): return -0.8
        
    # Check for English characters (often hallucinations if not in context)
    eng_in_resp = set(re.findall(r"[a-zA-Z]+", str(row["response_bn"]).lower()))
    eng_in_ctx = set(re.findall(r"[a-zA-Z]+", str(row["context"]).lower()))
    if eng_in_resp and not eng_in_resp.issubset(eng_in_ctx):
        return -0.8
        
    return 0.2 if token_overlap(row["context"], row["response_bn"]) >= 0.5 else 0.0

dev["ground"]  = dev.apply(grounding_signal, axis=1)
test["ground"] = test.apply(grounding_signal, axis=1)
print("grounding | dev non-zero:", int((dev['ground']!=0).sum()), "| test non-zero:", int((test['ground']!=0).sum()))



FEWSHOT = [
    ("উইন্ডোজে ইউনিকোড ভিত্তিক বাংলা লেখার জন্য মেহদী হাসান খান ২০০৩ সালে অভ্র কীবোর্ড তৈরির কাজ শুরু করেন।",
     "অভ্র কিবোর্ড কে উদ্ভাবন করেন ?", "মেহদী হাসান খান",
     "The context explicitly credits Mehdi Hasan Khan with creating Avro. The response matches.", 1),
    ("পশ্চিমের মালভূমির সর্বোচ্চ অংশ অযোধ্যা পাহাড় (৬৭০ মিটার)।",
     "পুরুলিয়া জেলার অযোধ্যা পাহাড়ের উচ্চতা কত ?", "প্রায় ৭২০ মিটার।",
     "The context states 670 metres, but the response says 720 metres. The number contradicts the context.", 0),
    ("", "‘আগুনপাখি ‘ উপন্যাসের রচয়িতা কে?", "হাসান আজিজুল হক",
     "Aagunpakhi was written by Hasan Azizul Huq. The response is factually correct.", 1),
    ("", "‘হুলিয়া’ কবিতা কার রচনা?", "মহাদেব সাহা",
     "The poem 'Hulia' was written by Nirmalendu Goon, not Mahadev Saha. Wrong attribution.", 0),
    ("", "৩০ থেকে ৪০ পর্যন্ত সংখ্যা থেকে যেকোনো একটি নিলে সেটি মৌলিক বা ৫ এর গুণিতক হওয়ার সম্ভাবনা কত?", "৬/১১",
     "Integers 30..40 = 11 numbers; primes {31,37} plus multiples of 5 {30,35,40} = 5 favourable, so 5/11, not 6/11.", 0),
    ("", "আয়তনে পৃথিবীর সবচেয়ে ছোট দেশ?", "মালদ্বীপ",
     "The smallest country by area is Vatican City, not the Maldives. Factually wrong.", 0),
    ("", "‘আকাশ কুসুম’ বাগধারাটির অর্থ কী?", "অবাস্তব কল্পনা",
     "The Bengali idiom 'আকাশ কুসুম' means an impossible dream or absurd imagination. The response correctly provides the figurative meaning.", 1),
    ("বাংলাদেশ সংবিধানের ২ক অনুচ্ছেদ অনুযায়ী প্রজাতন্ত্রের রাষ্ট্রধর্ম ইসলাম।", "সংবিধান অনুযায়ী রাষ্ট্রধর্ম কী?", "হিন্দু ধর্ম",
     "The context states the state religion is Islam, but the response says Hinduism. Factual contradiction.", 0),
    ("", "মাথা খাওয়া বাগধারাটির অর্থ কী?", "কারো মাথা খেয়ে ফেলা",
     "The idiom 'মাথা খাওয়া' means to spoil someone or to swear on someone's head. The response gives a literal translation ('eating someone\'s head'), which is a hallucination of figurative meaning.", 0)
]
_FS_PROMPTS = {p for _, p, _, _, _ in FEWSHOT}
print(len(FEWSHOT), "few-shot exemplars loaded.")


def _banner(msg, ok=True):
    print("\n" + "="*72 + f"\n{'✅' if ok else '⚠️'}  {msg}\n" + "="*72 + "\n")

def has_internet(host="huggingface.co", port=443, timeout=4):
    try:
        with socket.create_connection((host, port), timeout=timeout): return True
    except Exception: return False

def discover_local_models():
    hits = []
    for cfg in glob.glob("/kaggle/input/**/config.json", recursive=True):
        d = os.path.dirname(cfg)
        if glob.glob(os.path.join(d, "*.safetensors")) or glob.glob(os.path.join(d, "*.bin")):
            hits.append(d)
    kw = ["instruct", "chat", "-it", "qwen", "llama", "gemma", "aya", "mistral", "phi"]
    return sorted(set(hits), key=lambda p: sum(w in p.lower() for w in kw), reverse=True)

try:
    import torch
    HAS_GPU = torch.cuda.is_available()
    N_GPU   = torch.cuda.device_count() if HAS_GPU else 0
    GPU_GB  = (torch.cuda.get_device_properties(0).total_memory/1e9) if HAS_GPU else 0.0
except Exception:
    HAS_GPU, N_GPU, GPU_GB = False, 0, 0.0
TOTAL_GB = GPU_GB * max(N_GPU, 1)
NET = has_internet()
LOCAL = discover_local_models()
print(f"GPU: {HAS_GPU} | #GPU: {N_GPU} | per-GPU: {GPU_GB:.1f} GB | total: {TOTAL_GB:.1f} GB")
print(f"Internet: {NET} | attached models: {LOCAL if LOCAL else 'none'}")

# ensure bitsandbytes for 4-bit
BNB_OK = False
if USE_4BIT and HAS_GPU:
    try:
        import bitsandbytes  # noqa
        BNB_OK = True
    except Exception:
        if NET:
            import subprocess, sys
            print("installing bitsandbytes ...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes"], check=False)
            try:
                import importlib; import bitsandbytes; importlib.reload(bitsandbytes); BNB_OK = True
            except Exception as e:
                print("bitsandbytes unavailable -> fp16:", repr(e)[:100])
print(f"4-bit available: {BNB_OK}")

CAND = [(m, p) for (m, p) in CAND_MODELS if p <= MAX_PARAMS_B]
plans = []
for lm in LOCAL:
    if BNB_OK: plans.append((lm, "4bit"))
    plans.append((lm, "fp16"))
if NET:
    if BNB_OK:
        for mid, p in CAND:
            if TOTAL_GB >= p*0.6 + 2:          
                plans.append((mid, "4bit"))
    for mid, p in CAND:
        if TOTAL_GB >= p*2 + 3:                
            plans.append((mid, "fp16"))

LLM_OK = False
tokenizer = model = tok_one = tok_zero = ACTIVE_MODEL = None
if not HAS_GPU:
    _banner("GPU is OFF -> skipping LLM (CPU impractical). Enable Accelerator -> GPU, Run all.", ok=False)
elif not plans:
    _banner("No model fits / no source. Use T4 x2 + Internet ON, or attach a model.", ok=False)
else:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    last_err = None
    for src, strat in plans:
        try:
            print(f"\nloading: {src}  [{strat}]")
            tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
            if tok.pad_token is None: tok.pad_token = tok.eos_token
            tok.padding_side = "left"
            kw = dict(trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
            if strat == "4bit":
                from transformers import BitsAndBytesConfig
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
            mdl = AutoModelForCausalLM.from_pretrained(src, **kw).eval()
            tokenizer, model, ACTIVE_MODEL = tok, mdl, f"{src} [{strat}]"
            tok_one  = tokenizer.encode("1", add_special_tokens=False)[-1]
            tok_zero = tokenizer.encode("0", add_special_tokens=False)[-1]
            if strat == "fp16": BATCH_SIZE = min(BATCH_SIZE, 2)
            LLM_OK = True; break
        except Exception as e:
            last_err = e; print(f"  failed: {repr(e)[:180]}")
            try: del mdl
            except Exception: pass
            try: torch.cuda.empty_cache()
            except Exception: pass
    if LLM_OK:
        try: free = torch.cuda.mem_get_info()[0]/1e9
        except Exception: free = float('nan')
        _banner(f"PRIMARY LLM JUDGE ACTIVE -> {ACTIVE_MODEL} | mode={JUDGE_MODE} | batch={BATCH_SIZE} | free~{free:.1f} GB", ok=True)
    else:
        _banner(f"All candidates failed -> FALLBACK. Last error: {repr(last_err)[:150]}", ok=False)


SYS_COT = (
    "You are a meticulous Bengali fact-checking expert. Evaluate the RESPONSE for the following hallucination types:\n"
    "1. Factual Inconsistency: Are dates, names, or numbers contradicting the CONTEXT or world knowledge?\n"
    "2. Reasoning Errors: Is the math or logic flawed?\n"
    "3. Idiomatic Errors: Does the response literally interpret a Bengali idiom (বাগধারা)?\n"
    "4. Code-Mixing Errors: Are English/Bengali translations or mixed-language terms misused?\n"
    "Reason step-by-step, then output exactly: 'VERDICT: 1' (faithful) or 'VERDICT: 0' (hallucinated)."
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

def _messages(row):
    prompt_text = str(row["prompt_bn"])
    context_text = str(row["context"])
    response_text = str(row["response_bn"])
    
    # 1. Idiom Injection
    for idiom, meaning in IDIOM_DICT.items():
        if idiom in prompt_text or idiom in response_text:
            prompt_text += f"\n\nNote: The phrase '{idiom}' is a Bengali idiom meaning '{meaning}'. Evaluate based on figurative meaning."
            break
            
    # 2. Offline RAG for Closed-Book
    if not row["has_context"]:
        retrieved = retrieve_context(prompt_text)
        if retrieved:
            context_text = retrieved
            
    if JUDGE_MODE == "cot":
        m = [{"role": "system", "content": SYS_COT}]
        
        # Dynamic Few-shot Selection using TF-IDF
        fs_prompts = [p for _, p, _, _, _ in FEWSHOT]
        if fs_prompts:
            vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4)).fit(fs_prompts + [prompt_text])
            tfidf_matrix = vectorizer.transform(fs_prompts)
            query_vec = vectorizer.transform([prompt_text])
            sims = cosine_similarity(query_vec, tfidf_matrix)[0]
            top_indices = sims.argsort()[-N_FEWSHOT:][::-1]
            for idx in top_indices:
                c, p, r, reason, v = FEWSHOT[idx]
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

def _last_logits(enc):
    import torch
    for kw in (dict(logits_to_keep=1), dict(num_logits_to_keep=1), dict()):
        try:
            return model(**enc, **kw).logits[:, -1, :]
        except TypeError:
            continue
    return model(**enc).logits[:, -1, :]

def judge(df):
    '''Return (labels[int], confidence[float in 0..1]).'''
    import torch
    rows = df.reset_index(drop=True)
    labels = np.full(len(rows), -1, dtype=int)
    conf   = np.full(len(rows), 0.5, dtype=float)
    for s in range(0, len(rows), BATCH_SIZE):
        batch = rows.iloc[s:s+BATCH_SIZE]
        texts = [tokenizer.apply_chat_template(_messages(r), tokenize=False, add_generation_prompt=True)
                 for _, r in batch.iterrows()]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=MAX_LEN).to(model.device)
        with torch.no_grad():
            if JUDGE_MODE == "cot":
                gkw = dict(max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.pad_token_id,
                           num_return_sequences=N_SAMPLES)
                if N_SAMPLES > 1:
                    gkw.update(do_sample=True, temperature=0.7, top_p=0.9)
                else:
                    gkw.update(do_sample=False)
                out = model.generate(**enc, **gkw)
                gen = out[:, enc.input_ids.shape[1]:]
                dec = tokenizer.batch_decode(gen, skip_special_tokens=True)
                for i in range(len(batch)):
                    comp = dec[i*N_SAMPLES:(i+1)*N_SAMPLES]
                    votes = [v for v in (_parse_verdict(t) for t in comp) if v is not None]
                    if votes:
                        mean_v = float(np.mean(votes))
                        labels[s+i] = int(round(mean_v)); conf[s+i] = mean_v
            else:
                last = _last_logits(enc)
                p = torch.softmax(last[:, [tok_one, tok_zero]].float(), -1)[:, 0].cpu().numpy()
                labels[s:s+len(batch)] = (p >= 0.5).astype(int); conf[s:s+len(batch)] = p
        del enc
        try: torch.cuda.empty_cache()
        except Exception: pass
        print(f"  judged {s+len(batch)}/{len(rows)}", end="\r")
    print()
    return labels, conf

def finalize(labels, conf, ground):
    out = labels.astype(float).copy()
    und = out < 0                                   
    out = np.where(und & (ground < 0), 0.0, out)
    out = np.where(und & (ground >= 0), 1.0, out)
    out = np.where(ground <= -0.8, 0.0, out)       
    return out.astype(int)


from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
def _c(n): return FunctionTransformer(lambda d, n=n: d[n].to_numpy(), validate=False)
def _b(n): return Pipeline([("s", _c(n)), ("t", TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4), max_features=20000))])
def build_tfidf_lr(): return Pipeline([("f", FeatureUnion([("p", _b("prompt_bn")), ("r", _b("response_bn"))])),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED))])


from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, classification_report
y = dev["label"].values
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
print(f"TF-IDF LR floor   macro-F1 = {f1_score(y, cross_val_predict(build_tfidf_lr(), dev, y, cv=cv), average='macro'):.4f}")

if LLM_OK:
    print(f"\nScoring dev with the judge (mode={JUDGE_MODE}, samples={N_SAMPLES})...")
    dl, dc = judge(dev)
    dfin = finalize(dl, dc, dev["ground"].values)
    m = ~dev["prompt_bn"].isin(_FS_PROMPTS)
    ye, pe = y[m.values], dfin[m.values]
    print(f"resolved by judge: {(dl>=0).mean():.1%}")
    print(f"JUDGE + grounding macro-F1 = {f1_score(ye, pe, average='macro'):.4f}   <-- N3 (dev)")
    print("\n" + classification_report(ye, pe, target_names=["hallucinated(0)","faithful(1)"]))
else:
    print("Judge unavailable -> N3 uses the TF-IDF + grounding fallback (see §5 banner).")


import numpy as np, torch
from transformers import StoppingCriteria, StoppingCriteriaList

FAST_MAX_NEW = 64     
FAST_BATCH   = 8      
CHECK_EVERY  = 8      

class StopWhenAllVerdicts(StoppingCriteria):
    """Stop the batch as soon as every row has emitted a parseable VERDICT."""
    def __init__(self, tok, plen): self.tok, self.plen, self.step = tok, plen, 0
    def __call__(self, input_ids, scores, **kw):
        self.step += 1
        if self.step % CHECK_EVERY: return False
        gen = input_ids[:, self.plen:]
        return all(_parse_verdict(t) is not None
                   for t in self.tok.batch_decode(gen, skip_special_tokens=True))

def fast_judge(df):
    rows = df.reset_index(drop=True)
    labels = np.full(len(rows), -1, int); conf = np.full(len(rows), 0.5)
    for s in range(0, len(rows), FAST_BATCH):
        batch = rows.iloc[s:s+FAST_BATCH]
        texts = [tokenizer.apply_chat_template(_messages(r), tokenize=False, add_generation_prompt=True)
                 for _, r in batch.iterrows()]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=MAX_LEN).to(model.device)
        plen = enc.input_ids.shape[1]
        sc = StoppingCriteriaList([StopWhenAllVerdicts(tokenizer, plen)])
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=FAST_MAX_NEW, do_sample=False,
                                  use_cache=True, pad_token_id=tokenizer.pad_token_id,
                                  stopping_criteria=sc)
        for i, t in enumerate(tokenizer.batch_decode(out[:, plen:], skip_special_tokens=True)):
            v = _parse_verdict(t)
            if v is not None: labels[s+i] = v; conf[s+i] = float(v)
        del enc, out
        try: torch.cuda.empty_cache()
        except Exception: pass
        print(f"  judged {s+len(batch)}/{len(rows)}", end="\r")
    print()
    return labels, conf

if LLM_OK:
    print("Judging test set (fast CoT: early-stop + greedy)...")
    tl, tc = (fast_judge(test) if JUDGE_MODE == "cot" else judge(test))
    test_pred = finalize(tl, tc, test["ground"].values)
    path_used = f"PRIMARY LLM JUDGE ({ACTIVE_MODEL}, mode={JUDGE_MODE}, fast)"
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

