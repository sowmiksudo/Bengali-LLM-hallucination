import re

with open('build_improved_notebook.py', 'r') as f:
    content = f.read()

# 1. Update SYS_COT
old_sys_cot = """SYS_COT = (
    "You are a meticulous Bengali fact-checking expert. Evaluate the RESPONSE for these hallucination types:\n"
    "1. Factual Inconsistency: Do dates, names, numbers, or entities contradict the CONTEXT or world knowledge?\n"
    "2. Reasoning/Math Errors: Is the logic, arithmetic, or calculation flawed?\n"
    "3. Idiomatic Errors (বাগধারা): Does the response give a LITERAL meaning of a Bengali idiom instead of its figurative meaning?\n"
    "4. Code-Mixing Errors: Are English/Bengali translations or transliterations incorrect?\n"
    "5. World Knowledge: Is the response factually wrong based on common knowledge?\n\n"
    "Reason step-by-step in 2-3 sentences, then output exactly: 'VERDICT: 1' (faithful) or 'VERDICT: 0' (hallucinated)."
)"""

new_sys_cot = """SYS_COT = (
    "You are a meticulous Bengali fact-checking expert. Evaluate the RESPONSE for these hallucination types:\n"
    "1. Factual Inconsistency: Do dates, names, numbers, or entities contradict the CONTEXT or world knowledge?\n"
    "2. Reasoning/Math Errors: Calculate step-by-step for mathematical questions before determining the verdict. Is the logic, arithmetic, or calculation flawed?\n"
    "3. Temporal Errors: Extract and compare dates/years carefully for temporal questions.\n"
    "4. Idiomatic Errors (বাগধারা): Does the response give a LITERAL meaning of a Bengali idiom instead of its figurative meaning?\n"
    "5. Code-Mixing Errors: Are English/Bengali translations or transliterations incorrect?\n"
    "6. World Knowledge: Is the response factually wrong based on common knowledge?\n\n"
    "Reason step-by-step in 2-3 sentences, then output exactly: 'VERDICT: 1' (faithful) or 'VERDICT: 0' (hallucinated)."
)"""

content = content.replace(old_sys_cot, new_sys_cot)

# 2. Add Cell 2 injection
cell2_injection = r'''
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
    return \"\"\"""

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
    return \"\"\"

new_cell2 = orig_cell2.replace(old_rag_block, new_rag_block)
set_cell(2, new_cell2.split('\\n'))
'''

if '# CELL 2' not in content:
    parts = content.split('# ============================================================\n# CELL 6')
    content = parts[0] + cell2_injection + '\n# ============================================================\n# CELL 6' + parts[1]

with open('build_improved_notebook.py', 'w') as f:
    f.write(content)
