#!/usr/bin/env python3
"""
evaluate_legal.py
Retrieval + Generation benchmark for ALL local Ollama models
on Precedence_collection only.
"""
import json, csv, re, argparse, time
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
from sklearn.metrics import ndcg_score
from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer
import httpx

# ---------------- config -----------------
COLLECTION      = "Precedence_collection"
K               = 3
TEXT_SLICE      = None
LOCAL_MODELS    = {
    "llama3.2:1b": "Llama-3.2-1B",
    "phi:2.7b"   : "Phi-2.7B",
    "gemma:1b"   : "Gemma-3-1B",
    "qwen3:1.7b" : "Qwen3-1.7B",
    "mistral:7b" : "Mistral-7B",
}
OLLAMA_URL      = "http://localhost:11434/api/generate"
CITATION_RE     = re.compile(r"\bAIR\s+\d{4}\b|\bSCC\s+\(\d+\)\s+\d+|\b\d{4}\s+SCC\s+\d+", re.I)
# -----------------------------------------
connections.connect(host="localhost", port="19530")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- NEW: String normalization ----------
def normalize_text(text):
    """Normalize whitespace for consistent comparison"""
    # Replace multiple spaces with single space
    # Replace multiple newlines with single newline
    # Strip leading/trailing whitespace
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

# ---------- helpers ----------
def doc_matches(retrieved_doc, expected_doc):
    """Check if documents match (handles truncated expected docs)"""
    ret_norm = normalize_text(retrieved_doc)
    exp_norm = normalize_text(expected_doc)
    
    # If expected is shorter, check if retrieved starts with expected
    if len(exp_norm) < len(ret_norm):
        return ret_norm.startswith(exp_norm)
    # If same length or expected is longer, require exact match
    return ret_norm == exp_norm

def precision_at_k(ret, rel, k):
    """Calculate precision allowing for truncated gold standard docs"""
    matches = 0
    for ret_doc in ret[:k]:
        for rel_doc in rel:
            if doc_matches(ret_doc, rel_doc):
                matches += 1
                break
    return matches / k

def recall_at_k(ret, rel, k):
    """Calculate recall allowing for truncated gold standard docs"""
    if not rel:
        return 0.0
    matches = 0
    for rel_doc in rel:
        for ret_doc in ret[:k]:
            if doc_matches(ret_doc, rel_doc):
                matches += 1
                break
    return matches / len(rel)

def mrr(retrieved, relevant):
    """Calculate MRR allowing for truncated gold standard docs"""
    for rank, ret_doc in enumerate(retrieved, 1):
        for rel_doc in relevant:
            if doc_matches(ret_doc, rel_doc):
                return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved, relevant, k):
    """Calculate NDCG allowing for truncated gold standard docs"""
    rel_vec = []
    for ret_doc in retrieved[:k]:
        is_relevant = any(doc_matches(ret_doc, rel_doc) for rel_doc in relevant)
        rel_vec.append(1 if is_relevant else 0)
    y_true, y_score = [rel_vec], [rel_vec]
    return ndcg_score(y_true, y_score, k=k)

def bleu(candidate, reference):
    return sentence_bleu([reference.split()], candidate.split(),
                         smoothing_function=SmoothingFunction().method1)

def bert_f1(cand, ref):
    P, R, F1 = bert_score([cand], [ref], lang="en", verbose=False)
    return F1.item()

def citation_accuracy(gen, gold_cits):
    gen_cits = set(CITATION_RE.findall(gen))
    gold_set = set(gold_cits)
    return len(gen_cits & gold_set) / len(gold_set) if gold_set else 1.0

def ask_ollama(model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 512}
    }
    try:
        r = httpx.post(OLLAMA_URL, json=payload, timeout=180)
        r.raise_for_status()
        return r.json()["response"].strip()
    except Exception as e:
        return f"[{model}] error – {e}"

# ---------- retrieval ----------
def retrieve(query, top_k=K):
    coll = Collection(COLLECTION)
    coll.load()
    emb = embedder.encode([query])[0].tolist()
    res = coll.search(
        data=[emb],
        anns_field="vector",
        param={"metric_type": "L2", "params": {"nprobe": 16}},
        limit=top_k,
        output_fields=["text"]
    )
    return [hit.entity.get("text") for hits in res for hit in hits]

# ---------- main ----------
def evaluate(gold_jsonl, out_csv):
    gold = [json.loads(l) for l in open(gold_jsonl)]
    rows = []

    for g in tqdm(gold, desc="Eval"):
        q          = g["query"]
        ref_docs   = g.get("relevant_docs", [])
        ref_ans    = g.get("answer", "")
        ref_cits   = g.get("relevant_citations", [])

        retrieved = retrieve(q)

        # ---- debug output ----
        print("\n" + "="*80)
        print("QUERY:", q)
        print("\n--- RETRIEVED[0] length:", len(retrieved[0]) if retrieved else 0)
        print("--- EXPECTED[0] length:", len(ref_docs[0]) if ref_docs else 0)
        
        # Check if they match
        if retrieved and ref_docs:
            match = doc_matches(retrieved[0], ref_docs[0])
            print(f">>> DOCUMENT MATCH: {match}")
        print("="*80 + "\n")
        # ----------------------

        p   = precision_at_k(retrieved, ref_docs, K)
        r   = recall_at_k(retrieved, ref_docs, K)
        m   = mrr(retrieved, ref_docs)
        nd  = ndcg_at_k(retrieved, ref_docs, K)

        system = "You are a legal assistant. Use only Indian case law."
        user   = f"Context:\n" + "\n\n".join(retrieved) + f"\n\nQuestion: {q}"
        full_prompt = f"{system}\n\n{user}"

        for model_id, model_name in LOCAL_MODELS.items():
            generated = ask_ollama(model_id, full_prompt)

            ble = bleu(generated, ref_ans)
            bfi = bert_f1(generated, ref_ans)
            cit = citation_accuracy(generated, ref_cits)

            rows.append({
                "query": q,
                "model": model_name,
                "precision@3": p,
                "recall@3": r,
                "mrr": m,
                "ndcg@3": nd,
                "bleu": ble,
                "bert_f1": bfi,
                # "citation_acc": cit
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print("\n==========  AGGREGATE over {} queries  ==========".format(len(gold)))
    for model in LOCAL_MODELS.values():
        sub = df[df.model == model]
        print(sub[['precision@3','recall@3','mrr','ndcg@3','bleu','bert_f1']].mean().to_frame(model).T, "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, help="JSONL with query, relevant_docs, answer, relevant_citations")
    parser.add_argument("--out", default="precedence_model_benchmark.csv")
    args = parser.parse_args()
    evaluate(args.gold, args.out)