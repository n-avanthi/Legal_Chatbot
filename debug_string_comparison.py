#!/usr/bin/env python3
"""
Debug script to find exact differences between retrieved and expected docs
"""
import json
import re
from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer

# Setup
connections.connect(host="localhost", port="19530")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def normalize_text(text):
    """Normalize whitespace for consistent comparison"""
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def retrieve(query, top_k=3):
    coll = Collection("Precedence_collection")
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

def find_first_difference(s1, s2):
    """Find where two strings first differ"""
    for i, (c1, c2) in enumerate(zip(s1, s2)):
        if c1 != c2:
            return i, c1, c2
    # If we get here, one string is a prefix of the other
    if len(s1) != len(s2):
        return min(len(s1), len(s2)), "END", "END"
    return None, None, None

# Load gold standard
with open("gold_legal.jsonl") as f:
    gold = json.loads(f.readline())

query = gold["query"]
expected = gold["relevant_docs"][0]

print("="*100)
print(f"QUERY: {query}")
print("="*100)

# Retrieve
retrieved_docs = retrieve(query, top_k=3)
retrieved = retrieved_docs[0]

print(f"\nRETRIEVED LENGTH: {len(retrieved)}")
print(f"EXPECTED LENGTH:  {len(expected)}")
print(f"LENGTH DIFFERENCE: {len(retrieved) - len(expected)}")

print("\n" + "="*100)
print("CHECKING RAW STRINGS")
print("="*100)

pos, c1, c2 = find_first_difference(retrieved, expected)
if pos is not None:
    print(f"❌ First difference at position {pos}")
    print(f"   Retrieved char: {repr(c1)}")
    print(f"   Expected char:  {repr(c2)}")
    print(f"\n   Context (retrieved): ...{repr(retrieved[max(0,pos-50):pos+50])}...")
    print(f"   Context (expected):  ...{repr(expected[max(0,pos-50):pos+50])}...")
else:
    print("✓ Strings are IDENTICAL")

print("\n" + "="*100)
print("CHECKING NORMALIZED STRINGS")
print("="*100)

retrieved_norm = normalize_text(retrieved)
expected_norm = normalize_text(expected)

print(f"NORMALIZED RETRIEVED LENGTH: {len(retrieved_norm)}")
print(f"NORMALIZED EXPECTED LENGTH:  {len(expected_norm)}")
print(f"LENGTH DIFFERENCE: {len(retrieved_norm) - len(expected_norm)}")

pos, c1, c2 = find_first_difference(retrieved_norm, expected_norm)
if pos is not None:
    print(f"❌ First difference at position {pos}")
    print(f"   Retrieved char: {repr(c1)}")
    print(f"   Expected char:  {repr(c2)}")
    print(f"\n   Context (retrieved): ...{repr(retrieved_norm[max(0,pos-50):pos+50])}...")
    print(f"   Context (expected):  ...{repr(expected_norm[max(0,pos-50):pos+50])}...")
else:
    print("✓ Strings are IDENTICAL after normalization")

print("\n" + "="*100)
print("SHOWING LAST 200 CHARS OF EACH")
print("="*100)
print("RETRIEVED (last 200):")
print(repr(retrieved[-200:]))
print("\nEXPECTED (last 200):")
print(repr(expected[-200:]))

print("\n" + "="*100)
print("CHECKING ALL 3 RETRIEVED DOCS")
print("="*100)
for i, doc in enumerate(retrieved_docs):
    norm_doc = normalize_text(doc)
    matches = norm_doc == expected_norm
    print(f"Doc {i+1}: Length={len(doc)}, Normalized Length={len(norm_doc)}, MATCHES={matches}")
    if matches:
        print(f"   ✓ THIS ONE MATCHES!")