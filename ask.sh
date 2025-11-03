#!/usr/bin/env bash
set -e
QUERY="$1"
MODEL="${2:-phi:2.7b}"
TOPK="${3:-3}"

# 1. embedding (single quoted string fed to python)
EMB_JSON=$(python3 - <<PY
import sys, json
from sentence_transformers import SentenceTransformer
q = """$QUERY"""
emb = SentenceTransformer("all-MiniLM-L6-v2").encode([q])
print(json.dumps(emb.tolist()[0]))
PY
)

# 2. search Milvus
CHUNKS=$(python3 - <<PY
import json, pymilvus as pv
pv.connections.connect(host="localhost", port="19530")
coll = pv.Collection("Precedence_collection")
coll.load()
vec = json.loads('$EMB_JSON')
res = coll.search([vec], anns_field="vector",
                  # in ask.sh change one line
                  param={"metric_type":"L2","params":{"nprobe":16}},
                  limit=$TOPK, output_fields=["text"])
chunks = [h.entity.get("text").replace('\n',' ') for hits in res for h in hits]
print(json.dumps(chunks))
PY
)

# 3. build prompt
CONTEXT=$(echo "$CHUNKS" | jq -r '.[]' | paste -sd ' ')
PROMPT="You are a legal assistant. Use only Indian case law.\n\nContext: $CONTEXT\n\nQuestion: $QUERY"

# 4. call Ollama
curl -s http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d "{
        \"model\": \"$MODEL\",
        \"prompt\": \"$PROMPT\",
        \"stream\": false,
        \"options\": {\"temperature\":0.3, \"num_predict\":512}
      }" | jq -r .response