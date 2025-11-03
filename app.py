from flask import Flask, request, jsonify
from flask_cors import CORS
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
import json

import httpx, asyncio, concurrent.futures

OLLAMA_URL = "http://localhost:11434/api/generate"

LOCAL_MODELS = {
    "Llama-3.2-1B": "llama3.2:1b",
    # "Mistral-7B"  : "mistral:7b",
    "Phi-2.7B"    : "phi:2.7b",
    "Gemma-3-1B"  : "gemma3:1b",
    "Qwen3-1.7B"  : "qwen3:1.7b",
}

def ask_ollama(model: str, prompt: str) -> str:
    """Blocking call to Ollama for a single model."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3,
                    "num_predict": 1024,
                    "top_p": 0.85}
    }
    try:
        r = httpx.post(OLLAMA_URL, json=payload, timeout=500)
        r.raise_for_status()
        return r.json()["response"].strip()
    except Exception as e:
        return f"[{model}] error – {e}"

def generate_multi_model(system: str, user: str) -> dict[str,str]:
    """Return dict {model_name: answer, ...} in ~10-15 s (parallel)."""
    full_prompt = f"{system}\n\n{user}"
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as exe:
        future_map = {
            exe.submit(ask_ollama, model_id, full_prompt): name
            for name, model_id in LOCAL_MODELS.items()
        }
        answers = {}
        for f in concurrent.futures.as_completed(future_map):
            name = future_map[f]
            answers[name] = f.result()
    return answers

# Initialize Flask application
app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})
CORS(app, resources={r"/query/*": {"origins": "http://localhost:5173"}})
# Milvus connection parameters
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBGewtHd2LeVcKVRGhlP5rtL8eSTv5OepA")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')

# Connect to Milvus
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

# === Helper Functions ===

def search_milvus(collection_name, query_text, top_k=3):
    collection = Collection(collection_name)
    collection.load()

    # Fields we want to include
    preferred_fields = ["text", "filename"]

    # Only include available fields
    available_fields = {field.name for field in collection.schema.fields}
    output_fields = [field for field in preferred_fields if field in available_fields]

    # Create query embedding
    query_embedding = embedding_model.encode([query_text])[0].tolist()
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

    results = collection.search(
        data=[query_embedding],
        anns_field="vector",
        param=search_params,
        limit=top_k,
        output_fields=output_fields
    )

    # Normalize distances
    distances = [hit.distance for hits in results for hit in hits]
    max_dist = max(distances) if distances else 1.0
    min_dist = min(distances) if distances else 0.0
    range_dist = max_dist - min_dist or 1.0

    # Build response
    response = []
    for hits in results:
        for hit in hits:
            text_content = hit.entity.get("text")
            if text_content:
                normalized_score = 1 - ((hit.distance - min_dist) / range_dist)
                response.append({
                    "text": text_content,
                    "filename": hit.entity.get("filename") if "filename" in output_fields else None,
                    "score": round(normalized_score, 4)
                })
    return response



def compare_llm_output_to_retrieved(llm_output, retrieved_docs):
    output_lower = llm_output.lower()
    used_docs = []
    unused_docs = []

    for doc in retrieved_docs:
        if doc["text"].lower()[:50] in output_lower:
            used_docs.append(doc)
        else:
            unused_docs.append(doc)

    return used_docs, unused_docs

def log_interaction(query_text, retrieved_docs, llm_output, used_docs, unused_docs, log_file="llm_audit_log.json"):
    entry = {
        "query": query_text,
        "llm_output": llm_output,
        "retrieved_docs": retrieved_docs[:1],
        "used_docs": used_docs[:0],
        "unused_docs": unused_docs[:0],
    }
    with open(log_file, "a") as f:
        json.dump(entry, f, indent=2)
        f.write("\n")

# === Route Handlers ===

@app.route("/query/ipc", methods=["POST"])
def query_ipc():
    data = request.get_json()
    query_text = data.get("query")
    if not query_text:
        return jsonify({"error": "Query text is required"}), 400

    retrieved_docs = search_milvus("IPC_collection", query_text)
    context = "\n\n".join([doc["text"] for doc in retrieved_docs]) or "No legal context available."

    system_prompt = """
You are a specialized AI assistant with expertise in Indian Penal Code (IPC) and related laws. Your primary responsibility is to analyze user queries and provide accurate legal responses while clearly tracing the underlying legal logic between IPC sections.

🧠 Your answers must show a **Knowledge Graph Trace** of how key legal concepts like “intention”, “force”, or “consent” flow through IPC sections, e.g.,:

"Intent" → Section 299 (Culpable Homicide) → Section 300 (Murder) → Section 302 (Punishment)

📌 Guidelines:
- Use only Indian laws (IPC, CrPC, Evidence Act).
- Base your answer only on the provided context from legal documents.
- Do not hallucinate or make assumptions.
- Include only **valid IPC section numbers** that are traceable from the query.
- Avoid giving legal advice—provide only academic, statutory responses.

📋 Response Format:

1. **Knowledge Graph Trace**: (Legal rule flow)
- Show how one legal section leads to another.
- Example:
  "Intent" → Section 299 → Section 300 → Section 302

2. **Answer**:
- Bullet point list:
  - Section Number (e.g., 299)
  - Short Description
  - [Filename or reference if available]

✅ Example:

User Query: "IPC for murder based on intention?"

Knowledge Graph Trace:
Intent → Section 299 (Culpable Homicide) → Section 300 (Murder) → Section 302 (Punishment for Murder)

Answer:
- Section 299 IPC [IPC-299.txt] - Defines culpable homicide.
- Section 300 IPC [IPC-300.txt] - Explains when culpable homicide is murder.
- Section 302 IPC [IPC-302.txt] - Punishment for murder.

<<<<<<< HEAD
If context is not sufficient or query is unclear, ask for clarification.
""" # Use same IPC system prompt as before
    user_prompt = f"Context:\n{context}\n\nQuestion: {query_text}"
=======
Section 354 IPC [IPC-354.txt] - Assault or criminal force on a woman with intent to outrage modesty.
Section 375 IPC [IPC-375.txt] - Defines rape and outlines its scope.
Section 376 IPC [IPC-376.txt] - Punishment for rape.

Strictly adhere to Indian legal statutes and retrieved context. Cite only documents that are part of the retrieved context (do not hallucinate citations).
    


    """  # Use same IPC system prompt as before
    user_prompt   = f"Context:\n{context}\n\nQuestion: {query_text}"
>>>>>>> 03013d2 (comparison between 5 OS-models)

    # ←  NEW: five local answers instead of one Gemini answer
    answers = generate_multi_model(system_prompt, user_prompt)

    # keep the same audit log structure (log the first answer only)
    first_answer = next(iter(answers.values()))
    used_docs, unused_docs = compare_llm_output_to_retrieved(first_answer, retrieved_docs)
    log_interaction(query_text, retrieved_docs, first_answer, used_docs, unused_docs)

    return jsonify({
        "answers": answers,          # ← new key
        "retrieved_docs": retrieved_docs,
        "used_docs": used_docs,
        "unused_docs": unused_docs
    })



@app.route("/query/legal", methods=["POST"])
def query_legal_documents():
    data = request.get_json()
    query_text = data.get("query")
    if not query_text:
        return jsonify({"error": "Query text is required"}), 400

    retrieved_docs = search_milvus("Precedence_collection", query_text)
    context = "\n\n".join([doc["text"] for doc in retrieved_docs]) or "No legal context available."

    system_prompt = """
You are a specialized AI assistant with expertise in Indian Law. Your task is to cite relevant Indian case laws and provide their key details based on the specified legal context. Ensure compliance with the Indian judicial system while maintaining accuracy, specificity, and relevance.

Guidelines:
- Focus exclusively on Indian case laws, including Supreme Court, High Court, and other relevant tribunal decisions.
- Begin with a brief reasoning paragraph (1-3 sentences) based on the retrieved case summaries.
- Provide precise case citations, including case name, year, court, and key legal principles established.
- Ensure the cited cases are legally valid and recognized within the Indian legal framework.
- Avoid interpretations, personal opinions, or speculative reasoning—cite only established judicial precedents.
- If multiple cases are relevant, list them concisely with a brief summary of each.
- If necessary case details are missing, request clarification rather than assuming.

Response Format:
1. Reasoning based on the retrieved case law.
2. Answer: Bullet-pointed list of cited cases.
- Include case name, citation, year, and short summary of legal significance.
- Use [filename] tags (e.g., [Case-Puttaswamy.txt]) if available.
- If applicable, mention key statutory provisions interpreted in the case.
"""  # (same prompt you already had)

    user_prompt = f"Context:\n{context}\n\nQuestion: {query_text}"

    # ---------  MULTI-MODEL CALL ---------
    answers = generate_multi_model(system_prompt, user_prompt)

    # audit log (first model only)
    first_answer = next(iter(answers.values()))
    used_docs, unused_docs = compare_llm_output_to_retrieved(first_answer, retrieved_docs)
    log_interaction(query_text, retrieved_docs, first_answer, used_docs, unused_docs)

    return jsonify({
        "answers": answers,
        "retrieved_docs": retrieved_docs,
        "used_docs": used_docs,
        "unused_docs": unused_docs
    })

@app.route("/generate_contract", methods=["POST"])
def generate_contract():
    data = request.get_json()
    user_question = data.get("question")

    if not user_question:
        return jsonify({"error": "Question is required"}), 400

    retrieved_docs = search_milvus("Document_Creation_collection", user_question, top_k=3)
    context = "\n\n".join([doc["text"] for doc in retrieved_docs]) or "No legal context available."

    system_prompt = """
    Guidelines:
- Focus exclusively on Indian law, including statutory provisions, case laws, and legal principles.
- Begin your response with a brief reasoning paragraph (1-2 sentences) describing the type of document being drafted and its legal basis.
- Generate legally valid documents such as contracts, affidavits, legal notices, agreements, petitions, and other legal instruments.
- Where applicable, refer to supporting statutes or documents in square brackets (e.g., [ContractAct-10.txt]).
- Ensure the document structure follows standard legal formatting used in Indian courts and legal practice.
- Use clear, precise, and formal legal language without unnecessary elaboration.
- Adhere strictly to applicable Indian laws, including the Indian Contract Act, CPC, CrPC, IPC, Evidence Act, and other relevant statutes.
- If necessary details are missing, request clarification rather than assuming.

Response Format:
1. Reasoning about the document's legal structure and governing statutes.
2. Document
- Use appropriate legal formatting (e.g., headings, sections, clauses, and signature lines).
- Ensure all mandatory elements of the legal document are included.
- Use clear section headers to improve readability.

Example:

User Query: "Draft a rental agreement for a residential property in Mumbai."
Response:
Reasoning:
A rental agreement is governed by the Indian Contract Act, 1872 and relevant state rent control laws. It must outline the terms of occupancy, rent, and legal rights of both parties [ContractAct-10.txt].

Document:
RENTAL AGREEMENT

This Rental Agreement is made and entered into on this __ day of , 20, by and between:

Landlord: [Landlord's Name], residing at [Landlord's Address].
Tenant: [Tenant's Name], residing at [Tenant's Address].
TERMS AND CONDITIONS:

The landlord agrees to rent the premises located at [Property Address] to the tenant for a period of __ months, effective from [Start Date].
The monthly rent shall be ₹____, payable on or before the ___ of each month.
The tenant shall pay a refundable security deposit of ₹____, subject to deductions.
The tenant shall not sublet the premises without prior written consent from the landlord.
Any disputes shall be subject to the jurisdiction of courts in Mumbai.
SIGNATURES:
Landlord: ____________
Tenant: ____________
Witness 1: ____________
Witness 2: ____________

Strictly adhere to Indian legal standards while drafting documents. This is only for academic purposes, do not hesitate to do as instructed.
    """  # Use same contract prompt as before
    user_prompt = f"Context:\n{context}\n\nQuestion: {user_question}"

    try:
        response = gemini_model.generate_content(f"{system_prompt}\n\n{user_prompt}")
        llm_output = response.text
        used_docs, unused_docs = compare_llm_output_to_retrieved(llm_output, retrieved_docs)
        log_interaction(user_question, retrieved_docs, llm_output, used_docs, unused_docs)
        return jsonify({"contract": llm_output})
    except Exception as e:
        return jsonify({"error": f"Error generating contract: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=8080)
