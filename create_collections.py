import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from pymilvus import connections

# ---------- Helper Class ----------
class SentenceTransformerEmbeddings:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


# ---------- PDF Extraction ----------
PDF_PATH = "./IPC.pdf"  # Replace with your PDF path
print("Extracting text from PDF...")
doc = fitz.open(PDF_PATH)
text = "\n".join([page.get_text("text") for page in doc if page.get_text("text")])
print("Text extraction completed.")

# ---------- Text Splitting ----------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.create_documents([text])
print(f"Split document into {len(chunks)} chunks.")

# ---------- Initialize Embeddings ----------
embedding_function = SentenceTransformerEmbeddings()
print("Generating embeddings...")
embeddings = embedding_function.embed_documents([chunk.page_content for chunk in chunks])

# ---------- Connect to Milvus ----------
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT, secure=False)
print("Connected to Milvus successfully!")

# ---------- Prepare Documents with Metadata ----------
docs_with_metadata = [
    Document(page_content=chunk.page_content, metadata={"source": f"page_{i}"})
    for i, chunk in enumerate(chunks)
]

# ---------- Store in Milvus ----------
COLLECTION_NAME = "IPC_collection"
vector_store = Milvus.from_documents(
    documents=docs_with_metadata,
    embedding=embedding_function,
    collection_name=COLLECTION_NAME,
    connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
    index_params={"index_type": "FLAT"}  # FLAT index ensures accuracy
)

print("Embeddings stored in Milvus. Ready for queries!")
