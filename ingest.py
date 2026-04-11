# backend/ingest.py

import os
import numpy as np
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
DATA_PATH   = os.path.join(os.path.dirname(__file__), "data", "career.md")

HF_API_KEY = os.getenv("HF_API_KEY")
EMBED_MODEL = "intfloat/multilingual-e5-large"


class HuggingFaceEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.client = InferenceClient(provider="hf-inference", api_key=HF_API_KEY)

    def __call__(self, input: Documents) -> Embeddings:
        result = self.client.feature_extraction(input, model=EMBED_MODEL)
        arr = np.array(result)
        if arr.ndim == 3:
            arr = arr.mean(axis=1)
        return arr.tolist()


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def ingest():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()

    embedding_fn = HuggingFaceEmbeddingFunction()
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection("career")
    except Exception:
        pass

    collection = client.create_collection(
        name="career",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    chunks = chunk_text(raw_text)
    print(f"Indexing {len(chunks)} chunks...")

    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    print("✅ Ingestion complete.")


if __name__ == "__main__":
    ingest()