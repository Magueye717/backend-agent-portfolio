# backend/ingest.py

import os
import chromadb
from chromadb.utils import embedding_functions

# Path to ChromaDB storage folder
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
DATA_PATH   = os.path.join(os.path.dirname(__file__), "data", "career.md")

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """
    Split text into overlapping chunks.
    Overlap ensures context is not lost at chunk boundaries.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap   # slide window with overlap
    return chunks

def ingest():
    # Read career Markdown file
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Use a free, local embedding model
    # "all-MiniLM-L6-v2" is fast and good for semantic search (~90 MB)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Connect to (or create) the local ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete existing collection so we can re-index cleanly
    try:
        client.delete_collection("career")
    except Exception:
        pass

    collection = client.create_collection(
        name="career",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}   # cosine similarity for semantic search
    )

    # Chunk the text and add to ChromaDB
    chunks = chunk_text(raw_text)
    print(f"Indexing {len(chunks)} chunks...")

    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    print("✅ Ingestion complete.")

if __name__ == "__main__":
    ingest()