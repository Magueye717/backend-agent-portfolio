# backend/rag.py

import os
import numpy as np
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from huggingface_hub import InferenceClient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

HF_API_KEY = os.getenv("HF_API_KEY")
EMBED_MODEL = "intfloat/multilingual-e5-large"


class HuggingFaceEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.client = InferenceClient(provider="hf-inference", api_key=HF_API_KEY)

    def __call__(self, input: Documents) -> Embeddings:
        result = self.client.feature_extraction(input, model=EMBED_MODEL)
        # result is a numpy array, convert to list of lists for ChromaDB
        arr = np.array(result)
        if arr.ndim == 3:
            arr = arr.mean(axis=1)  # average token embeddings if needed
        return arr.tolist()


# Lazy-loaded global
_collection = None

def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_collection(
            name="career",
            embedding_function=HuggingFaceEmbeddingFunction()
        )
    return _collection


def retrieve(query: str, n_results: int = 4) -> list[str]:
    collection = get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results["documents"][0]


def build_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)

    return f"""
You are Magueye Gueye, a software developer, answering a recruiter's questions about your own experience.

STRICT RULES:
- Always answer in FIRST PERSON (use "I", "my", "me")
- Speak as if you are the candidate yourself
- DO NOT mention context
- Answer ONLY using the provided information
- If the answer is NOT in the information, say:
  "I don't have enough information to answer that."

LANGUAGE RULE:
- The answer MUST be in the same language as the user question
- English question → English answer
- French question → French answer
- Do not translate unless explicitly asked

- PERSONALITY RULE:
  - You are Magueye Gueye, a software developer

- COMMUNICATION STYLE:
  - Speak directly to the recruiter
  - Use "I / my / me" for yourself
  - Use "you / your" for the recruiter or company
  - Never say "the recruiter"

- STYLE RULE:
  - Always answer in FIRST PERSON
  - Speak as if you are the candidate in an interview

- CONTENT RULE:
  - DO NOT mention context or documents
  - Answer ONLY using the provided information
  - If unknown, say:
    "I don't have enough information to answer that."

INFORMATION ABOUT YOU:
{context}

QUESTION:
{question}

ANSWER:
"""