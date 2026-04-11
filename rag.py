# backend/rag.py
#lazy loading of ChromaDB client and embedding function

import os
import chromadb
from chromadb.utils import embedding_functions

# Get the absolute path to the directory containing this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct the absolute path to the chroma_db folder
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

# Lazy-loaded globals — nothing loads at startup
_embedding_fn = None
_collection = None

def get_collection():
    global _embedding_fn, _collection
    if _collection is None:
        _embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_collection(
            name="career",
            embedding_function=_embedding_fn
        )
    return _collection


def retrieve(query: str, n_results: int = 4) -> list[str]:
    """
    Find the N most relevant chunks for the query.
    ChromaDB embeds the query with the same model and does cosine search.
    """
    collection = get_collection()  # loads only on first call
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    # results["documents"] is a list-of-lists (one per query)
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