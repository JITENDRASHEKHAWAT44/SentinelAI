import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_postgres import PGVector

from backend.database.database import engine

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")
load_dotenv(Path.cwd() / ".env")

# NVIDIA API key
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY not found in .env")

# Initialize embedding model with max_batch_size=1 to prevent NVIDIA NIM 500 batch inference errors
embeddings = NVIDIAEmbeddings(
    model="nvidia/nemotron-3.5-embed-1b",
    nvidia_api_key=NVIDIA_API_KEY,
    max_batch_size=1,
)

# Vector collection
COLLECTION_NAME = "sentinelrag_documents"

# Initialize PGVector
vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=engine,
    use_jsonb=True,
)


def add_documents(chunks, batch_size: int = 4):
    """
    Safely stores documents in PostgreSQL pgvector in manageable batches.
    Handles rate limits and API retries to prevent batch payload inference errors.
    """
    if not chunks:
        return []

    all_ids = []
    total = len(chunks)
    print(f"Ingesting {total} chunks in safe sub-batches of {batch_size}...")

    for i in range(0, total, batch_size):
        sub_batch = chunks[i : i + batch_size]
        try:
            ids = vector_store.add_documents(sub_batch)
            all_ids.extend(ids)
            print(f"  - Progress: {len(all_ids)}/{total} chunks embedded")
            time.sleep(0.15)
        except Exception as exc:
            print(f"  [WARN] Batch {i // batch_size + 1} encountered {exc}. Waiting 1s and retrying item-by-item...")
            time.sleep(1.0)
            for doc in sub_batch:
                try:
                    ids = vector_store.add_documents([doc])
                    all_ids.extend(ids)
                    time.sleep(0.2)
                except Exception as inner_exc:
                    print(f"  [ERROR] Skipping chunk due to: {inner_exc}")

    print(f"Successfully stored {len(all_ids)}/{total} chunks in pgvector!")
    return all_ids


def retrieve_documents(query: str, k: int = 5):
    results = vector_store.similarity_search(query=query, k=k)
    return results


def retrieve_documents_with_scores(query: str, k: int = 5):
    """Return (document, distance) pairs for evidence ranking in the UI."""
    return vector_store.similarity_search_with_score(query=query, k=k)
