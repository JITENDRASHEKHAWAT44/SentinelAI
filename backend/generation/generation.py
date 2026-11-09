import os
import sys
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, Union

# Handle Windows Application Control policy blocking native C xxhash module
if "xxhash" not in sys.modules:
    try:
        import xxhash  # noqa: F401
    except ImportError:
        class _FallbackXXHash:
            @staticmethod
            def xxh3_128(data: bytes):
                class _Digest:
                    def digest(self):
                        return hashlib.md5(data).digest()
                return _Digest()
        sys.modules["xxhash"] = _FallbackXXHash()

from dotenv import load_dotenv

# Load environment variables (from project root and backend)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR / "backend" / ".env")

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY not found in environment")

from backend.retrievel.retriever import retrieve_context
from backend.security import get_guardrails, SentinelRAGGuard


def generate_answer(
    query: str,
    k: int = 5,
    return_details: bool = False
) -> Union[str, Dict[str, Any]]:
    """
    Generate an answer for the user query using retrieved document context
    protected by NeMo Guardrails (Input rails, Context sanitization, Output rails, and Fact checking).

    Args:
        query: User question or prompt.
        k: Number of chunks to retrieve from vector store.
        return_details: If True, returns full dict with security metadata, else returns answer string.

    Returns:
        Clean answer string or dictionary with security metrics and answer.
    """
    # 1. Retrieve relevant documents from vector store
    documents = retrieve_context(query, k=k)

    context = ""
    if documents:
        context = "\n\n".join(doc["content"] for doc in documents)

    # 2. Access the Guardrails engine
    guard: SentinelRAGGuard = get_guardrails()

    # 3. Generate guarded response
    # NeMo Guardrails executes:
    # - Input Rail: Blocks prompt injections, jailbreaks, and system prompt exfiltration.
    # - Sanitizer: Disarms indirect injections embedded in document context.
    # - LLM Generation: Answers strictly using the provided context.
    # - Output Rail: Verifies no system prompt or confidential leaks in the output.
    # - Fact-Checking Rail: Validates answer is grounded in the retrieved chunks.
    result = guard.generate(query=query, context=context, sanitize=True)

    if return_details:
        result["retrieved_chunks"] = documents
        return result

    return result["answer"]


async def generate_answer_async(
    query: str,
    k: int = 5,
    return_details: bool = False,
    sanitize: bool = True,
) -> Union[str, Dict[str, Any]]:
    """
    Asynchronous version of generate_answer for FastAPI backends and async pipelines.
    """
    documents = retrieve_context(query, k=k)

    context = ""
    if documents:
        context = "\n\n".join(doc["content"] for doc in documents)

    guard: SentinelRAGGuard = get_guardrails()
    result = await guard.generate_async(query=query, context=context, sanitize=sanitize)

    if return_details:
        result["retrieved_chunks"] = documents
        return result

    return result["answer"]


if __name__ == "__main__":
    test_query = "who performed really well and it's scores?"
    print(f"Testing SentinelRAG with Guardrails for query: '{test_query}'\n")
    response = generate_answer(test_query, return_details=True)
    print("SentinelRAG Answer:\n")
    print(response["answer"])
    print("\nSecurity Details:")
    print(f"- Blocked: {response['is_blocked']}")
    print(f"- Rail Triggered: {response['rail_triggered']}")
