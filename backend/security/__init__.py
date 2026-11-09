"""
SentinelRAG Security & NeMo Guardrails Package.
Provides multi-layer security for RAG systems:
- Input Rails (Prompt injection, jailbreak, harmful content prevention)
- Untrusted Context Sanitization (Indirect injection mitigation)
- Output Rails (Secret/instruction leakage prevention)
- Fact-Checking Rails (Grounded document evidence validation)
"""

from backend.security.rag_guard import (
    SentinelRAGGuard,
    get_guardrails,
    guarded_answer,
)
from backend.security.actions import (
    sanitize_context,
    mask_sensitive_pii,
    check_context_relevance,
)

__all__ = [
    "SentinelRAGGuard",
    "get_guardrails",
    "guarded_answer",
    "sanitize_context",
    "mask_sensitive_pii",
    "check_context_relevance",
]
