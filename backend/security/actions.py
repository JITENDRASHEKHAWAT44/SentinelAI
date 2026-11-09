import re
import logging
from typing import Optional, List, Dict, Any
from nemoguardrails.actions import action

logger = logging.getLogger(__name__)

# Patterns for indirect prompt injection commonly embedded in scraped/retrieved documents
INDIRECT_INJECTION_PATTERNS = [
    re.compile(r"(?i)(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|prompts|commands)"),
    re.compile(r"(?i)new\s+instruction\s*:\s*"),
    re.compile(r"(?i)system\s*prompt\s*:\s*"),
    re.compile(r"(?i)\[system\s*(?:message|instruction)\]"),
    re.compile(r"(?i)<(?:system|developer|admin)>"),
    re.compile(r"(?i)!\[.*?\]\(https?://[^\s)]+\)"),  # Markdown image exfiltration
    re.compile(r"(?i)you\s+must\s+now\s+(?:act\s+as|say|output|respond)"),
]

# PII & Secret Patterns
PII_PATTERNS = [
    # Credit Cards (Visa, MC, Amex)
    (re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"), "[REDACTED_CARD]"),
    # US SSN
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    # Generic API Keys / Secrets (e.g. nvapi-..., sk-...)
    (re.compile(r"\b(nvapi-[A-Za-z0-9_-]{30,}|sk-[A-Za-z0-9_-]{20,})\b"), "[REDACTED_API_KEY]"),
]


@action(name="sanitize_context")
def sanitize_context(context: str) -> str:
    """
    Sanitizes untrusted document context before passing to LLM.
    Disarms indirect prompt injections embedded inside retrieved document chunks.
    """
    if not context:
        return ""

    sanitized = context
    for pattern in INDIRECT_INJECTION_PATTERNS:
        sanitized = pattern.sub("[REMOVED_SUSPICIOUS_DIRECTIVE]", sanitized)

    return sanitized


@action(name="mask_sensitive_pii")
def mask_sensitive_pii(text: str) -> str:
    """
    Masks sensitive Personally Identifiable Information and API credentials.
    """
    if not text:
        return ""

    masked = text
    for pattern, replacement in PII_PATTERNS:
        masked = pattern.sub(replacement, masked)

    return masked


@action(name="check_context_relevance")
def check_context_relevance(query: str, context: str) -> bool:
    """
    Fast keyword-overlap heuristic to assess if retrieved context is minimally relevant.
    """
    if not query or not context:
        return False

    query_words = set(re.findall(r"\w+", query.lower()))
    # Remove common stop words
    stop_words = {"what", "when", "where", "which", "who", "whom", "this", "that", "these", "those", "is", "are", "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of"}
    meaningful_words = query_words - stop_words

    if not meaningful_words:
        return True

    context_lower = context.lower()
    matches = sum(1 for word in meaningful_words if word in context_lower)
    return (matches / len(meaningful_words)) >= 0.25
