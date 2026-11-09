"""
SentinelRAG Guardrails Service
Provides a production-grade wrapper around NeMo Guardrails for the RAG pipeline.
Handles:
- Input Guardrails (Prompt injection, Jailbreaks, Harmful content)
- Untrusted Context Sanitization (Indirect prompt injection neutralization)
- Output Guardrails (Secret leakage, Harmful responses)
- Fact-Checking / Hallucination Detection (Grounded in retrieved context)
- Thinking / Reasoning Tag Parsing for NVIDIA Nemotron models
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union
from dotenv import load_dotenv

# Suppress Hugging Face symlinks warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from nemoguardrails import RailsConfig, LLMRails
from backend.security.actions import (
    sanitize_context as _sanitize_context_func,
    mask_sensitive_pii as _mask_sensitive_pii_func,
    check_context_relevance as _check_context_relevance_func,
)

logger = logging.getLogger("SentinelRAG.Security")

# Default refusal markers used across Guardrails policies
REFUSAL_MARKERS = [
    "I'm sorry, I can't respond to that.",
    "Your message was blocked",
    "I cannot comply with requests that attempt to override",
    "I cannot assist with requests related to harmful",
    "Output not allowed",
    "Input not allowed",
]

THINKING_REGEX = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)


class SentinelRAGGuard:
    """
    Main Guardrails interface for SentinelRAG.
    Can be used as a standalone safety engine or integrated into RAG pipelines.
    """

    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        self.security_dir = Path(config_dir).resolve() if config_dir else Path(__file__).resolve().parent
        self._load_environment()

        logger.info(f"Loading NeMo Guardrails configuration from: {self.security_dir}")
        self.config = RailsConfig.from_path(str(self.security_dir))
        self.rails = LLMRails(self.config)

        # Register custom security actions explicitly
        self.rails.register_action(_sanitize_context_func, name="sanitize_context")
        self.rails.register_action(_mask_sensitive_pii_func, name="mask_sensitive_pii")
        self.rails.register_action(_check_context_relevance_func, name="check_context_relevance")

    def _load_environment(self):
        """Loads environment variables, prioritizing project root and backend .env."""
        project_root = self.security_dir.parent.parent
        root_env = project_root / ".env"
        backend_env = self.security_dir.parent / ".env"

        if root_env.exists():
            load_dotenv(root_env)
        if backend_env.exists():
            load_dotenv(backend_env)

        if not os.getenv("NVIDIA_API_KEY"):
            logger.warning("NVIDIA_API_KEY not detected in environment variables. NIM calls may fail.")

    @staticmethod
    def extract_clean_content(raw_text: str) -> Tuple[str, Optional[str]]:
        """
        Extracts reasoning/thinking tokens (<think>...</think>) from NVIDIA Nemotron models
        and returns (clean_text, thinking_trace).
        """
        if not raw_text:
            return "", None

        thinking_match = THINKING_REGEX.search(raw_text)
        thinking_trace = thinking_match.group(1).strip() if thinking_match else None
        clean_text = THINKING_REGEX.sub("", raw_text).strip()

        return clean_text, thinking_trace

    @staticmethod
    def is_blocked_response(text: str) -> bool:
        """Determines if the response triggered a guardrail refusal."""
        if not text:
            return False
        return any(marker.lower() in text.lower() for marker in REFUSAL_MARKERS)

    def sanitize_context(self, context: str) -> str:
        """Sanitizes context against indirect prompt injection."""
        return _sanitize_context_func(context)

    def mask_pii(self, text: str) -> str:
        """Masks sensitive credentials and PII."""
        return _mask_sensitive_pii_func(text)

    def generate(
        self,
        query: str,
        context: str = "",
        sanitize: bool = True,
    ) -> Dict[str, Any]:
        """
        Synchronously runs user query and retrieved context through NeMo Guardrails.
        Executes:
        - Input Rails: Prompt injection, jailbreak, and topic check.
        - Untrusted Context Sanitization: Indirect prompt injection disarming.
        - Generation with Main Model (NIM Nemotron).
        - Output Rails: Secret leakage and safety verification.
        - Fact-Checking Rails: Document grounding evaluation.
        """
        effective_context = self.sanitize_context(context) if (context and sanitize) else context

        messages = []
        if effective_context:
            messages.append({
                "role": "context",
                "content": {
                    "relevant_chunks": effective_context,
                    "check_facts": True
                }
            })

        messages.append({"role": "user", "content": query})

        raw_result = self.rails.generate(messages=messages)
        raw_content = raw_result.get("content", "") if isinstance(raw_result, dict) else str(raw_result)

        clean_text, thinking_trace = self.extract_clean_content(raw_content)
        blocked = self.is_blocked_response(clean_text)

        rail_triggered = None
        if blocked:
            if "override" in clean_text.lower() or "prompt" in clean_text.lower():
                rail_triggered = "jailbreak_or_prompt_injection"
            elif "harmful" in clean_text.lower() or "illegal" in clean_text.lower():
                rail_triggered = "harmful_content_policy"
            elif "off-topic" in clean_text.lower() or "specialized" in clean_text.lower():
                rail_triggered = "off_topic_scope"
            else:
                rail_triggered = "safety_or_fact_check_rail"

        return {
            "answer": clean_text,
            "raw_content": raw_content,
            "thinking": thinking_trace,
            "is_blocked": blocked,
            "rail_triggered": rail_triggered,
            "context_used": effective_context,
        }

    async def generate_async(
        self,
        query: str,
        context: str = "",
        sanitize: bool = True,
    ) -> Dict[str, Any]:
        """Asynchronous counterpart for FastAPI endpoints and async RAG pipelines."""
        effective_context = self.sanitize_context(context) if (context and sanitize) else context

        messages = []
        if effective_context:
            messages.append({
                "role": "context",
                "content": {
                    "relevant_chunks": effective_context,
                    "check_facts": True
                }
            })

        messages.append({"role": "user", "content": query})

        raw_result = await self.rails.generate_async(messages=messages)
        raw_content = raw_result.get("content", "") if isinstance(raw_result, dict) else str(raw_result)

        clean_text, thinking_trace = self.extract_clean_content(raw_content)
        blocked = self.is_blocked_response(clean_text)

        rail_triggered = None
        if blocked:
            if "override" in clean_text.lower() or "prompt" in clean_text.lower():
                rail_triggered = "jailbreak_or_prompt_injection"
            elif "harmful" in clean_text.lower() or "illegal" in clean_text.lower():
                rail_triggered = "harmful_content_policy"
            elif "off-topic" in clean_text.lower() or "specialized" in clean_text.lower():
                rail_triggered = "off_topic_scope"
            else:
                rail_triggered = "safety_or_fact_check_rail"

        return {
            "answer": clean_text,
            "raw_content": raw_content,
            "thinking": thinking_trace,
            "is_blocked": blocked,
            "rail_triggered": rail_triggered,
            "context_used": effective_context,
        }


# Singleton cache for performance
_GUARD_INSTANCE: Optional[SentinelRAGGuard] = None


def get_guardrails(config_dir: Optional[Union[str, Path]] = None) -> SentinelRAGGuard:
    """Returns or initializes the singleton instance of SentinelRAGGuard."""
    global _GUARD_INSTANCE
    if _GUARD_INSTANCE is None:
        _GUARD_INSTANCE = SentinelRAGGuard(config_dir)
    return _GUARD_INSTANCE


def guarded_answer(query: str, context: str = "") -> str:
    """Convenience functional helper returning only the answer string."""
    guard = get_guardrails()
    result = guard.generate(query=query, context=context)
    return result["answer"]
