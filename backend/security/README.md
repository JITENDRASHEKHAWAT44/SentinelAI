# SentinelRAG Security: NeMo Guardrails Integration

This directory contains the production-grade **NeMo Guardrails** configuration and runtime security layer for **SentinelRAG**. It enforces multi-tier security boundaries across user inputs, untrusted retrieval documents, and generated model outputs.

---

## 1. Security Architecture & Threat Model

```
                    ┌───────────────────────────┐
                    │     User Query Prompt     │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                ┌───────────────────────────────────┐
                │        LAYER 1: INPUT RAILS       │
                │ • Direct Prompt Injection Filter  │
                │ • Jailbreak & DAN Detection       │
                │ • System Prompt Exfiltration Block│
                │ • Content Safety & Policy Checks  │
                └─────────────────┬─────────────────┘
                                  │ (Allowed)
                                  ▼
                ┌───────────────────────────────────┐
                │   LAYER 2: CONTEXT SANITIZATION   │
                │ • Indirect Prompt Injection Scrub │
                │ • Untrusted Directive Stripping   │
                │ • Markdown Exfiltration Guard     │
                │ • PII & Secret Redaction          │
                └─────────────────┬─────────────────┘
                                  │
                                  ▼
                ┌───────────────────────────────────┐
                │     LAYER 3: LLM GENERATION       │
                │ • NVIDIA NIM Nemotron Model       │
                │ • Strict Document Grounding Prompt│
                └─────────────────┬─────────────────┘
                                  │
                                  ▼
                ┌───────────────────────────────────┐
                │       LAYER 4: OUTPUT RAILS       │
                │ • Secret & System Prompt Leakage  │
                │ • Toxic / Harmful Output Block    │
                └─────────────────┬─────────────────┘
                                  │
                                  ▼
                ┌───────────────────────────────────┐
                │    LAYER 5: FACT-CHECKING RAIL    │
                │ • Self-Check Facts against chunks │
                │ • Anti-Hallucination Verification │
                └─────────────────┬─────────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │ Verified Safe RAG Answer  │
                    └───────────────────────────┘
```

---

## 2. Directory Structure

```
backend/security/
├── config.yml              # Core NeMo Guardrails config (NIM models, general instructions, rails)
├── prompts.yml             # Task prompt templates (self_check_input, self_check_output, self_check_facts)
├── actions.py              # Custom Python actions (context sanitizer, PII redactor, grounding heuristic)
├── rag_guard.py            # SentinelRAGGuard service interface (sync & async, think tag stripper)
├── __init__.py             # Public exports (SentinelRAGGuard, get_guardrails, guarded_answer)
├── test_guardrails.py      # Main test entrypoint
├── test_security_rails.py  # Full 7-stage test suite covering injection, leaks, facts, and sanitization
├── README.md               # Architecture and usage documentation
└── rails/                  # Colang definitions
    ├── greeting.co         # Fast-path friendly greeting flow
    ├── security.co         # Jailbreak and harmful intent flows
    ├── input.co            # Off-topic domain boundary flows
    └── test_guardrails.py  # Standalone test runner in rails directory
```

---

## 3. Configuration Breakdown

### `config.yml`
- **Main Model**: `nvidia/nemotron-3.5-lightning-30b-a3b` running on the high-performance NVIDIA NIM engine (`https://integrate.api.nvidia.com/v1`).
- **Instructions**: Explicit system constraints defining SentinelRAG as an assistant that strictly adheres to provided documents and treats retrieved context as untrusted data.
- **Enabled Rails**:
  - `input.flows`: `self check input`
  - `output.flows`: `self check output`, `self check facts`

### `prompts.yml`
- **`self_check_input`**: Inspects user queries for prompt injections ("ignore previous instructions"), jailbreaks (DAN, bypass), system prompt exfiltration, and cyber/malicious requests.
- **`self_check_output`**: Verifies that generated answers do not leak system prompts, API keys, or harmful materials.
- **`self_check_facts`**: Evaluates whether every factual claim in the response is strictly grounded in the document evidence.

### `actions.py`
- `sanitize_context(context)`: Strips indirect prompt injection attempts embedded inside scraped or retrieved documents (e.g., hidden override directives, fake system delimiters, markdown image tracking tags).
- `mask_sensitive_pii(text)`: Redacts credit card numbers, SSNs, and exposed API keys (`nvapi-...`).
- `check_context_relevance(query, context)`: Fast overlap heuristic to evaluate context relevance before LLM invocation.

---

## 4. How to Use the Guardrails in the RAG Project

The security layer is packaged as a clean drop-in service. When ready to connect it to `backend/generation/generation.py` or FastAPI in `backend/main.py`, simply use the following pattern:

### Synchronous Usage
```python
from backend.security import get_guardrails

# 1. Initialize guardrails (singleton instance)
guard = get_guardrails()

# 2. Retrieve your document context from vector store
context = "\n\n".join(doc["content"] for doc in documents)

# 3. Generate guarded answer
result = guard.generate(
    query="What are the key findings in the student performance report?",
    context=context,
    sanitize=True
)

# 4. Access verified response and security metadata
answer = result["answer"]
is_blocked = result["is_blocked"]
rail_triggered = result["rail_triggered"]
```

### Asynchronous Usage (FastAPI)
```python
from backend.security import get_guardrails

guard = get_guardrails()

@app.post("/chat")
async def chat_endpoint(query: str):
    context = await retrieve_relevant_chunks(query)
    result = await guard.generate_async(query=query, context=context)
    return {
        "response": result["answer"],
        "blocked": result["is_blocked"],
        "rail_triggered": result["rail_triggered"]
    }
```

---

## 5. Running the Tests

To run the automated security test suite:

```powershell
.\RAG\Scripts\python.exe backend\security\test_guardrails.py
```

Or run the test suite directly:

```powershell
.\RAG\Scripts\python.exe backend\security\test_security_rails.py
```

### Test Coverage
1. **Greeting Rail**: Verifies fast greeting response without unnecessary LLM generation calls.
2. **Document Grounding**: Verifies accurate fact-checked answer extracted from sample report data.
3. **Direct Prompt Injection**: Verifies attempts like `Ignore all instructions` are blocked.
4. **System Prompt Leak**: Verifies requests to reveal developer instructions are rejected.
5. **Harmful Content**: Verifies requests for dangerous tools/malware are denied.
6. **Indirect Prompt Injection**: Verifies malicious directives hidden inside document chunks are sanitized before inference.
7. **Anti-Hallucination (Missing Context)**: Verifies the assistant does not invent facts when information is not present in document chunks.
