"""
SentinelRAG FastAPI application.

Serves the Kotaemon-inspired frontend and exposes REST endpoints for
chat, document ingestion, retrieval evidence, guardrail testing, and RAGAS evaluation.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / "backend" / ".env")

import backend.Evaluation.compat  # noqa: F401

FRONTEND_DIR = PROJECT_ROOT / "frontend"
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
STATE_FILE = DATA_DIR / "app_state.json"
EVAL_REPORT = PROJECT_ROOT / "backend" / "Evaluation" / "evaluation_report.json"
GOLDENS_FILE = PROJECT_ROOT / "backend" / "Evaluation" / "goldens.json"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"conversations": [], "files": []}


def save_state(state: Dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def get_conversation(state: Dict[str, Any], conversation_id: str) -> Optional[Dict[str, Any]]:
    for conv in state["conversations"]:
        if conv["id"] == conversation_id:
            return conv
    return None


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None
    k: int = Field(default=5, ge=1, le=20)
    sanitize: bool = True


class ConversationCreate(BaseModel):
    title: Optional[str] = None


class RenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)


class GoldenUpdateRequest(BaseModel):
    id: str
    ground_truth: str = Field(..., min_length=1)


class EvalRunRequest(BaseModel):
    samples: Optional[int] = Field(default=1, ge=1, le=10)


class SecurityTestRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    attack_type: Optional[str] = "custom"


app = FastAPI(
    title="SentinelRAG",
    description="Guarded retrieval-augmented generation API with Kotaemon-inspired UI.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")


@app.on_event("startup")
def startup_event():
    try:
        from backend.database.database import init_database
        from backend.database.vector_store import vector_store
        init_database()
        vector_store.create_tables_if_not_exists()
    except Exception as exc:
        print(f"[WARN] Startup table check: {exc}")


@app.get("/")
def index():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend is missing.")
    return FileResponse(index_path)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        'stroke="#0d7377" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>'
        '</svg>'
    )
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/api/health")
def health():
    db_ok = False
    db_error = None
    total_embeddings = 0
    try:
        from backend.database.database import engine

        with engine.connect() as connection:
            res = connection.execute(text("SELECT count(*) FROM langchain_pg_embedding"))
            total_embeddings = res.scalar() or 0
        db_ok = True
    except Exception as exc:
        db_error = str(exc)

    nvidia_ok = bool(os.getenv("NVIDIA_API_KEY"))
    status = "ok" if db_ok and nvidia_ok else "degraded"
    return {
        "status": status,
        "service": "SentinelRAG",
        "nvidia_api_key": nvidia_ok,
        "database": db_ok,
        "database_error": db_error,
        "total_embeddings": total_embeddings,
        "timestamp": utc_now(),
    }


@app.get("/api/database/stats")
def database_stats():
    from backend.database.database import engine

    try:
        with engine.connect() as conn:
            count_res = conn.execute(text("SELECT count(*) FROM langchain_pg_embedding"))
            total = count_res.scalar() or 0

            sources_res = conn.execute(
                text(
                    "SELECT DISTINCT cmetadata->>'source' as src, count(*) as cnt FROM langchain_pg_embedding GROUP BY cmetadata->>'source'"
                )
            )
            sources = []
            for row in sources_res:
                src_path = row[0] or "unknown"
                sources.append({
                    "filename": Path(src_path).name,
                    "path": src_path,
                    "chunks": row[1],
                })
        return {
            "total_chunks": total,
            "sources": sources,
            "status": "connected",
        }
    except Exception as exc:
        return {
            "total_chunks": 0,
            "sources": [],
            "status": f"error: {exc}",
        }


@app.get("/api/settings")
def settings():
    return {
        "models": {
            "llm": "nvidia/nemotron-3.5-lightning-30b-a3b",
            "embeddings": "nvidia/nemotron-3.5-embed-1b",
        },
        "defaults": {"k": 5, "sanitize": True},
        "supported_files": sorted(SUPPORTED_EXTENSIONS),
        "features": [
            "Verifiable page citations",
            "Zero-hallucination factual grounding",
            "High-density pgvector embeddings (2048-dim)",
            "Step-by-step reasoning verification",
        ],
    }


@app.get("/api/suggestions")
def suggestions():
    state = load_state()
    if not state.get("files"):
        return {"suggestions": []}
    
    filename = state["files"][0].get("filename", "the document")
    return {
        "suggestions": [
            {"id": "q1", "question": f"Summarize the main themes and key takeaways from {filename}.", "category": "summary"},
            {"id": "q2", "question": "What are the core methodologies, topics, or findings outlined here?", "category": "analysis"},
            {"id": "q3", "question": "What important recommendations or conclusions does the document provide?", "category": "insights"},
            {"id": "q4", "question": "List the most critical points and definitions explained in this text.", "category": "breakdown"},
        ]
    }


@app.post("/api/generate-questions")
async def generate_questions():
    """
    Dynamically generates tailored questions from the active stored document in pgvector.
    Analyzes the uploaded document excerpts and returns questions for one-click querying.
    """
    state = load_state()
    if not state.get("files"):
        return {"questions": [], "source": "none"}

    active_file = state["files"][0]
    raw_filename = active_file.get("filename", "the document")
    clean_fn = re.sub(r'^[a-f0-9]{32}_', '', raw_filename, flags=re.I)
    stored_path = active_file.get("stored_path", "")

    chunks_sample = []
    try:
        from backend.database.database import engine
        with engine.connect() as conn:
            # Query chunks belonging to the uploaded document
            res = conn.execute(
                text(
                    "SELECT document FROM langchain_pg_embedding "
                    "WHERE cmetadata->>'filename' = :fn OR cmetadata->>'source' = :sp "
                    "ORDER BY id ASC LIMIT 4"
                ),
                {"fn": raw_filename, "sp": stored_path},
            )
            for row in res:
                if row[0]:
                    chunks_sample.append(row[0][:500])

            # If metadata match returned 0, fallback to recent chunks
            if not chunks_sample:
                res = conn.execute(
                    text("SELECT document FROM langchain_pg_embedding ORDER BY id DESC LIMIT 3")
                )
                for row in res:
                    if row[0]:
                        chunks_sample.append(row[0][:500])

        if chunks_sample and os.getenv("NVIDIA_API_KEY"):
            try:
                from langchain_nvidia_ai_endpoints import ChatNVIDIA
                llm = ChatNVIDIA(
                    model="nvidia/nemotron-3.5-lightning-30b-a3b",
                    temperature=0.2,
                    max_completion_tokens=256,
                )
                sample_text = "\n---\n".join(chunks_sample)
                prompt = (
                    f"You are analyzing the document '{clean_fn}'. "
                    "Based strictly on the following document excerpts, write 4 insightful, natural questions "
                    "that a user can ask to understand the content of this document.\n"
                    "Requirements:\n"
                    "- Write complete grammatical questions ending with a question mark.\n"
                    "- Return only the 4 questions, one per line, with no introductory text, no numbers, and no bullets:\n\n"
                    f"{sample_text}"
                )
                response = await llm.ainvoke(prompt)
                lines = [
                    line.strip().lstrip("0123456789.-*•) ")
                    for line in response.content.split("\n")
                    if line.strip()
                ]
                generated = []
                for q in lines:
                    q = q.strip().lstrip(",.-: ")
                    if len(q) >= 15 and (" " in q):
                        if not q.endswith("?"):
                            q += "?"
                        generated.append({"id": f"gen_{len(generated)+1}", "question": q, "category": "generated"})
                        if len(generated) >= 5:
                            break
                if generated:
                    return {"questions": generated, "source": "llm"}
            except Exception as e:
                print(f"[WARN] Dynamic question generation failed: {e}")
    except Exception as e:
        print(f"[WARN] Document chunk query failed: {e}")

    # Tailored generic fallback based on the active file
    default_qs = [
        {"id": "q1", "question": f"What are the main concepts and topics covered in {clean_fn}?", "category": "summary"},
        {"id": "q2", "question": "Summarize the key takeaways and important findings.", "category": "insights"},
        {"id": "q3", "question": "What are the most critical questions or points highlighted in this document?", "category": "analysis"},
        {"id": "q4", "question": "Explain the core methodologies or recommendations described.", "category": "breakdown"},
    ]
    return {"questions": default_qs, "source": "document_fallback"}


@app.post("/api/session/clear")
def clear_session():
    """
    Purges all uploaded documents, PostgreSQL vector embeddings, and conversation history
    so no data from the session persists for any subsequent user.
    """
    from backend.database.database import engine

    try:
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE langchain_pg_embedding CASCADE;"))
    except Exception as exc:
        print(f"Error truncating embeddings: {exc}")

    # Clear uploads directory
    if UPLOAD_DIR.exists():
        for f in UPLOAD_DIR.glob("*"):
            try:
                if f.is_file():
                    f.unlink()
                elif f.is_dir():
                    shutil.rmtree(f)
            except Exception as exc:
                print(f"Error deleting file {f}: {exc}")

    # Reset state file
    empty_state = {"conversations": [], "files": []}
    save_state(empty_state)

    return {"ok": True, "message": "Session and document data cleared."}


@app.get("/api/conversations")
def list_conversations():
    state = load_state()
    conversations = sorted(state["conversations"], key=lambda c: c.get("updated_at", ""), reverse=True)
    return {"conversations": conversations}


@app.post("/api/conversations")
def create_conversation(payload: ConversationCreate):
    state = load_state()
    conv = {
        "id": str(uuid.uuid4()),
        "title": payload.title or "New conversation",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "messages": [],
    }
    state["conversations"].insert(0, conv)
    save_state(state)
    return conv


@app.patch("/api/conversations/{conversation_id}")
def rename_conversation(conversation_id: str, payload: RenameRequest):
    state = load_state()
    conv = get_conversation(state, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    conv["title"] = payload.title
    conv["updated_at"] = utc_now()
    save_state(state)
    return conv


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    state = load_state()
    before = len(state["conversations"])
    state["conversations"] = [c for c in state["conversations"] if c["id"] != conversation_id]
    if len(state["conversations"]) == before:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    save_state(state)
    return {"ok": True}


@app.post("/api/chat")
async def chat(payload: ChatRequest):
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    state = load_state()
    conv = get_conversation(state, payload.conversation_id) if payload.conversation_id else None
    if conv is None:
        conv = {
            "id": str(uuid.uuid4()),
            "title": query[:72],
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "messages": [],
        }
        state["conversations"].insert(0, conv)

    # Check for conversation exit command
    if query.lower().strip() in ("exit", "quit", "stop", "bye", "/exit", "exit.", "end", "end chat"):
        user_message = {
            "id": str(uuid.uuid4()),
            "role": "user",
            "content": query,
            "created_at": utc_now(),
        }
        assistant_message = {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": "Conversation ended. Thank you! You can ask another question or upload a document whenever you're ready.",
            "created_at": utc_now(),
            "is_exit": True,
            "sources": [],
        }
        conv["messages"].extend([user_message, assistant_message])
        conv["updated_at"] = utc_now()
        save_state(state)
        return {
            "conversation_id": conv["id"],
            "title": conv["title"],
            "message": assistant_message,
            "k": payload.k,
            "is_exit": True,
        }

    try:
        from backend.generation.generation import generate_answer_async

        result = await generate_answer_async(
            query=query,
            k=payload.k,
            return_details=True,
            sanitize=payload.sanitize,
        )
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    chunks = result.get("retrieved_chunks") or []
    sources = [
        {
            "index": idx,
            "content": chunk.get("content", ""),
            "metadata": chunk.get("metadata") or {},
            "filename": (chunk.get("metadata") or {}).get("filename")
            or Path(str((chunk.get("metadata") or {}).get("source", "unknown"))).name,
            "page": (chunk.get("metadata") or {}).get("page", 0) + 1
            if isinstance((chunk.get("metadata") or {}).get("page"), int)
            else None,
            "score": chunk.get("score"),
            "relevance": chunk.get("relevance"),
        }
        for idx, chunk in enumerate(chunks, start=1)
    ]

    user_message = {
        "id": str(uuid.uuid4()),
        "role": "user",
        "content": query,
        "created_at": utc_now(),
    }
    assistant_message = {
        "id": str(uuid.uuid4()),
        "role": "assistant",
        "content": result.get("answer", ""),
        "created_at": utc_now(),
        "is_blocked": bool(result.get("is_blocked")),
        "rail_triggered": result.get("rail_triggered"),
        "thinking": result.get("thinking"),
        "sources": sources,
    }

    conv["messages"].extend([user_message, assistant_message])
    if conv["title"] in {"New conversation", ""}:
        conv["title"] = query[:72]
    conv["updated_at"] = utc_now()
    save_state(state)

    return {
        "conversation_id": conv["id"],
        "title": conv["title"],
        "message": assistant_message,
        "k": payload.k,
    }


@app.get("/api/files")
def list_files():
    state = load_state()
    return {"files": sorted(state["files"], key=lambda f: f.get("uploaded_at", ""), reverse=True)}


@app.post("/api/files")
async def upload_files(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    from backend.ingestion.loader import ingest_document

    state = load_state()
    ingested = []
    errors = []

    for upload in files:
        filename = Path(upload.filename or "document").name
        extension = Path(filename).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            errors.append({"filename": filename, "error": f"Unsupported type {extension}"})
            continue

        dest = UPLOAD_DIR / f"{uuid.uuid4().hex}_{filename}"
        content = await upload.read()
        dest.write_bytes(content)

        try:
            ids = ingest_document(str(dest))
            record = {
                "id": str(uuid.uuid4()),
                "filename": filename,
                "stored_path": str(dest),
                "size_bytes": len(content),
                "chunks": len(ids),
                "uploaded_at": utc_now(),
            }
            state["files"].insert(0, record)
            ingested.append(record)
        except Exception as exc:
            traceback.print_exc()
            errors.append({"filename": filename, "error": str(exc)})

    save_state(state)
    return {"ingested": ingested, "errors": errors}


@app.get("/api/evaluation")
def evaluation_report():
    if not EVAL_REPORT.exists():
        return {"available": False, "report": None}
    try:
        report = json.loads(EVAL_REPORT.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Evaluation report is not valid JSON.")
    return {"available": True, "report": json_safe(report)}


@app.get("/api/evaluation/goldens")
def get_goldens():
    if not GOLDENS_FILE.exists():
        return {"goldens": []}
    try:
        data = json.loads(GOLDENS_FILE.read_text(encoding="utf-8"))
        return {"goldens": data}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/evaluation/goldens")
def update_golden(payload: GoldenUpdateRequest):
    from backend.Evaluation.goldens import update_golden_answer

    success = update_golden_answer(payload.id, payload.ground_truth)
    if not success:
        raise HTTPException(status_code=404, detail=f"Golden ID '{payload.id}' not found.")
    return {"ok": True, "message": f"Updated golden '{payload.id}' successfully."}


@app.post("/api/evaluation/run")
def trigger_evaluation(payload: EvalRunRequest):
    from backend.Evaluation.evaluator import run_evaluation

    try:
        report = run_evaluation(limit=payload.samples, verbose=False)
        return {"ok": True, "report": json_safe(report)}
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}")


@app.post("/api/security/test")
async def test_security(payload: SecurityTestRequest):
    from backend.generation.generation import generate_answer_async

    try:
        res = await generate_answer_async(
            query=payload.prompt,
            k=3,
            return_details=True,
            sanitize=True,
        )
        return {
            "prompt": payload.prompt,
            "attack_type": payload.attack_type,
            "is_blocked": res.get("is_blocked", False),
            "rail_triggered": res.get("rail_triggered"),
            "response": res.get("answer", ""),
            "thinking": res.get("thinking", ""),
            "retrieved_count": len(res.get("retrieved_chunks", [])),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/retrieve")
async def retrieve(payload: ChatRequest):
    try:
        from backend.retrievel.retriever import retrieve_context

        chunks = retrieve_context(payload.query.strip(), k=payload.k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc
    return {"chunks": chunks}


if __name__ == "__main__":
    import asyncio
    import uvicorn

    # Ensure asyncio.run tolerates loop_factory even if nest_asyncio patched it
    _orig_asyncio_run = asyncio.run

    def _safe_asyncio_run(main, *, debug=False, **kwargs):
        try:
            return _orig_asyncio_run(main, debug=debug, **kwargs)
        except TypeError:
            return _orig_asyncio_run(main, debug=debug)

    asyncio.run = _safe_asyncio_run

    print("\n" + "=" * 70)
    print("  LAUNCHING SENTINEL-RAG WEB SERVER")
    print("  UI & REST API: http://127.0.0.1:8000")
    print("  Interactive Swagger Docs: http://127.0.0.1:8000/docs")
    print("=" * 70 + "\n")

    try:
        uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=False)
    except (KeyboardInterrupt, SystemExit):
        print("\n[INFO] Server gracefully stopped.")
        sys.exit(0)
    except OSError as e:
        if "10048" in str(e):
            print("\n[ERROR] Port 8000 is already in use by another process.")
        else:
            print(f"\n[ERROR] Server socket error: {e}")
        sys.exit(1)
