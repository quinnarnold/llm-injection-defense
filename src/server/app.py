# src/server/app.py
"""
FastAPI server wiring: preprocessor -> classifier -> Haystack RAG -> vLLM -> LLM Guard.

Usage: uvicorn src.server.app:app --host 0.0.0.0 --port 8080
"""
import asyncio
import os
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.preprocessor.pipeline import preprocess
from src.classifier.model import InjectionClassifier
from src.classifier.heuristics import check_heuristics
from src.guards import scan_input, scan_response
from src.server.prompt_template import build_messages
from src.rag.haystack_pipe import retrieve

REFUSAL = (
    "I'm sorry, but I'm unable to help with that request. "
    "Can I assist you with something about Bryant University?"
)

_ip_timestamps: dict[str, list[float]] = defaultdict(list)
_turn_counts: dict[str, int] = defaultdict(int)
_valid_sessions: set[str] = set()
_session_last_seen: dict[str, float] = {}
MAX_PER_MINUTE = int(os.environ.get("MAX_REQUESTS_PER_MINUTE_PER_IP", "20"))
MAX_TURNS = int(os.environ.get("MAX_TURNS_PER_SESSION", "50"))
SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", "3600"))

classifier: InjectionClassifier | None = None
retrieval_pipeline = None
vllm_url: str = ""
_http_client: httpx.AsyncClient | None = None


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = Field(
        default=None,
        description="Server-issued token for turn limiting only. Conversation history is stateless.",
    )


class ChatResponse(BaseModel):
    response: str
    session_id: str


@asynccontextmanager
async def lifespan(app):
    global classifier, retrieval_pipeline, vllm_url, _http_client
    classifier = InjectionClassifier(
        Path(os.environ.get("CLASSIFIER_PATH", "models/classifier-trained")),
    )
    vllm_url = os.environ.get("VLLM_URL", "http://localhost:8000")

    from src.rag.haystack_pipe import build_retrieval_pipeline
    from haystack_integrations.document_stores.chroma import ChromaDocumentStore
    chroma = ChromaDocumentStore(
        collection_name="bryant_corpus",
        persist_path=os.environ.get("CHROMA_PATH", "data/chroma"),
    )
    retrieval_pipeline, bm25_store = build_retrieval_pipeline(chroma)
    bm25_docs = chroma.filter_documents()
    bm25_store.write_documents(bm25_docs)
    retrieve(retrieval_pipeline, "warmup")
    _http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5.0, read=55.0, write=5.0, pool=5.0)
    )
    yield
    await _http_client.aclose()


app = FastAPI(title="Ask Tupper", version="0.2.0", lifespan=lifespan)

_allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["POST"],
    allow_headers=["*"],
)


def _get_client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def reset_runtime_state() -> None:
    """Clear in-memory rate-limit and session state."""
    _ip_timestamps.clear()
    _turn_counts.clear()
    _valid_sessions.clear()
    _session_last_seen.clear()


def _prune_expired_state(now: float | None = None) -> None:
    """Drop stale rate-limit buckets and expired session records."""
    current_time = time.time() if now is None else now

    for ip, timestamps in list(_ip_timestamps.items()):
        recent_timestamps = [timestamp for timestamp in timestamps if current_time - timestamp < 60]
        if recent_timestamps:
            _ip_timestamps[ip] = recent_timestamps
        else:
            _ip_timestamps.pop(ip, None)

    expired_sessions = [
        session_id
        for session_id, last_seen in _session_last_seen.items()
        if current_time - last_seen >= SESSION_TTL_SECONDS
    ]
    for session_id in expired_sessions:
        _session_last_seen.pop(session_id, None)
        _turn_counts.pop(session_id, None)
        _valid_sessions.discard(session_id)


def _check_ip_rate(ip: str, now: float | None = None) -> bool:
    """Record a request for an IP and report whether it stays within the current window."""
    current_time = time.time() if now is None else now
    _prune_expired_state(current_time)
    timestamps = _ip_timestamps.setdefault(ip, [])
    if len(timestamps) >= MAX_PER_MINUTE:
        return False
    timestamps.append(current_time)
    return True


def _issue_session_id(provided: str | None, now: float) -> str:
    """Reuse a known session token or mint a new one for stateless turn limiting."""
    if provided and provided in _valid_sessions:
        session_id = provided
    else:
        session_id = str(uuid.uuid4())
        _valid_sessions.add(session_id)

    _turn_counts.setdefault(session_id, 0)
    _session_last_seen[session_id] = now
    return session_id


def _extract_model_response_text(response: httpx.Response) -> str:
    """Parse the OpenAI-compatible vLLM response payload."""
    try:
        payload = response.json()
        return payload["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise HTTPException(502, "Model backend returned an invalid response.") from exc


async def _generate_model_response(messages: list[dict]) -> str:
    """Call the model backend and translate transport failures into API errors."""
    if _http_client is None:
        raise HTTPException(503, "Service not ready")

    try:
        response = await _http_client.post(
            f"{vllm_url}/v1/chat/completions",
            json={
                "model": "qwen3-32b-bryant",
                "messages": messages,
                "max_tokens": 512,
                "temperature": 0.0,
            },
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(502, "Model backend unavailable.") from exc

    return _extract_model_response_text(response)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    if classifier is None or retrieval_pipeline is None or _http_client is None:
        raise HTTPException(503, "Service not ready")

    now = time.time()
    client_ip = _get_client_ip(request)
    if not _check_ip_rate(client_ip, now=now):
        raise HTTPException(429, "Rate limit exceeded.")

    session_id = _issue_session_id(req.session_id, now)
    _turn_counts[session_id] += 1
    _session_last_seen[session_id] = now
    if _turn_counts[session_id] > MAX_TURNS:
        raise HTTPException(429, "Session turn limit reached. Start a new conversation.")

    # Layer 1: Custom preprocessor (decode obfuscated attacks)
    cleaned = preprocess(req.message)

    # Layer 2: Custom injection classifier + unconditional heuristic check.
    # Heuristics run first (pure regex, microseconds) to short-circuit before
    # neural inference on obvious attacks. Either signal alone is sufficient to block.
    if check_heuristics(cleaned).flagged:
        return ChatResponse(response=REFUSAL, session_id=session_id)
    result = await asyncio.to_thread(classifier.predict, cleaned)
    if result.blocked:
        return ChatResponse(response=REFUSAL, session_id=session_id)

    # Layer 3: LLM Guard input scan (generic injection + toxicity)
    _, input_safe, _ = await asyncio.to_thread(scan_input, cleaned)
    if not input_safe:
        return ChatResponse(response=REFUSAL, session_id=session_id)

    # Layer 4: Haystack hybrid retrieval
    chunks = await asyncio.to_thread(retrieve, retrieval_pipeline, cleaned)
    chunk_dicts = [{"text": d.content, "id": d.id} for d in chunks]

    # Layer 5: Build prompt and call vLLM
    messages = build_messages(cleaned, chunk_dicts)
    response_text = await _generate_model_response(messages)

    # Layer 6: LLM Guard output scan (code, PII, sensitive content)
    sanitized, _, _ = await asyncio.to_thread(scan_response, cleaned, response_text)

    return ChatResponse(response=sanitized, session_id=session_id)


@app.get("/health")
async def health():
    return {"status": "ok"}
