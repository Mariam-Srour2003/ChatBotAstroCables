"""
app/api.py — FastAPI server.

Endpoints:
    GET  /                        → serves the chat widget (static/index.html)
    POST /chat                    → non-streaming JSON response (LangGraph)
    POST /chat/stream             → SSE streaming response (Groq / local fallback)
    POST /session/{id}/email      → send transcript to email immediately
    GET  /health                  → liveness probe
    DELETE /session/{id}          → clear session history

Start:
    uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
"""
import asyncio
import json
import string
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from app.config import (
    GREETING_MAX_WORDS, GREETING_RESPONSE, GREETING_WORDS,
    INACTIVITY_MINUTES, OFF_TOPIC_RESPONSE, SALES_PROMPT,
)
from app.graph import app_graph, retrieve_context, rewrite_query, stream_llm

# ── Session stores ────────────────────────────────────────────────────────────

_active_sessions:    set[str]              = set()
_stream_histories:   dict[str, list[dict]] = {}
_session_last_active: dict[str, float]     = {}   # session_id → unix timestamp
_session_emailed:    set[str]              = set() # sessions already auto-emailed
_session_contacts:   dict[str, dict]       = {}   # session_id → contact info dict

MAX_HISTORY         = 20
_INACTIVITY_SECS    = INACTIVITY_MINUTES * 60

# ── Inactivity monitor ────────────────────────────────────────────────────────

async def _inactivity_monitor() -> None:
    """Background task: auto-emails sessions idle for INACTIVITY_MINUTES."""
    from app.email_sender import send_transcript
    while True:
        await asyncio.sleep(60)
        now = time.time()
        for sid, last_t in list(_session_last_active.items()):
            if sid in _session_emailed:
                continue
            if now - last_t < _INACTIVITY_SECS:
                continue
            history = _stream_histories.get(sid, [])
            if len(history) < 2:
                continue
            contact = _session_contacts.get(sid, {})
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(None, send_transcript, list(history), sid, contact)
                _session_emailed.add(sid)
                print(f"[email] Auto-sent transcript for session {sid[:8]}…")
            except Exception as exc:
                print(f"[email] Auto-send failed for {sid[:8]}: {exc}")

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):
    task = asyncio.create_task(_inactivity_monitor())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Astro Power Cables — Sales Chatbot API",
    version="3.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Schemas ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question:   str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer:     str
    session_id: str

class ContactRequest(BaseModel):
    name:  Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None

class EmailRequest(BaseModel):
    name:  Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None

# ── SSE helper ────────────────────────────────────────────────────────────────

def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"

_SSE_HEADERS = {
    "Cache-Control":     "no-cache",
    "X-Accel-Buffering": "no",
    "Connection":        "keep-alive",
}

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")


@app.get("/widget")
def serve_widget():
    return FileResponse("static/widget.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Non-streaming endpoint backed by LangGraph + MemorySaver."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty.")

    session_id = req.session_id or str(uuid.uuid4())
    _active_sessions.add(session_id)
    config = {"configurable": {"thread_id": session_id}}

    try:
        result = app_graph.invoke(
            {"messages": [HumanMessage(content=req.question)]}, config
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return ChatResponse(answer=result["messages"][-1].content, session_id=session_id)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    SSE streaming endpoint.
    Events:
        {"type": "token",  "content": "..."}
        {"type": "done",   "session_id": "..."}
        {"type": "error",  "message": "..."}
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty.")

    session_id = req.session_id or str(uuid.uuid4())
    history    = _stream_histories.setdefault(session_id, [])
    question   = req.question.strip()

    # Track activity; new message → allow re-email if conversation continues
    _session_last_active[session_id] = time.time()
    _session_emailed.discard(session_id)

    # ── Greeting detection ──────────────────────────────────────────────────
    words = {w.strip(string.punctuation) for w in question.lower().split()}
    if words & GREETING_WORDS and len(words) <= GREETING_MAX_WORDS:
        async def _greet():
            yield _sse({"type": "token", "content": GREETING_RESPONSE})
            yield _sse({"type": "done",  "session_id": session_id})
        _append_history(history, question, GREETING_RESPONSE)
        return StreamingResponse(_greet(), media_type="text/event-stream", headers=_SSE_HEADERS)

    # ── Query rewriting & retrieval (run in thread — both are synchronous) ──
    loop         = asyncio.get_event_loop()
    standalone_q = await loop.run_in_executor(None, rewrite_query, question, history)
    context, route, sources = await loop.run_in_executor(None, retrieve_context, standalone_q)

    if route == "off_topic":
        async def _off():
            yield _sse({"type": "token", "content": OFF_TOPIC_RESPONSE})
            yield _sse({"type": "done",  "session_id": session_id})
        _append_history(history, question, OFF_TOPIC_RESPONSE)
        return StreamingResponse(_off(), media_type="text/event-stream", headers=_SSE_HEADERS)

    # ── Build prompt ────────────────────────────────────────────────────────
    prompt = SALES_PROMPT.format(context=context, question=question)

    # ── Stream ──────────────────────────────────────────────────────────────
    async def _generate():
        full     = ""
        buf      = ""
        released = False

        try:
            async for token in stream_llm(prompt):
                full += token

                if not released:
                    buf += token
                    if len(buf) > 12:
                        if buf.lstrip().startswith("OFF_TOPIC"):
                            full = OFF_TOPIC_RESPONSE
                            released = True
                            yield _sse({"type": "token", "content": OFF_TOPIC_RESPONSE})
                            break
                        else:
                            released = True
                            yield _sse({"type": "token", "content": buf})
                else:
                    yield _sse({"type": "token", "content": token})

            if not released:
                if buf.strip() == "OFF_TOPIC":
                    full = OFF_TOPIC_RESPONSE
                    yield _sse({"type": "token", "content": OFF_TOPIC_RESPONSE})
                elif buf:
                    yield _sse({"type": "token", "content": buf})

        except Exception as exc:
            yield _sse({"type": "error", "message": str(exc)})
        finally:
            answer = full.strip()
            if answer:
                _append_history(history, question, answer)
            yield _sse({"type": "done", "session_id": session_id})

    return StreamingResponse(_generate(), media_type="text/event-stream", headers=_SSE_HEADERS)


@app.patch("/session/{session_id}/contact")
async def save_contact(session_id: str, req: ContactRequest):
    """Store contact info for a session (used by auto-send and manual email)."""
    contact = {k: v for k, v in {"name": req.name, "email": req.email,
                                   "phone": req.phone}.items() if v}
    _session_contacts[session_id] = contact
    return {"saved": True}


@app.post("/session/{session_id}/email")
async def email_transcript(
    session_id: str,
    name:  Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
):
    """Send the full chat transcript (+ optional contact info) as an HTML email."""
    from app.email_sender import send_transcript
    history = _stream_histories.get(session_id, [])
    if len(history) < 2:
        raise HTTPException(status_code=400, detail="No conversation to send yet.")

    # Merge stored contact with anything passed as query params
    contact = dict(_session_contacts.get(session_id, {}))
    if name:  contact["name"]  = name
    if email: contact["email"] = email
    if phone: contact["phone"] = phone
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, send_transcript, list(history), session_id, contact)
        _session_emailed.add(session_id)
        return {"sent": True, "exchanges": len(history) // 2}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Email failed: {exc}")


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    _active_sessions.discard(session_id)
    _stream_histories.pop(session_id, None)
    _session_last_active.pop(session_id, None)
    _session_emailed.discard(session_id)
    _session_contacts.pop(session_id, None)
    return {"cleared": True}


@app.get("/sessions")
def list_sessions():
    return {"active_sessions": list(_active_sessions)}

# ── Internal ──────────────────────────────────────────────────────────────────

def _append_history(history: list, question: str, answer: str) -> None:
    history.append({"role": "user",      "content": question})
    history.append({"role": "assistant", "content": answer})
    if len(history) > MAX_HISTORY:
        del history[:2]

# ── Dev runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=False)
