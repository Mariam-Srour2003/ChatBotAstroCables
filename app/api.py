"""
app/api.py — FastAPI server.

Endpoints:
    GET  /                        → serves the chat widget (static/index.html)
    POST /chat                    → non-streaming JSON response (LangGraph)
    POST /chat/stream             → SSE streaming response (Groq / local fallback)
    POST /session/{id}/email      → send transcript to email immediately
    GET  /health                  → liveness probe
    DELETE /session/{id}          → clear session history

    /webhook/whatsapp             → WhatsApp channel (see app/whatsapp.py)

Both channels share app/engine.py and app/sessions.py, so the website widget
and WhatsApp give the same answers and both get transcript emails.

Start:
    uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
"""
import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from app import sessions
from app.config import OFF_TOPIC_RESPONSE, WHATSAPP_ENABLED
from app.engine import STATIC, prepare
from app.graph import app_graph, stream_llm
from app.whatsapp import router as whatsapp_router

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):
    task = asyncio.create_task(sessions.inactivity_monitor())
    print(f"WhatsApp channel: {'ENABLED' if WHATSAPP_ENABLED else 'disabled (no credentials)'}")
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

app.include_router(whatsapp_router)

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
    sessions.active.add(session_id)
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
    question   = req.question.strip()
    history    = sessions.touch(session_id, channel="web")

    # ── Decide how to answer (shared with WhatsApp; both calls are blocking) ─
    loop = asyncio.get_event_loop()
    kind, payload = await loop.run_in_executor(None, prepare, question, list(history))

    # Greetings and off-topic questions have a canned answer — send it as one
    # SSE token so the browser sees the same event shape either way.
    if kind is STATIC:
        async def _static():
            yield _sse({"type": "token", "content": payload})
            yield _sse({"type": "done",  "session_id": session_id})
        sessions.append(history, question, payload)
        return StreamingResponse(_static(), media_type="text/event-stream", headers=_SSE_HEADERS)

    prompt = payload

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
                sessions.append(history, question, answer)
            yield _sse({"type": "done", "session_id": session_id})

    return StreamingResponse(_generate(), media_type="text/event-stream", headers=_SSE_HEADERS)


@app.patch("/session/{session_id}/contact")
async def save_contact(session_id: str, req: ContactRequest):
    """Store contact info for a session (used by auto-send and manual email)."""
    contact = {k: v for k, v in {"name": req.name, "email": req.email,
                                   "phone": req.phone}.items() if v}
    sessions.set_contact(session_id, contact)
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
    history = sessions.histories.get(session_id, [])
    if len(history) < 2:
        raise HTTPException(status_code=400, detail="No conversation to send yet.")

    # Merge stored contact with anything passed as query params
    contact = sessions.merge_contact(session_id, name=name, email=email, phone=phone)
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, send_transcript, list(history), session_id, contact)
        sessions.emailed.add(session_id)
        return {"sent": True, "exchanges": len(history) // 2}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Email failed: {exc}")


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    sessions.clear(session_id)
    return {"cleared": True}


@app.get("/sessions")
def list_sessions():
    return {"active_sessions": list(sessions.active)}

# ── Dev runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=False)
