"""
app/sessions.py — conversation state shared by every channel.

The website widget and WhatsApp both store their history here, so a single
inactivity monitor emails transcripts for both. Session ids are namespaced by
channel:

    web       → the uuid4 the browser sends
    whatsapp  → "wa:<phone number>"

Nothing here is persistent: restarting the server clears every conversation.
"""
import asyncio
import time

from app.config import INACTIVITY_MINUTES

MAX_HISTORY      = 20            # messages (10 exchanges), oldest dropped first
_INACTIVITY_SECS = INACTIVITY_MINUTES * 60

# ── Stores ────────────────────────────────────────────────────────────────────

active:      set[str]              = set()
histories:   dict[str, list[dict]] = {}
last_active: dict[str, float]      = {}   # session_id → unix timestamp
emailed:     set[str]              = set()  # sessions already auto-emailed
contacts:    dict[str, dict]       = {}   # session_id → {name, email, phone}
channels:    dict[str, str]        = {}   # session_id → "web" | "whatsapp"

# ── Mutators ──────────────────────────────────────────────────────────────────

def touch(session_id: str, channel: str = "web") -> list[dict]:
    """
    Mark a session as active and return its history list.

    Clearing the emailed flag means a conversation that resumes after the
    transcript went out will be sent again once it next goes quiet.
    """
    active.add(session_id)
    channels[session_id] = channel
    last_active[session_id] = time.time()
    emailed.discard(session_id)
    return histories.setdefault(session_id, [])


def append(history: list[dict], question: str, answer: str) -> None:
    history.append({"role": "user",      "content": question})
    history.append({"role": "assistant", "content": answer})
    if len(history) > MAX_HISTORY:
        del history[:2]


def set_contact(session_id: str, contact: dict) -> None:
    contacts[session_id] = contact


def merge_contact(session_id: str, **fields) -> dict:
    """Add non-empty fields to a session's contact info and return the result."""
    contact = dict(contacts.get(session_id, {}))
    contact.update({k: v for k, v in fields.items() if v})
    contacts[session_id] = contact
    return contact


def clear(session_id: str) -> None:
    active.discard(session_id)
    histories.pop(session_id, None)
    last_active.pop(session_id, None)
    emailed.discard(session_id)
    contacts.pop(session_id, None)
    channels.pop(session_id, None)

# ── Inactivity monitor ────────────────────────────────────────────────────────

async def inactivity_monitor() -> None:
    """Background task: emails the transcript of any session idle too long."""
    from app.email_sender import send_transcript
    while True:
        await asyncio.sleep(60)
        now = time.time()
        for sid, last_t in list(last_active.items()):
            if sid in emailed:
                continue
            if now - last_t < _INACTIVITY_SECS:
                continue
            history = histories.get(sid, [])
            if len(history) < 2:
                continue
            contact = contacts.get(sid, {})
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(
                    None, send_transcript, list(history), sid, contact
                )
                emailed.add(sid)
                channel = channels.get(sid, "web")
                print(f"[email] Auto-sent {channel} transcript for session {sid[:12]}")
            except Exception as exc:
                print(f"[email] Auto-send failed for {sid[:12]}: {exc}")
