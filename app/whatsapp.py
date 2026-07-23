"""
app/whatsapp.py — WhatsApp channel via the Meta Cloud API.

Same bot, same vectorstore, same prompts as the website: every reply goes
through app/engine.py. Only the transport differs.

Endpoints (mounted by app/api.py):
    GET  /webhook/whatsapp   → Meta's one-time subscription handshake
    POST /webhook/whatsapp   → inbound messages
    GET  /whatsapp/status    → local check that the credentials are wired up

Flow of an inbound message:
    Meta → POST webhook → signature check → 200 OK immediately
         → background task → engine.answer() → Graph API send

Replying fast matters: Meta retries a webhook it considers failed, which would
answer the customer twice. So the HTTP response never waits for the LLM.
"""
import asyncio
import hashlib
import hmac
import re
from collections import OrderedDict

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Response

from app import sessions
from app.config import (
    WHATSAPP_APP_SECRET, WHATSAPP_ENABLED, WHATSAPP_ERROR_RESPONSE,
    WHATSAPP_GRAPH_URL, WHATSAPP_MAX_CHARS, WHATSAPP_PHONE_NUMBER_ID,
    WHATSAPP_RESET_WORDS, WHATSAPP_TOKEN, WHATSAPP_UNSUPPORTED_RESPONSE,
    WHATSAPP_VERIFY_TOKEN,
)
from app.engine import answer as engine_answer

router = APIRouter(tags=["whatsapp"])

# ── De-duplication ────────────────────────────────────────────────────────────
# Meta re-delivers a webhook if we are slow or return an error, and the retry
# carries the same message id. Remembering recent ids keeps the customer from
# getting the same answer twice.

_SEEN_LIMIT = 512
_seen_messages: OrderedDict[str, None] = OrderedDict()


def _already_handled(message_id: str) -> bool:
    if message_id in _seen_messages:
        return True
    _seen_messages[message_id] = None
    while len(_seen_messages) > _SEEN_LIMIT:
        _seen_messages.popitem(last=False)
    return False


# ── Per-number serialisation ──────────────────────────────────────────────────
# Someone typing three messages in a row would otherwise have them answered
# concurrently, interleaving their history. One lock per number keeps each
# conversation in order; different customers still run in parallel.

_locks: dict[str, asyncio.Lock] = {}


def _lock_for(wa_id: str) -> asyncio.Lock:
    return _locks.setdefault(wa_id, asyncio.Lock())


# ── Formatting ────────────────────────────────────────────────────────────────

_BOLD_RE    = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_HEADING_RE = re.compile(r"^\s*#{1,6}\s*(.+?)\s*$", re.MULTILINE)
_BULLET_RE  = re.compile(r"^(\s*)[-*]\s+", re.MULTILINE)


def to_whatsapp_markdown(text: str) -> str:
    """
    Translate the model's markdown into WhatsApp's own formatting.

    WhatsApp uses *single asterisks* for bold, so **double** would show up as
    literal characters. Headings have no equivalent and become bold lines.
    """
    text = _HEADING_RE.sub(r"*\1*", text)
    text = _BOLD_RE.sub(r"*\1*", text)
    text = _BULLET_RE.sub(r"\1• ", text)
    return text.strip()


def split_message(text: str, limit: int = WHATSAPP_MAX_CHARS) -> list[str]:
    """
    Cut a long answer into WhatsApp-sized chunks, breaking on paragraph then
    line boundaries so specs and bullet lists are not sliced mid-sentence.
    """
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text

    while len(remaining) > limit:
        window = remaining[:limit]
        cut = window.rfind("\n\n")
        if cut < limit // 2:
            cut = window.rfind("\n")
        if cut < limit // 2:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = limit
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].lstrip()

    if remaining:
        chunks.append(remaining)
    return [c for c in chunks if c]


# ── Graph API calls ───────────────────────────────────────────────────────────

def _require_config() -> None:
    if not WHATSAPP_ENABLED:
        raise RuntimeError(
            "WhatsApp is not configured - set WHATSAPP_TOKEN and "
            "WHATSAPP_PHONE_NUMBER_ID in .env"
        )


async def _graph_post(payload: dict) -> dict:
    _require_config()
    url = f"{WHATSAPP_GRAPH_URL}/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type":  "application/json",
    }
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(url, headers=headers, json=payload)
    if resp.status_code >= 400:
        # Meta puts the useful part in the body, not the status line.
        raise RuntimeError(f"Graph API {resp.status_code}: {resp.text}")
    return resp.json()


async def send_text(to: str, body: str) -> None:
    """Send one or more text messages, splitting anything over the limit."""
    for chunk in split_message(to_whatsapp_markdown(body)):
        await _graph_post({
            "messaging_product": "whatsapp",
            "recipient_type":    "individual",
            "to":                to,
            "type":              "text",
            "text":              {"preview_url": False, "body": chunk},
        })


async def mark_read(message_id: str, typing: bool = True) -> None:
    """
    Show the blue ticks and (optionally) the typing bubble.

    Cosmetic only — an answer that takes a few seconds looks much less broken
    with it, so a failure here is logged and swallowed.
    """
    payload = {
        "messaging_product": "whatsapp",
        "status":            "read",
        "message_id":        message_id,
    }
    if typing:
        payload["typing_indicator"] = {"type": "text"}
    try:
        await _graph_post(payload)
    except Exception as exc:
        print(f"[whatsapp] mark_read failed: {exc}")


# ── Signature verification ────────────────────────────────────────────────────

def verify_signature(raw_body: bytes, header: str | None) -> bool:
    """
    Check Meta's X-Hub-Signature-256 against the raw request body.

    Without this anyone who learns the webhook URL can feed the bot fake
    customers. Skipped only when no app secret is configured.
    """
    if not WHATSAPP_APP_SECRET:
        return True
    if not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(
        WHATSAPP_APP_SECRET.encode(), raw_body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, header.split("=", 1)[1].strip())


# ── Message handling ──────────────────────────────────────────────────────────

def _extract_messages(payload: dict) -> list[tuple[dict, dict]]:
    """
    Pull (message, contact-profile) pairs out of a webhook payload.

    Meta nests these several levels deep and mixes in delivery-status events we
    do not care about, so anything unexpected is skipped rather than raised.
    """
    out: list[tuple[dict, dict]] = []
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            messages = value.get("messages") or []
            if not messages:
                continue          # delivery/read status event
            profiles = {
                c.get("wa_id"): (c.get("profile") or {})
                for c in value.get("contacts", [])
            }
            for msg in messages:
                out.append((msg, profiles.get(msg.get("from"), {})))
    return out


async def handle_message(msg: dict, profile: dict) -> None:
    """Answer one inbound WhatsApp message. Runs outside the webhook response."""
    wa_id      = msg.get("from", "")
    message_id = msg.get("id", "")
    if not wa_id:
        return

    session_id = f"wa:{wa_id}"

    async with _lock_for(wa_id):
        if message_id:
            await mark_read(message_id)

        # Non-text messages (images, voice notes, locations, ...) — the bot is
        # text-only, so say so rather than silently ignoring the customer.
        if msg.get("type") != "text":
            await send_text(wa_id, WHATSAPP_UNSUPPORTED_RESPONSE)
            return

        question = (msg.get("text") or {}).get("body", "").strip()
        if not question:
            return

        if question.lower() in WHATSAPP_RESET_WORDS:
            sessions.clear(session_id)
            await send_text(wa_id, "Conversation cleared. What can I help you with?")
            return

        history = sessions.touch(session_id, channel="whatsapp")
        # The phone number is a real sales lead — attach it (and the WhatsApp
        # display name) so the emailed transcript says who was asking.
        sessions.merge_contact(
            session_id, name=profile.get("name"), phone=f"+{wa_id}"
        )

        loop = asyncio.get_event_loop()
        try:
            reply = await loop.run_in_executor(
                None, engine_answer, question, list(history)
            )
        except Exception as exc:
            print(f"[whatsapp] answer failed for +{wa_id}: {exc}")
            await send_text(wa_id, WHATSAPP_ERROR_RESPONSE)
            return

        try:
            await send_text(wa_id, reply)
        except Exception as exc:
            print(f"[whatsapp] send failed for +{wa_id}: {exc}")
            return

        sessions.append(history, question, reply)
        # ASCII only: the Windows console is cp1252, which has no "->" arrow and
        # raises UnicodeEncodeError on one, killing the handler mid-reply.
        print(f"[whatsapp] +{wa_id}: {question[:60]!r} -> {len(reply)} chars")


async def _process(payload: dict) -> None:
    for msg, profile in _extract_messages(payload):
        if _already_handled(msg.get("id", "")):
            continue
        try:
            await handle_message(msg, profile)
        except Exception as exc:
            print(f"[whatsapp] handler error: {exc}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/webhook/whatsapp")
async def verify_webhook(request: Request):
    """
    Meta calls this once when you save the webhook URL and expects the
    hub.challenge value echoed back as plain text.
    """
    params = request.query_params
    mode      = params.get("hub.mode")
    token     = params.get("hub.verify_token")
    challenge = params.get("hub.challenge", "")

    if not WHATSAPP_VERIFY_TOKEN:
        raise HTTPException(500, "WHATSAPP_VERIFY_TOKEN is not set in .env")

    if mode == "subscribe" and hmac.compare_digest(token or "", WHATSAPP_VERIFY_TOKEN):
        print("[whatsapp] Webhook verified by Meta.")
        return Response(content=challenge, media_type="text/plain")

    print("[whatsapp] Webhook verification REJECTED (token mismatch).")
    raise HTTPException(403, "Verification failed")


@router.post("/webhook/whatsapp")
async def receive_webhook(request: Request, background: BackgroundTasks):
    """Accept an inbound message and answer it in the background."""
    raw = await request.body()

    if not verify_signature(raw, request.headers.get("X-Hub-Signature-256")):
        print("[whatsapp] Rejected webhook: bad signature.")
        raise HTTPException(403, "Invalid signature")

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Malformed JSON")

    background.add_task(_process, payload)
    return {"status": "received"}


@router.get("/whatsapp/status")
def whatsapp_status():
    """Config check. Reports what is set, never the secrets themselves."""
    return {
        "enabled":          WHATSAPP_ENABLED,
        "phone_number_id":  WHATSAPP_PHONE_NUMBER_ID or None,
        "token_set":        bool(WHATSAPP_TOKEN),
        "verify_token_set": bool(WHATSAPP_VERIFY_TOKEN),
        "signature_check":  bool(WHATSAPP_APP_SECRET),
        "active_chats":     sum(
            1 for c in sessions.channels.values() if c == "whatsapp"
        ),
    }
