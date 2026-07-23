"""
whatsapp_test.py — check the WhatsApp setup without waiting on a real customer.

    python whatsapp_test.py check
        Show which credentials are present and ask Meta to confirm the phone
        number id and access token are valid.

    python whatsapp_test.py send 9613123456 "Hello from Astro"
        Send a text. Only works inside the 24-hour service window, i.e. that
        number must have messaged your WhatsApp number in the last 24 h
        (during testing, send "hi" from your phone first).

    python whatsapp_test.py subs
        List which Meta apps receive this WhatsApp account's webhooks. If your
        app is not in the list, Meta delivers your customers' messages
        somewhere else and your bot never sees them.

    python whatsapp_test.py subscribe
        Link this WhatsApp Business Account to your app so webhooks arrive
        here. Needs WHATSAPP_WABA_ID in .env. Safe to re-run.

    python whatsapp_test.py ask "what is NYA cable"
        Run a question through the exact engine WhatsApp uses, printing the
        reply the way it will appear in WhatsApp. No Meta account needed.

Number format: digits only, with country code, no "+" and no spaces.
"""
import asyncio
import sys

import httpx

from app.config import (
    WHATSAPP_APP_SECRET, WHATSAPP_ENABLED, WHATSAPP_GRAPH_URL,
    WHATSAPP_PHONE_NUMBER_ID, WHATSAPP_TOKEN, WHATSAPP_VERIFY_TOKEN,
    WHATSAPP_WABA_ID,
)


def _tick(ok: bool) -> str:
    return "OK     " if ok else "MISSING"


async def check() -> int:
    print("\nWhatsApp configuration")
    print("-" * 46)
    print(f"  {_tick(bool(WHATSAPP_TOKEN))}  WHATSAPP_TOKEN")
    print(f"  {_tick(bool(WHATSAPP_PHONE_NUMBER_ID))}  WHATSAPP_PHONE_NUMBER_ID"
          f"  ({WHATSAPP_PHONE_NUMBER_ID or '-'})")
    print(f"  {_tick(bool(WHATSAPP_VERIFY_TOKEN))}  WHATSAPP_VERIFY_TOKEN")
    print(f"  {_tick(bool(WHATSAPP_APP_SECRET))}  WHATSAPP_APP_SECRET"
          f"  {'' if WHATSAPP_APP_SECRET else '(signature check disabled)'}")
    print("-" * 46)

    if not WHATSAPP_ENABLED:
        print("\nNot configured yet - fill in .env, then run this again.\n")
        return 1

    url = f"{WHATSAPP_GRAPH_URL}/{WHATSAPP_PHONE_NUMBER_ID}"
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(
            url,
            headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}"},
            params={"fields": "display_phone_number,verified_name,quality_rating"},
        )

    if resp.status_code >= 400:
        print(f"\nMeta rejected the credentials ({resp.status_code}):\n{resp.text}\n")
        print("Common causes: the temporary token expired (they last 24 h), or "
              "the phone number id belongs to a different app.\n")
        return 1

    data = resp.json()
    print("\nMeta says these credentials are live:")
    print(f"  Number   : {data.get('display_phone_number', '?')}")
    print(f"  Name     : {data.get('verified_name', '?')}")
    print(f"  Quality  : {data.get('quality_rating', '?')}\n")
    return 0


def _my_app_id() -> str | None:
    """The app id the current access token belongs to."""
    try:
        r = httpx.get(
            f"{WHATSAPP_GRAPH_URL}/debug_token",
            params={"input_token": WHATSAPP_TOKEN, "access_token": WHATSAPP_TOKEN},
            timeout=20,
        )
        return str(r.json()["data"]["app_id"])
    except Exception:
        return None


def _require_waba() -> bool:
    if WHATSAPP_WABA_ID:
        return True
    print("WHATSAPP_WABA_ID is not set in .env.\n"
          "Find it on Meta -> WhatsApp -> API Setup, labelled\n"
          "'WhatsApp Business Account ID' (a ~15-digit number).")
    return False


async def subs() -> int:
    """Show which apps Meta delivers this WABA's webhooks to."""
    if not (WHATSAPP_ENABLED and _require_waba()):
        return 1

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(
            f"{WHATSAPP_GRAPH_URL}/{WHATSAPP_WABA_ID}/subscribed_apps",
            params={"access_token": WHATSAPP_TOKEN},
        )
    if resp.status_code >= 400:
        print(f"Failed ({resp.status_code}): {resp.text}")
        return 1

    mine = _my_app_id()
    rows = resp.json().get("data", [])
    print(f"\nApps receiving webhooks for WABA {WHATSAPP_WABA_ID}:")
    if not rows:
        print("  (none)")
    found = False
    for row in rows:
        info = row.get("whatsapp_business_api_data", {})
        app_id = str(info.get("id", "?"))
        is_mine = app_id == mine
        found = found or is_mine
        print(f"  {'>> ' if is_mine else '   '}{app_id}  {info.get('name', '?')}"
              f"{'   <- YOUR APP' if is_mine else ''}")

    print(f"\nYour app id: {mine or 'unknown'}")
    if found:
        print("Your app IS subscribed - webhooks will reach this bot.\n")
        return 0
    print("Your app is NOT subscribed - Meta is delivering messages elsewhere,\n"
          "so the bot will never see them. Fix it with:\n"
          "    python whatsapp_test.py subscribe\n")
    return 1


async def subscribe() -> int:
    """Link this WABA to the app that owns the access token."""
    if not (WHATSAPP_ENABLED and _require_waba()):
        return 1

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            f"{WHATSAPP_GRAPH_URL}/{WHATSAPP_WABA_ID}/subscribed_apps",
            params={"access_token": WHATSAPP_TOKEN},
        )
    if resp.status_code >= 400:
        print(f"Subscribe failed ({resp.status_code}): {resp.text}")
        return 1

    print(f"Subscribed app {_my_app_id()} to WABA {WHATSAPP_WABA_ID}.")
    print("Verifying...")
    return await subs()


async def send(to: str, body: str) -> int:
    from app.whatsapp import send_text
    if not WHATSAPP_ENABLED:
        print("WhatsApp is not configured — see .env.example.")
        return 1
    try:
        await send_text(to, body)
    except Exception as exc:
        print(f"Send failed: {exc}")
        return 1
    print(f"Sent to +{to}.")
    return 0


def ask(question: str) -> int:
    # Imported here so `check` and `send` stay fast — this pulls in the
    # embeddings model and the FAISS vectorstore.
    from app.engine import answer
    from app.whatsapp import split_message, to_whatsapp_markdown

    reply = answer(question, [])
    parts = split_message(to_whatsapp_markdown(reply))
    for i, part in enumerate(parts, 1):
        label = f" (message {i}/{len(parts)})" if len(parts) > 1 else ""
        print(f"\n--- WhatsApp reply{label} ---\n{part}")
    print()
    return 0


def main() -> int:
    args = sys.argv[1:]
    cmd  = args[0] if args else "check"

    if cmd == "check":
        return asyncio.run(check())
    if cmd == "subs":
        return asyncio.run(subs())
    if cmd == "subscribe":
        return asyncio.run(subscribe())
    if cmd == "send":
        if len(args) < 3:
            print('Usage: python whatsapp_test.py send <number> "<message>"')
            return 2
        return asyncio.run(send(args[1].lstrip("+"), " ".join(args[2:])))
    if cmd == "ask":
        if len(args) < 2:
            print('Usage: python whatsapp_test.py ask "<question>"')
            return 2
        return ask(" ".join(args[1:]))

    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
