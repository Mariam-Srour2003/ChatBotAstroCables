"""
app/engine.py — the one brain behind every channel.

Greeting detection, query rewriting, retrieval, off-topic routing and prompt
building live here so the website widget and WhatsApp answer identically. Change
the behaviour once and both channels follow.

    prepare(question, history) → ("static", final_text) | ("prompt", llm_prompt)
        Website streaming uses this, then streams the prompt itself.

    answer(question, history)  → final answer text
        WhatsApp (and any other non-streaming caller) uses this.

Both are blocking: call them from a thread when on the event loop.
"""
import string

from app.config import (
    GREETING_MAX_WORDS, GREETING_RESPONSE, GREETING_WORDS,
    OFF_TOPIC_RESPONSE, SALES_PROMPT,
)
from app.graph import invoke_llm, retrieve_context, rewrite_query

STATIC = "static"   # payload is the final answer, no LLM call needed
PROMPT = "prompt"   # payload is a prompt to send to the LLM


def is_greeting(question: str) -> bool:
    """True for short openers like 'hi' or 'good morning'."""
    words = {w.strip(string.punctuation) for w in question.lower().split()}
    return bool(words & GREETING_WORDS) and len(words) <= GREETING_MAX_WORDS


def prepare(question: str, history: list[dict]) -> tuple[str, str]:
    """
    Decide how to answer. Returns (kind, payload) where kind is STATIC (payload
    is the answer) or PROMPT (payload is the prompt to run through the LLM).
    """
    question = question.strip()

    if is_greeting(question):
        return STATIC, GREETING_RESPONSE

    standalone_q = rewrite_query(question, history)
    context, route, _sources = retrieve_context(standalone_q)

    if route == "off_topic":
        return STATIC, OFF_TOPIC_RESPONSE

    return PROMPT, SALES_PROMPT.format(context=context, question=question)


def answer(question: str, history: list[dict]) -> str:
    """Full answer for non-streaming channels. Blocking."""
    kind, payload = prepare(question, history)
    if kind is STATIC:
        return payload

    raw = invoke_llm(payload)
    # The prompt tells the model to reply with exactly OFF_TOPIC when the
    # retrieved context is irrelevant; swap that marker for the real message.
    return OFF_TOPIC_RESPONSE if "OFF_TOPIC" in raw else raw.strip()
