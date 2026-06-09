"""
app/graph.py — LangGraph chatbot pipeline.

Exported for use by api.py streaming endpoint:
    rewrite_query(question, history)  → standalone question
    retrieve_context(query)           → (context, route, sources)
    stream_llm(prompt)                → async token generator
    app_graph                         → compiled LangGraph (non-streaming /chat)
"""
import asyncio
import operator
import os
import re
import string
import threading
import time
from pathlib import Path
from typing import Annotated, AsyncGenerator, Dict, List, Optional, TypedDict

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.config import (
    COMPUTE_DEVICE, EMBEDDING_MODEL,
    GREETING_MAX_WORDS, GREETING_RESPONSE, GREETING_WORDS,
    GROQ_API_KEY, GROQ_MODEL, GROQ_REWRITE_MODEL,
    LLM_DEVICE, LLM_MAX_TOKENS, LLM_N_BATCH, LLM_N_THREADS,
    LLM_PROVIDER, LLM_STOP, MMR_FETCH_K, MODEL_PATH,
    OFF_TOPIC_RESPONSE, RELEVANCE_THRESHOLD, RETRIEVE_K,
    REWRITE_PROMPT, SALES_PROMPT, VECTORSTORE_PATH,
)

# ── State ─────────────────────────────────────────────────────────────────────

class ChatState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], operator.add]
    context:  Optional[str]
    route:    Optional[str]
    timing:   Optional[Dict[str, float]]

# ── Embeddings + vectorstore ──────────────────────────────────────────────────

print(f"Device : {COMPUTE_DEVICE.upper()}")
print(f"LLM    : {LLM_PROVIDER.upper()} "
      f"({'model=' + GROQ_MODEL if LLM_PROVIDER == 'groq' else MODEL_PATH})")

print("Loading embeddings...")
_embedding = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": COMPUTE_DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)

print("Loading vectorstore...")
_db = FAISS.load_local(
    VECTORSTORE_PATH, _embedding, allow_dangerous_deserialization=True
)

# ── Local GPT4All (fallback / primary when LLM_PROVIDER=gpt4all) ─────────────

_local_llm       = None
_local_ready     = threading.Event()
_local_available = False

def _load_local_model() -> None:
    global _local_llm, _local_available
    if not os.path.exists(MODEL_PATH):
        print(f"[local] Model not found at {MODEL_PATH} — local fallback disabled.")
        _local_ready.set()
        return
    try:
        from langchain_community.llms import GPT4All
        _local_llm = GPT4All(
            model=MODEL_PATH, backend="gptj", verbose=False,
            device=LLM_DEVICE, n_threads=LLM_N_THREADS,
            max_tokens=LLM_MAX_TOKENS, n_batch=LLM_N_BATCH,
        )
        _local_available = True
        print("[local] GPT4All ready.")
    except Exception as exc:
        print(f"[local] GPT4All failed: {exc}")
    finally:
        _local_ready.set()

def _invoke_local(prompt: str) -> str:
    _local_ready.wait()
    if not _local_available:
        return "I'm currently unavailable. Please try again in a moment."
    return _local_llm.invoke(prompt, stop=LLM_STOP)

# ── Groq clients ──────────────────────────────────────────────────────────────

_groq_client   = None
_rewrite_client = None

def _is_rate_limit(exc: Exception) -> bool:
    try:
        import groq as _g
        if isinstance(exc, _g.RateLimitError):
            return True
    except ImportError:
        pass
    msg = str(exc).lower()
    return "rate limit" in msg or "429" in msg or "too many requests" in msg

if LLM_PROVIDER == "groq":
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY not set. Add it to .env\n"
            "Get a free key at: https://console.groq.com"
        )
    from langchain_groq import ChatGroq
    _groq_client = ChatGroq(
        api_key=GROQ_API_KEY, model=GROQ_MODEL,
        max_tokens=LLM_MAX_TOKENS, temperature=0.3,
    )
    _rewrite_client = ChatGroq(
        api_key=GROQ_API_KEY, model=GROQ_REWRITE_MODEL,
        max_tokens=120, temperature=0,
    )
    print(f"Groq ready (main={GROQ_MODEL}, rewrite={GROQ_REWRITE_MODEL}).")
    print("[local] Pre-loading GPT4All in background for fallback...")
    threading.Thread(target=_load_local_model, daemon=True).start()

else:
    _load_local_model()
    _local_ready.wait()

print("All components ready.\n")

# ── Trailing-hallucination cleaner ────────────────────────────────────────────
_TRAILING = re.compile(
    r'\n(?:\[Q\]|\[CONTEXT\]|Customer|You\s*:|Question\s*:).*',
    re.IGNORECASE | re.DOTALL,
)

# ── Cable-size extractor (e.g. "1x10mm2" → cores="1", area="10") ─────────────
_SIZE_RE = re.compile(
    r'\b(\d+)\s*[xX]\s*(\d+(?:\.\d+)?)\s*mm[²2]?\b',
    re.IGNORECASE,
)

# Known cable families ordered longest-first so "NYA SOLID" beats "NYA"
_CABLE_FAMILIES = [
    "NYA SOLID", "NYA", "NYAF", "NYY", "NYM SOLID", "NYM",
    "N2XY", "NYBY", "N2XH", "N2XRY", "N2XBY", "NYZ", "NYRY",
    "NYMHY", "NYLHY", "H07Z", "PV",
]

def _exact_size_chunks(query: str) -> list:
    """
    When the query names a specific cable size (e.g. 1x10mm2), locate the
    vectorstore chunk(s) that literally contain that size for the mentioned
    cable type and return them so the LLM gets the precise spec data.

    Filtered to the cable family mentioned in the query (e.g. "NYA") so we
    never inject specs from unrelated cables.  Capped at 2 results.
    """
    m = _SIZE_RE.search(query)
    if not m:
        return []
    cores, area = m.group(1), m.group(2)
    targets = [
        f"of {cores}X{area} mm",
        f"of {cores}x{area} mm",
    ]

    # Detect which cable family the query is asking about
    query_upper = query.upper()
    cable_family = next((f for f in _CABLE_FAMILIES if f in query_upper), None)

    found = []
    try:
        for doc in _db.docstore._dict.values():
            if not any(t in doc.page_content for t in targets):
                continue
            if cable_family is not None:
                src = Path(doc.metadata.get("source", "")).stem.upper()
                if cable_family not in src:
                    continue
            found.append(doc)
            if len(found) >= 2:
                break
    except Exception:
        pass
    return found


# ── Public helpers (imported by api.py) ───────────────────────────────────────

def rewrite_query(question: str, history: list[dict]) -> str:
    """Return a self-contained version of question using conversation history."""
    if not history or _rewrite_client is None:
        return question
    history_str = "\n".join(
        f"{m['role'].title()}: {m['content']}" for m in history[-6:]
    )
    prompt = REWRITE_PROMPT.format(history=history_str, question=question)
    try:
        return _rewrite_client.invoke(
            [{"role": "user", "content": prompt}]
        ).content.strip()
    except Exception:
        return question


def retrieve_context(query: str) -> tuple[str, str, list[str]]:
    """
    Returns (context, route, sources).
    route = "off_topic" | "rag"
    Uses quick similarity check for relevance, then MMR for diverse context.
    For queries that name a specific cable size, injects the exact-match chunk
    so the LLM always has the precise spec data.
    """
    check = _db.similarity_search_with_score(query, k=1)
    if not check or check[0][1] > RELEVANCE_THRESHOLD:
        return "", "off_topic", []

    docs = _db.max_marginal_relevance_search(
        query, k=RETRIEVE_K, fetch_k=MMR_FETCH_K
    )

    # Inject exact-size chunks at front; cap total at RETRIEVE_K to keep
    # context bounded for local-model fallback (GPT4All 2048 token limit)
    existing_texts = {d.page_content for d in docs}
    injected = [d for d in _exact_size_chunks(query) if d.page_content not in existing_texts]
    if injected:
        docs = injected + docs[:RETRIEVE_K - len(injected)]

    context = "\n\n".join(doc.page_content for doc in docs)
    sources = list({
        Path(doc.metadata.get("source", "")).stem
        for doc in docs if doc.metadata.get("source")
    })
    return context, "rag", sources


def _groq_error_label(exc: Exception) -> str:
    """Return a short label for logging Groq failures."""
    if _is_rate_limit(exc):
        return "rate-limit"
    msg = str(exc).lower()
    if "401" in msg or "invalid api key" in msg or "authentication" in msg:
        return "auth-error (invalid API key)"
    return type(exc).__name__


def _invoke_llm(prompt: str) -> str:
    """Synchronous LLM call with Groq → local fallback on any error."""
    if LLM_PROVIDER == "groq" and _groq_client:
        try:
            raw = _groq_client.invoke(
                [{"role": "user", "content": prompt}], stop=LLM_STOP
            ).content
            return _TRAILING.sub("", raw).strip()
        except Exception as exc:
            label = _groq_error_label(exc)
            print(f"[fallback] Groq {label} — using local model.")
            return _invoke_local(prompt)
    return _invoke_local(prompt)


async def stream_llm(prompt: str) -> AsyncGenerator[str, None]:
    """Async generator that streams tokens from Groq or falls back to local."""
    loop = asyncio.get_event_loop()
    if LLM_PROVIDER == "groq" and _groq_client:
        try:
            async for chunk in _groq_client.astream(
                [{"role": "user", "content": prompt}], stop=LLM_STOP
            ):
                if chunk.content:
                    yield chunk.content
            return
        except Exception as exc:
            label = _groq_error_label(exc)
            print(f"[fallback] Groq {label} during stream — using local model.")
            # Run synchronous local model in thread so event loop stays free
            result = await loop.run_in_executor(None, _invoke_local, prompt)
            if result:
                yield result
            return
    # Primary local path — also non-blocking
    result = await loop.run_in_executor(None, _invoke_local, prompt)
    if result:
        yield result

# ── LangGraph nodes ───────────────────────────────────────────────────────────

def router_node(state: ChatState) -> ChatState:
    text  = state["messages"][-1].content.strip().lower()
    words = {w.strip(string.punctuation) for w in text.split()}
    if words & GREETING_WORDS and len(words) <= GREETING_MAX_WORDS:
        return {**state, "route": "greet"}
    return {**state, "route": "rag"}


def greet_node(state: ChatState) -> ChatState:
    return {"messages": [AIMessage(content=GREETING_RESPONSE)]}


def rag_node(state: ChatState) -> ChatState:
    query = state["messages"][-1].content

    # Build conversation history from message list for query rewriting
    msgs = state.get("messages", [])
    history = [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant",
         "content": m.content}
        for m in msgs[:-1]   # exclude the current question
    ]

    t0           = time.perf_counter()
    standalone_q = rewrite_query(query, history)
    context, route, sources = retrieve_context(standalone_q)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    return {**state, "route": route, "context": context,
            "timing": {"retrieval_ms": retrieval_ms, "llm_ms": 0.0}}


def answer_node(state: ChatState) -> ChatState:
    timing = dict(state.get("timing") or {"retrieval_ms": 0.0, "llm_ms": 0.0})

    if state.get("route") == "off_topic":
        return {"messages": [AIMessage(content=OFF_TOPIC_RESPONSE)], "timing": timing}

    query   = state["messages"][-1].content
    context = state.get("context", "")
    prompt  = SALES_PROMPT.format(context=context, question=query)

    t0               = time.perf_counter()
    raw              = _invoke_llm(prompt)
    timing["llm_ms"] = (time.perf_counter() - t0) * 1000

    answer = OFF_TOPIC_RESPONSE if "OFF_TOPIC" in raw else raw
    return {"messages": [AIMessage(content=answer)], "timing": timing}

# ── Build graph ───────────────────────────────────────────────────────────────

def _route_router(state: ChatState) -> str:
    return state.get("route", "rag")

def _route_rag(state: ChatState) -> str:
    return "off_topic" if state.get("route") == "off_topic" else "answer"

def _build():
    g = StateGraph(ChatState)
    g.add_node("router", router_node)
    g.add_node("greet",  greet_node)
    g.add_node("rag",    rag_node)
    g.add_node("answer", answer_node)
    g.set_entry_point("router")
    g.add_conditional_edges("router", _route_router, {"greet": "greet", "rag": "rag"})
    g.add_conditional_edges("rag",    _route_rag,    {"answer": "answer", "off_topic": "answer"})
    g.add_edge("greet",  END)
    g.add_edge("answer", END)
    return g.compile(checkpointer=MemorySaver())

app_graph = _build()
