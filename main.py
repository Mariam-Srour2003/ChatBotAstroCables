"""
main.py — CLI entry point for the Astro Power Cables Sales Chatbot.

Usage:
    python main.py
"""
import time

from langchain_core.messages import HumanMessage

from app.graph import LLM_PROVIDER, app_graph

print("=" * 60)
print("  Astro Power Cables — Sales Chatbot")
print("  Type 'exit' to end the session.")
print("=" * 60 + "\n")

config = {"configurable": {"thread_id": "cli-session"}}

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBot: Thank you for contacting Astro Power Cables. Goodbye!")
        break

    if not user_input:
        continue
    if user_input.lower() in {"exit", "quit", "bye"}:
        print("Bot: Thank you for visiting Astro Power Cables. Goodbye!")
        break

    t_start = time.perf_counter()
    result  = app_graph.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config,
    )
    total_ms = (time.perf_counter() - t_start) * 1000

    print(f"Bot: {result['messages'][-1].content}\n")

    t            = result.get("timing") or {}
    retrieval_ms = t.get("retrieval_ms", 0.0)
    llm_ms       = t.get("llm_ms", 0.0)
    overhead_ms  = total_ms - retrieval_ms - llm_ms

    # Guess which backend handled this turn from the LLM time
    if LLM_PROVIDER == "groq" and llm_ms < 30_000:
        backend = "groq"
    elif llm_ms > 0:
        backend = "local"
    else:
        backend = "-"

    print(
        f"  [retrieval {retrieval_ms:6.0f} ms | "
        f"LLM {llm_ms:6.0f} ms ({backend}) | "
        f"total {total_ms:6.0f} ms]\n"
    )
