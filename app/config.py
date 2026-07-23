"""
app/config.py — all settings in one place. Edit here, touch nothing else.

LLM_PROVIDER=groq      → fast cloud API (free tier, ~2 s)
LLM_PROVIDER=gpt4all   → local model, offline, slow on CPU
"""
import os as _os
from dotenv import load_dotenv
load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH       = "./models/gpt4all-falcon-newbpe-q4_0.gguf"
VECTORSTORE_PATH = "./vectorstore/"
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
DOCS_DIR         = "./Astro Documents"

# ── LLM provider ──────────────────────────────────────────────────────────────
LLM_PROVIDER = _os.getenv("LLM_PROVIDER", "gpt4all").lower()

GROQ_API_KEY       = _os.getenv("GROQ_API_KEY", "")
GROQ_MODEL         = _os.getenv("GROQ_MODEL",         "llama-3.3-70b-versatile")
GROQ_REWRITE_MODEL = _os.getenv("GROQ_REWRITE_MODEL", "llama-3.1-8b-instant")

# ── Retrieval ─────────────────────────────────────────────────────────────────
RELEVANCE_THRESHOLD = 1.45
RETRIEVE_K          = 8      # MMR picks 8 diverse chunks (more recall for specific data)
MMR_FETCH_K         = 25     # candidate pool for MMR

CHUNK_SIZE    = 600
CHUNK_OVERLAP = 100

# ── Device ───────────────────────────────────────────────────────────────────
def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"

import os as _os2
COMPUTE_DEVICE = _detect_device()
LLM_DEVICE     = "gpu" if COMPUTE_DEVICE == "cuda" else "cpu"

# ── GPT4All settings (only when LLM_PROVIDER=gpt4all) ────────────────────────
LLM_MAX_TOKENS = 500
LLM_N_BATCH    = 256
LLM_N_THREADS  = 4 if COMPUTE_DEVICE == "cuda" else (_os2.cpu_count() or 4)
LLM_STOP       = ["\n[Q]", "\n[CONTEXT]", "\nCustomer", "\nQuestion:"]

# ── Greeting detection ────────────────────────────────────────────────────────
GREETING_WORDS: set[str] = {
    "hi", "hello", "hey", "good", "morning", "afternoon", "evening",
    "salam", "marhaba", "bonjour", "hola", "greetings", "howdy", "welcome",
}
GREETING_MAX_WORDS = 6

# ── Static responses ──────────────────────────────────────────────────────────
GREETING_RESPONSE = (
    "Welcome to Astro Power Cables! I'm your dedicated sales representative, "
    "happy to help you find the perfect cable solution.\n\n"
    "I can assist you with:\n"
    "  - Cable types and technical specifications\n"
    "  - Electrical and mechanical properties\n"
    "  - Company background, certifications, and quality standards\n"
    "  - Industries we serve and key projects\n"
    "  - Factory and manufacturing capabilities\n\n"
    "What can I help you with today?"
)

OFF_TOPIC_RESPONSE = (
    "Thank you for reaching out! My expertise is focused on Astro Power Cables "
    "and our product range, so I'm not able to assist with topics outside that scope.\n\n"
    "Feel free to ask me about:\n"
    "  - Our cable product catalogue\n"
    "  - Technical data (electrical & mechanical)\n"
    "  - ISO certifications and quality standards\n"
    "  - Company information and our factory"
)

# ── Email ─────────────────────────────────────────────────────────────────────
EMAIL_FROM          = _os.getenv("EMAIL_FROM", "mariammsrourr2020@gmail.com")
EMAIL_TO            = _os.getenv("EMAIL_TO",   "mariammsrourr2020@gmail.com")
# Optional second recipient. Leave blank to send to EMAIL_TO only; fill it in
# and transcripts go to both addresses.
EMAIL_TO_2          = _os.getenv("EMAIL_TO_2", "").strip()
EMAIL_PASS          = _os.getenv("EMAIL_PASS", "")

# Every address a transcript goes to, blanks dropped and duplicates removed
# so setting EMAIL_TO_2 to the same address as EMAIL_TO cannot double-send.
EMAIL_RECIPIENTS    = list(dict.fromkeys(
    a for a in (EMAIL_TO.strip(), EMAIL_TO_2) if a
))
SMTP_HOST           = _os.getenv("SMTP_HOST",  "smtp.gmail.com")
SMTP_PORT           = int(_os.getenv("SMTP_PORT", "587"))
INACTIVITY_MINUTES  = 30   # auto-email session after this many minutes of silence

# ── Prompts ───────────────────────────────────────────────────────────────────

# Used to make a follow-up question self-contained before retrieval
REWRITE_PROMPT = (
    "You are a query rewriter. Given a chat history and a follow-up question, "
    "rewrite the question to be fully self-contained.\n"
    "- Replace vague pronouns (it, that, this, they, its) with the actual cable/product name from history.\n"
    "- If the question is already self-contained OR is a greeting, return it UNCHANGED.\n"
    "- Return ONLY the rewritten question — no explanation, no quotes, nothing else.\n\n"
    "Chat History:\n{history}\n\n"
    "Follow-up: {question}\n"
    "Rewritten:"
)

SALES_PROMPT = (
    "You are a professional sales representative at Astro Power Cables, "
    "an ISO-certified cable manufacturer based in Lebanon.\n"
    "Rules:\n"
    "1. Answer using ONLY the [CONTEXT] below — never invent facts.\n"
    "2. Be professional, concise, and highlight product strengths.\n"
    "3. If asked for contact info, include the phone numbers and email found in context.\n"
    "4. If asked for pricing, say pricing varies by order size and invite them to contact the sales team.\n"
    "5. Naturally cite the source when useful (e.g. 'According to our NYA datasheet...').\n"
    "6. If the context partially answers the question, give what you can and invite them to contact sales for more detail.\n"
    "7. Only if the context has ABSOLUTELY NO relevance to the question (e.g. cooking, politics, sports), "
    "reply with exactly: OFF_TOPIC\n\n"
    "[CONTEXT]\n{context}\n\n"
    "[Q] {question}\n\n"
    "[A]"
)
