# Astro Power Cables — Sales Chatbot

A retrieval-augmented (RAG) sales chatbot for **Astro Power Cables**. It answers customer
questions about cable types, electrical/mechanical specs, certifications, and company
info using the company's own documents, and can email the chat transcript to sales when
a conversation goes idle or on request.

- **LLM**: Groq (cloud, fast, free tier) by default, with an offline **GPT4All** local
  model as automatic fallback if Groq errors or rate-limits.
- **Retrieval**: FAISS vector store over the company's `.txt` / `.docx` / `.xlsx` / `.csv`
  documents, embedded with `sentence-transformers/all-MiniLM-L6-v2`.
- **Orchestration**: LangGraph state machine (greeting → retrieval → answer).
- **Interfaces**: CLI (`main.py`), FastAPI server with a JSON endpoint and an SSE
  streaming endpoint, plus a ready-to-embed chat widget (`static/widget.html`).
- **Email**: auto-sends the transcript after 30 minutes of inactivity, or on demand,
  as an HTML email via SMTP.

## How it works

```
User message
   │
   ▼
router_node ── greeting? ──► greet_node ──► canned welcome message
   │ no
   ▼
rag_node
   │  1. rewrite_query()   → makes follow-up questions self-contained
   │                          (e.g. "what about its voltage?" → "what is
   │                          the NYA cable's voltage rating?")
   │  2. retrieve_context() → similarity check (off-topic filter) +
   │                          MMR search over the FAISS index (8 diverse
   │                          chunks) + exact-match injection when the
   │                          query names a specific cable size
   ▼
answer_node
   │  Builds the SALES_PROMPT with the retrieved context and asks the LLM
   │  (Groq, falling back to local GPT4All on error/rate-limit)
   ▼
Response back to user (+ conversation memory via LangGraph's MemorySaver)
```

The FastAPI layer (`app/api.py`) wraps this graph for both a non-streaming `/chat`
endpoint and a token-streaming `/chat/stream` endpoint (used by the widget), and adds
session tracking, contact capture, and the inactivity-based auto-email.

## Project structure

```
app/
  api.py            FastAPI app: routes, sessions, SSE streaming, inactivity monitor
  graph.py           LangGraph pipeline: embeddings, vectorstore, Groq/GPT4All calls
  config.py          All settings & prompts in one place (reads .env)
  email_sender.py    Builds & sends the HTML transcript email
main.py               CLI chatbot (terminal interface)
app.py                Alternate entrypoint used by hosts that expect a top-level app.py
start.sh              Install deps + launch uvicorn (used by Hugging Face Spaces)
setup.bat             One-command Windows setup (venv + pip install)
build_vectorstore.py  One-time script: reads "Astro Documents/" and builds the FAISS index
download_model.py     One-time script: downloads the local GPT4All fallback model (~4 GB)
static/
  index.html          Full chat UI
  widget.html          Embeddable widget version
requirements.txt
.env.example           Template for required environment variables
.env                    Your local secrets (gitignored, not committed)
```

## Requirements

- Python 3.10+
- A free [Groq API key](https://console.groq.com) (recommended — fast, no GPU needed)
- Windows, macOS, or Linux

## Setup

### 1. Install dependencies

**Windows** — just run the setup script from the project folder:

```bat
setup.bat
```

This creates the virtual environment at **`C:\venvs\astro`**, activates it, and installs
`requirements.txt`.

> **Why the venv lives outside the project folder:** Windows' 260-character path limit
> (`MAX_PATH`) truncates torch's deeply nested license tree when `site-packages` sits
> inside this repo, and `pip install` dies with
> `OSError: [WinError 206] The filename or extension is too long`, leaving a half-installed
> venv. Keeping it at a short path also stops OneDrive from syncing ~2 GB of packages.
> Change the location by editing `VENV_DIR` at the top of `setup.bat`.

Note that `setup.bat` activates the venv only inside its own process — that activation
does **not** carry over to your terminal. Activate it yourself before running anything
(see [Running](#running)).

**macOS/Linux** or manual setup:

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy `.env.example` to `.env` (already done in this project) and fill in your values:

| Variable              | Purpose                                                             |
|------------------------|----------------------------------------------------------------------|
| `LLM_PROVIDER`         | `groq` (recommended) or `gpt4all` (local, offline, slow on CPU)      |
| `GROQ_API_KEY`         | Your Groq API key — required when `LLM_PROVIDER=groq`               |
| `GROQ_MODEL`           | Main answer model (default `llama-3.3-70b-versatile`)               |
| `GROQ_REWRITE_MODEL`   | Small/fast model used for follow-up question rewriting               |
| `EMAIL_FROM`           | Gmail address the transcript is sent from                            |
| `EMAIL_TO`             | Address that receives transcripts                                    |
| `EMAIL_TO_2`           | Optional second recipient — leave blank to send to `EMAIL_TO` only   |
| `EMAIL_PASS`           | Gmail **App Password** (not your regular password — see below)      |
| `SMTP_HOST` / `SMTP_PORT` | SMTP server (defaults to Gmail: `smtp.gmail.com:587`)             |

> **Gmail App Password**: with 2FA enabled on the Google account, generate one at
> [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords) and use
> that as `EMAIL_PASS` — not the account's login password.

⚠️ `.env` contains real secrets and is already listed in `.gitignore` — never commit it
or paste its contents anywhere public.

### 3. Build the vector store (one-time)

Put your source documents in `Astro Documents/` (`.txt`, `.docx`, `.xlsx`, `.csv`), then:

```bash
python build_vectorstore.py
```

This creates `vectorstore/` (a FAISS index) that the chatbot searches at runtime. Re-run
it any time the source documents change.

### 4. (Optional) Download the local fallback model

Only needed if you want the offline GPT4All fallback to actually work (~4 GB download).
Without it, if Groq fails the bot just returns a short "unavailable" message instead of
a local answer.

```bash
python download_model.py
```

## Running

### Windows quick start (web UI)

Open a new terminal in the project folder and run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
C:\venvs\astro\Scripts\Activate.ps1
uvicorn app.api:app --port 8000
```

You'll see:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Then open <http://localhost:8000> in your browser.

- Leave that terminal **open** — closing it kills the server.
- Stop the server with **`Ctrl+C`**.
- Run those first two lines again in every new terminal: `Set-ExecutionPolicy` only
  applies to the current window, and so does the venv activation.
- Your prompt should read `(astro)` once activated. If it doesn't, `python` and
  `uvicorn` will resolve to the global interpreter and fail with `ModuleNotFoundError`.

### CLI chatbot

```bash
python main.py
```

Type your question, `exit`/`quit`/`bye` to end. Each reply prints retrieval/LLM timing
and which backend answered (Groq vs local).

### API server (used by the web widget)

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

Then open:
- `http://localhost:8000/` — full chat UI
- `http://localhost:8000/widget` — embeddable widget
- `http://localhost:8000/health` — health check

### API endpoints

| Method | Path                          | Description                                  |
|--------|-------------------------------|-----------------------------------------------|
| GET    | `/`                            | Serves the chat UI                            |
| GET    | `/widget`                      | Serves the embeddable widget                  |
| POST   | `/chat`                        | Non-streaming JSON reply                      |
| POST   | `/chat/stream`                 | SSE token-streaming reply                     |
| PATCH  | `/session/{id}/contact`        | Save a customer's name/email/phone             |
| POST   | `/session/{id}/email`          | Send the transcript immediately                |
| DELETE | `/session/{id}`                | Clear a session's history                      |
| GET    | `/sessions`                    | List active session IDs                        |
| GET    | `/health`                      | Liveness probe                                 |

## Deploying to Hugging Face Spaces

See [HF_DEPLOY.md](HF_DEPLOY.md) for the full step-by-step guide (creating the Space,
pushing the repo, and setting environment variables in the Space's Settings page).

## Notes

- `RELEVANCE_THRESHOLD`, `RETRIEVE_K`, chunking, prompts, and other tuning knobs all
  live in [app/config.py](app/config.py) — that's the one file to edit for behavior
  changes.
- Conversation memory is per-session (`thread_id` for the CLI, `session_id` for the API)
  and is kept in-process only (LangGraph `MemorySaver`) — it resets on restart.
- Inactivity auto-email fires after `INACTIVITY_MINUTES` (default 30) of no new
  messages in a session, once per session.
- Transcripts go to `EMAIL_TO`, plus `EMAIL_TO_2` when that is filled in. Both addresses
  receive a single message and both appear in its `To:` header, so replies keep the whole
  thread together. Blank entries are skipped and duplicates are collapsed, so setting
  `EMAIL_TO_2` to the same address as `EMAIL_TO` will not send twice. Adding a third
  recipient means appending to `EMAIL_RECIPIENTS` in [app/config.py](app/config.py).
