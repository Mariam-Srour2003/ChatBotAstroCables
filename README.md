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

### One bot, two channels

The website widget and WhatsApp are two front doors onto the same brain:

```
   Website widget                     WhatsApp
   (SSE streaming)                (Meta Cloud API)
         │                               │
         └──────────► app/engine.py ◄────┘
                      greeting → rewrite → retrieve →
                      off-topic filter → SALES_PROMPT
                              │
                      app/graph.py (FAISS + Groq/GPT4All)

              both store history in app/sessions.py
              → same 30-minute auto-email of transcripts
```

Anything you change — the prompts in `config.py`, the documents behind the vectorstore,
the off-topic rules — applies to both channels at once. There is no second copy to keep
in sync.

## Project structure

```
app/
  api.py            FastAPI app: routes, SSE streaming, mounts the WhatsApp router
  engine.py          Shared brain: what to answer (used by BOTH channels)
  sessions.py        Shared conversation state + inactivity auto-email monitor
  whatsapp.py        WhatsApp Cloud API channel: webhook, signing, sending
  graph.py           LangGraph pipeline: embeddings, vectorstore, Groq/GPT4All calls
  config.py          All settings & prompts in one place (reads .env)
  email_sender.py    Builds & sends the HTML transcript email
whatsapp_test.py      Check WhatsApp credentials / send a test message / test a reply
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
| GET    | `/webhook/whatsapp`            | Meta's webhook verification handshake          |
| POST   | `/webhook/whatsapp`            | Inbound WhatsApp messages                      |
| GET    | `/whatsapp/status`             | Shows which WhatsApp credentials are set       |

## WhatsApp channel

The bot answers on WhatsApp through the **Meta WhatsApp Cloud API** — Meta's own
official API, no reseller in between. Same answers, same documents, same transcript
emails as the website.

### Why Meta and not something else

| Option | Cost | Verdict |
|--------|------|---------|
| **Meta Cloud API** | Free to use. Customer-initiated ("service") conversations are free, and this bot only ever *replies* — it never starts a chat. | **Use this.** |
| Twilio / 360dialog / Wati | Meta's fees **plus** the reseller's monthly fee | Easier signup, but you pay monthly for a wrapper around the same API. |
| `whatsapp-web.js`, Baileys (unofficial) | Free | **Avoid.** These drive a logged-in WhatsApp session and break Meta's terms — numbers get banned, usually right when the bot starts getting real traffic. Not worth risking the company number. |

Meta's fees only apply to *business-initiated* template messages (marketing, utility,
authentication). Replying to someone who messaged you first, within the 24-hour
"customer service window", is free and unlimited — which is exactly what this bot does.
Meta has changed this pricing model more than once, so confirm on
[their pricing page](https://developers.facebook.com/docs/whatsapp/pricing) before
launch.

> **The 24-hour window matters.** The bot can only reply to someone within 24 hours of
> their last message. If a customer goes quiet for a day, the bot can no longer message
> them out of the blue — that would require a pre-approved paid template. For a
> reply-only sales bot this is never a problem.

### What to ask the client for

Test numbers need nothing from the client — Meta gives you one instantly. Everything
below is for **production**, when the bot moves onto the company's own number.

**1. A phone number for WhatsApp Business**
- Must be able to receive an **SMS or a phone call** for the one-time verification code.
- Must **not already be in use on WhatsApp** (neither the normal app nor WhatsApp
  Business). If the number is currently on WhatsApp, they have to delete that account
  first — *deleting it erases that account's chat history*, so warn them before they do
  it. A brand-new SIM or a landline that can take a voice call avoids the whole problem.
- Must stay with the company — it becomes the customer-facing sales number.

**2. Access to the Meta Business account**
- The **Meta Business Portfolio** (business.facebook.com) for Astro Power Cables, or
  permission to create one.
- They add you as an **admin** — ask them to go to Business settings → People → Invite,
  and give you the *Full control / admin* role. Without this you cannot create the
  WhatsApp Business Account or generate a permanent token.
- The Facebook account they use must have two-factor authentication enabled (Meta
  requires it for anyone touching WhatsApp API assets).

**3. Business verification documents** (Meta reviews these; usually 1–3 business days,
sometimes longer)
- Official business registration / commercial register extract
- A utility bill or bank statement showing the **business name and address**, dated
  within the last 3 months
- The **company website** (must show the same business name, and preferably the phone
  number and address that match the documents)
- A **business email** on the company domain (e.g. `info@astro-lb.com`, not Gmail)

**4. Decisions to confirm with them**
- **Display name** — the name customers see, e.g. "Astro Power Cables". Meta checks it
  matches the business, so avoid slogans or extra words.
- **Profile details** — logo image, business description, address, email, website. These
  fill in the WhatsApp business profile.
- **Where transcripts go** — the sales inbox addresses for `EMAIL_TO` / `EMAIL_TO_2`.
  WhatsApp leads arrive there automatically with the customer's phone number attached.
- **Handover to a human** — decide what happens when a customer wants a real
  salesperson. Right now the bot answers everything; see *Going further* below.

### Testing (free, no client input, ~15 minutes)

Meta gives every developer app a **free test number**. It can message up to **5
recipient numbers** that you verify yourself, which is plenty to demo the bot.

**Step 1 — Create the app**
1. Go to [developers.facebook.com/apps](https://developers.facebook.com/apps) and log in.
2. **Create app** → use case **Other** → type **Business** → name it e.g.
   `Astro Cables Bot`.
3. On the app dashboard find **WhatsApp** → **Set up**.

**Step 2 — Collect the credentials**

On **WhatsApp → API Setup** you will see:
- a **temporary access token** (top of the page) → this is `WHATSAPP_TOKEN`
- **Phone number ID** under the "From" dropdown → this is `WHATSAPP_PHONE_NUMBER_ID`
  (the long number, *not* the +1 555... phone number)

Under "To", click **Manage phone number list** and add your own mobile number — Meta
sends it a code to confirm. Only numbers on that list can talk to the test bot.

> The temporary token **expires after 24 hours**. That is fine for testing; you will
> swap it for a permanent one before launch (see *Production* below).

**Step 3 — Fill in `.env`**

```bash
WHATSAPP_TOKEN=EAAG...the temporary token...
WHATSAPP_PHONE_NUMBER_ID=123456789012345
WHATSAPP_VERIFY_TOKEN=any_random_string_you_invent
WHATSAPP_APP_SECRET=
```

`WHATSAPP_VERIFY_TOKEN` is not issued by Meta — you make it up, and type the same value
into Meta's form in step 5. Leave `WHATSAPP_APP_SECRET` blank while testing.

Check it worked:

```bash
python whatsapp_test.py check
```

**Step 4 — Expose your local server to the internet**

Meta has to reach your machine over **HTTPS**, so `localhost:8000` is not enough. Start
the server:

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Then, in a second terminal, open a tunnel with
[cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)
(free, no account needed):

```bash
cloudflared tunnel --url http://localhost:8000
```

It prints a public URL like `https://random-words-here.trycloudflare.com`. (ngrok works
equally well if you already have it.) Keep this terminal open — the URL changes every
time you restart the tunnel, and you would have to update Meta again.

**Step 5 — Point Meta at your webhook**

In **WhatsApp → Configuration → Edit webhook**:
- **Callback URL**: `https://your-tunnel-url.trycloudflare.com/webhook/whatsapp`
- **Verify token**: the exact string you put in `WHATSAPP_VERIFY_TOKEN`
- Click **Verify and save** — your terminal should print
  `[whatsapp] Webhook verified by Meta.`

Then click **Manage** next to Webhook fields and **Subscribe** to **`messages`**. This
is the step people forget; without it Meta never sends anything and the bot stays
silent.

**Step 5b — Subscribe your app to the WhatsApp account (do not skip)**

There is a *second* subscription the dashboard does not show you. A fresh test account
is wired to Meta's own demo app (`WA DevX Webhook Events 1P App`), not to yours, so
every message is delivered there and your webhook receives nothing.

Put the **WhatsApp Business Account ID** (API Setup page, below the phone number) into
`WHATSAPP_WABA_ID` in `.env`, then:

```bash
python whatsapp_test.py subs        # who currently receives the webhooks
python whatsapp_test.py subscribe   # link this WABA to your app
```

`subs` should end with `Your app IS subscribed`. Until it does, the bot cannot receive
anything — no matter how green the dashboard looks.

**Step 6 — Talk to it**

Send "hi" from the phone you verified to the test number shown on the API Setup page.
You should get the standard Astro welcome, then real answers about cables.

```bash
python whatsapp_test.py ask "what is NYA cable used for"   # test replies, no Meta needed
python whatsapp_test.py send 9613123456 "test message"     # send to a verified number
```

### Production

**1. Add the real phone number**
WhatsApp → API Setup → **Add phone number**. Enter the display name and business
details, then verify the number by SMS or voice call. Its new **Phone number ID**
replaces the test one in `.env`.

**2. Get a permanent access token**
The temporary token dies daily. In
[business.facebook.com](https://business.facebook.com) → **Business settings** →
**Users → System users**:
- **Add** a system user, role **Admin**, name it e.g. `astro-bot`
- **Add assets** → your app and your WhatsApp Business Account → full control
- **Generate new token** → select your app → tick **`whatsapp_business_messaging`** and
  **`whatsapp_business_management`** → set expiry **Never**
- Copy it into `WHATSAPP_TOKEN`. **It is shown once** — losing it means generating a new
  one.

**3. Turn on signature verification**
Copy **App settings → Basic → App secret** into `WHATSAPP_APP_SECRET`. Without it,
anyone who discovers your webhook URL can feed the bot fake messages. With it set, the
bot rejects any callback not signed by Meta.

**4. Host it somewhere permanent**
The tunnel is only for development. Deploy where the server has a stable HTTPS URL
(Hugging Face Spaces — see [HF_DEPLOY.md](HF_DEPLOY.md) — Railway, Render, a VPS…), set
the same environment variables there, and update the webhook Callback URL to
`https://your-domain/webhook/whatsapp`.

**5. Complete business verification and go live**
In the app dashboard, submit business verification with the documents listed above,
then switch the app from **Development** to **Live** (toggle at the top of the
dashboard). Until the app is Live, only your verified test numbers can reach the bot.

New numbers start at a **1,000-conversation/day** limit, which rises automatically as
you send more messages at good quality.

### How the WhatsApp side behaves

- Each phone number is one conversation, remembering the last 10 exchanges — so
  follow-ups like *"and what's its voltage rating?"* work exactly as on the website.
- Replies are marked read and show a typing indicator while the LLM works.
- Markdown is converted to WhatsApp's own formatting (`**bold**` → `*bold*`, `-` bullets
  → `•`), and answers over 3,800 characters are split across messages at paragraph
  breaks.
- Non-text messages (photos, voice notes, PDFs) get a polite "text only please" reply.
- A customer sending **reset**, **restart**, **clear**, or **new chat** wipes their
  history and starts fresh.
- After 30 minutes of silence, the transcript is emailed to the sales inbox with the
  customer's WhatsApp name and phone number attached — the same auto-email the website
  already uses.
- Repeat deliveries from Meta are de-duplicated by message id, so a retry never answers
  the customer twice.

### Going further

Things worth adding once the basics are live, roughly in order of value:

- **Human handover** — a keyword like "agent" that pauses the bot for that number and
  emails sales immediately, so a person can take over the chat.
- **Persistent sessions** — history currently lives in memory and resets when the server
  restarts. SQLite or Redis keyed by `wa:<number>` would survive deploys.
- **Arabic** — Groq's Llama models handle Arabic well; the constraint is that the source
  documents in the vectorstore are English, so answers cite English specs. Adding
  "reply in the customer's language" to `SALES_PROMPT` is a one-line experiment.
- **Rate limiting** per phone number, if the bot ever attracts spam.

### WhatsApp troubleshooting

| Symptom | Cause |
|---------|-------|
| "Verify and save" fails in Meta | Server not running, tunnel URL wrong or expired, or `WHATSAPP_VERIFY_TOKEN` doesn't match character-for-character |
| Webhook verified, but messages never arrive | Either you didn't **subscribe to the `messages` field** (Configuration → Webhook fields), or your app isn't subscribed to the WABA — run `python whatsapp_test.py subs` |
| Everything green, sending works, still no inbound | The WABA is subscribed to Meta's demo app instead of yours. `python whatsapp_test.py subscribe` fixes it. This one is invisible in the dashboard. |
| Bot receives messages but sending fails | Token expired (temporary tokens last 24 h) — run `python whatsapp_test.py check` |
| `(#131030) recipient not in allowed list` | App is still in Development mode; add the number under "Manage phone number list" |
| Nothing happens, no log lines at all | Meta is hitting a stale tunnel URL — restart the tunnel and update the Callback URL |
| `403 Invalid signature` in the log | `WHATSAPP_APP_SECRET` doesn't match this app's secret; clear it to disable the check while debugging |

## Deploying to Hugging Face Spaces

See [HF_DEPLOY.md](HF_DEPLOY.md) for the full step-by-step guide (creating the Space,
pushing the repo, and setting environment variables in the Space's Settings page).

## Notes

- `RELEVANCE_THRESHOLD`, `RETRIEVE_K`, chunking, prompts, and other tuning knobs all
  live in [app/config.py](app/config.py) — that's the one file to edit for behavior
  changes.
- Conversation memory is per-session (`thread_id` for the CLI, `session_id` for the API,
  `wa:<phone number>` for WhatsApp) and is kept in-process only — it resets on restart.
- Inactivity auto-email fires after `INACTIVITY_MINUTES` (default 30) of no new
  messages in a session, once per session — for website and WhatsApp chats alike.
- The website and WhatsApp share `app/engine.py`, so prompt or document changes affect
  both. If you add a channel later, call `engine.answer()` and it inherits everything.
- Transcripts go to `EMAIL_TO`, plus `EMAIL_TO_2` when that is filled in. Both addresses
  receive a single message and both appear in its `To:` header, so replies keep the whole
  thread together. Blank entries are skipped and duplicates are collapsed, so setting
  `EMAIL_TO_2` to the same address as `EMAIL_TO` will not send twice. Adding a third
  recipient means appending to `EMAIL_RECIPIENTS` in [app/config.py](app/config.py).
