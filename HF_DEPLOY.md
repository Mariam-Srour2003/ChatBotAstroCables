# Deploy to Hugging Face Spaces (free for public Spaces)

Follow these steps to deploy the backend to a Hugging Face Space. This guide assumes
you already have a Hugging Face account.

1) Create a new Space on Hugging Face
- In the web UI choose: *New Space* → give it a name → Choose *Other* as SDK (or "Gradio/Streamlit/Other" if Other not present) → Visibility: Public

2) Locally prepare the repo and push
Open a shell (PowerShell on Windows) in this project folder and run:

```powershell
git init
git add .
git commit -m "Initial HF Space deploy"
# Replace <USERNAME> and <SPACE-NAME> with your values
git remote add origin https://huggingface.co/spaces/<USERNAME>/<SPACE-NAME>
git push origin main
```

3) Environment variables to set in the Space settings (Settings → Variables):
- `LLM_PROVIDER=groq`            # if using Groq as primary LLM
- `GROQ_API_KEY=...`            # your Groq API key (if using Groq)
- `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`
- `VECTORSTORE_PATH=vectorstore` # ensure `vectorstore/index.faiss` is in the repo
- Optionally: `MODEL_PATH=` leave empty to disable local GPT4All (recommended)

Notes:
- The repo already includes `vectorstore/index.faiss` and `static/` assets — keep them in the Space repo so the app can load them.
- The local GPT4All fallback will only load if `MODEL_PATH` points to a model file present in the Space; leave `MODEL_PATH` unset to avoid the ~4GB local model.

4) Start command
- Hugging Face will usually detect `app.py` and run it automatically. If not, set the start command to:

```bash
bash start.sh
```

5) After push, open the Space URL and test `/health` and `/widget`
- `https://huggingface.co/spaces/<USERNAME>/<SPACE-NAME>/` serves the static UI
- Health check: `https://huggingface.co/spaces/<USERNAME>/<SPACE-NAME>/api/health`
