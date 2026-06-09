"""
Entrypoint for hosting on Hugging Face Spaces or other hosts that expect
a top-level `app.py`. This simply starts the existing FastAPI app.

Run: `python app.py` or the `start.sh` script which forwards `$PORT`.
"""
import os
import uvicorn


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app.api:app", host="0.0.0.0", port=port, log_level="info")
