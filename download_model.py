"""
download_model.py
One-time script to download the GPT4All Falcon model (~4 GB).

Run before the first chatbot launch:
    python download_model.py
"""
import os
import requests
from tqdm import tqdm

MODEL_URL  = "https://gpt4all.io/models/gguf/gpt4all-falcon-newbpe-q4_0.gguf"
MODEL_PATH = "models/gpt4all-falcon-newbpe-q4_0.gguf"

os.makedirs("models", exist_ok=True)

if os.path.exists(MODEL_PATH):
    print(f"Model already exists at: {MODEL_PATH}")
else:
    print(f"Downloading model to: {MODEL_PATH}")
    response   = requests.get(MODEL_URL, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(MODEL_PATH, "wb") as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(1024):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"\nModel saved to: {MODEL_PATH}")
