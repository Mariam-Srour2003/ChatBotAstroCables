#!/usr/bin/env bash
# Simple start script used by Hugging Face Spaces and similar hosts.
set -euo pipefail

echo "Installing requirements..."
python -m pip install -r requirements.txt

PORT=${PORT:-8080}
echo "Starting uvicorn on port $PORT"
exec uvicorn app.api:app --host 0.0.0.0 --port "$PORT"
