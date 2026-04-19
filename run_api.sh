#!/bin/bash
# ============================================
# Ergonomic Chair AI Training - API Server Launcher
# Version: 1.0.1
# ============================================

set -e

echo "============================================"
echo "  Ergonomic Chair AI Training - API Server"
echo "  Version: 1.0.1"
echo "============================================"

cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "[ERROR] Virtual environment not found!"
    echo "Please run setup first or activate venv manually."
    exit 1
fi

source venv/bin/activate

echo "[INFO] Checking dependencies..."
python -c "import fastapi; import uvicorn" 2>/dev/null || {
    echo "[WARN] Installing missing API dependencies..."
    pip install fastapi uvicorn pydantic -i https://pypi.org/simple/ --trusted-host pypi.org
}

HOST="${API_HOST:-0.0.0.0}"
PORT="${API_PORT:-8000}"
MODEL_PATH="${CHAIR_MODEL_PATH:-}"

if [ -n "$MODEL_PATH" ] && [ -f "$MODEL_PATH" ]; then
    export CHAIR_MODEL_PATH="$MODEL_PATH"
    echo "[INFO] Pre-loading model from: $MODEL_PATH"
fi

echo ""
echo "[INFO] Starting API server..."
echo "       Host: $HOST"
echo "       Port: $PORT"
echo "       Docs: http://$HOST:$PORT/docs"
echo ""

exec uvicorn api.main:app --host "$HOST" --port "$PORT" --reload
