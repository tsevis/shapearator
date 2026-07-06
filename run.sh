#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Choose a Python interpreter: explicit override, project venv, or system
# python3 — whichever is found first. Set PYTHON_BIN to force a specific one.
if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN}" ]]; then
  PY="$PYTHON_BIN"
elif [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
  PY="$SCRIPT_DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  echo "[Shapearator Launcher] Error: no Python interpreter found. Install Python 3.10+." >&2
  exit 1
fi

# Install dependencies on first run (idempotent and quiet afterwards).
if ! "$PY" -c "import cv2, numpy, PIL, requests, huggingface_hub" >/dev/null 2>&1; then
  echo "[Shapearator Launcher] Installing Python dependencies..."
  "$PY" -m pip install -r "$SCRIPT_DIR/requirements.txt"
fi

exec "$PY" "$SCRIPT_DIR/main.py" "$@"
