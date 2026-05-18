#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd)/src"
python scripts/evaluate.py
uvicorn rag_service.main:app --reload --host 0.0.0.0 --port 8000
