#!/bin/sh

# Default port if not provided
PORT=${PORT:-8000}

echo "Running initialization (this may take a while)..."
python - <<'PY'
from app import initialize_system
initialize_system()
PY

echo "Starting gunicorn on 0.0.0.0:$PORT"
exec gunicorn -b 0.0.0.0:${PORT} app:app --workers 1 --threads 4
