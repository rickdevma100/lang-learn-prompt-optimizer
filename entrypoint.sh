#!/bin/bash
set -e
cd /app

# Commit any PVC-mounted prompt changes so DVC can diff against them
git add -A 2>/dev/null && git commit -m "Runtime state" --allow-empty 2>/dev/null || true

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
