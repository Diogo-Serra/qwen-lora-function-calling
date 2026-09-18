#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    printf 'Virtual environment not found. Run: make install\n' >&2
    exit 1
fi

cd "$PROJECT_ROOT"
exec "$PYTHON" -m src "$@"