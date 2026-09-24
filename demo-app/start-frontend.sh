#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/frontend" && pwd)"
cd "$ROOT"
if [[ ! -d node_modules ]]; then npm ci; fi
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
