from __future__ import annotations

import runpy
import sys
from pathlib import Path

import site

# Load Uvicorn and its dependencies only after Conda's site-packages take
# precedence over user-level packages with conflicting versions.
user_site = str(site.getusersitepackages()).lower()
user_paths = [path for path in sys.path if path.lower().startswith(user_site)]
for path in user_paths:
    sys.path.remove(path)
sys.path.extend(user_paths)

repo_root = Path(__file__).resolve().parents[1]
sys.argv = [
    "uvicorn",
    "main:app",
    "--app-dir",
    str(repo_root / "demo-app" / "backend"),
    "--host",
    "127.0.0.1",
    "--port",
    "18000",
    "--workers",
    "1",
]
runpy.run_module("uvicorn", run_name="__main__")
