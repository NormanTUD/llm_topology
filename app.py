#!/usr/bin/env python3
"""Bootstrap and start the LLM Topology Lab.

Run `python3 app.py` (or `uv run app.py`) and uv will provision the local
environment on the first start. The process re-executes itself once inside
the uv environment; the environment flag prevents recursion.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BOOTSTRAP_FLAG = "LLM_TOPOLOGY_BOOTSTRAPPED"


def _find_uv() -> str | None:
    found = shutil.which("uv")
    if found:
        return found
    local = Path.home() / ".local" / "bin" / "uv"
    if local.exists():
        return str(local)
    return None


def _ensure_uv() -> str:
    uv = _find_uv()
    if uv:
        return uv
    print("uv was not found. Trying to install it with pip --user ...", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--user", "uv"],
        check=False,
        capture_output=False,
    )
    uv = _find_uv()
    if result.returncode == 0 and uv:
        return uv
    raise SystemExit(
        "uv is required but could not be installed.\n"
        "Install it manually, then rerun: python3 app.py\n"
        "  https://docs.astral.sh/uv/getting-started/installation/"
    )


def _reexec_inside_uv() -> None:
    uv = _ensure_uv()
    env = os.environ.copy()
    env[BOOTSTRAP_FLAG] = "1"
    command = [
        uv,
        "run",
        "--project",
        str(ROOT),
        "python",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    os.execvpe(uv, command, env)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LLM Topology Lab locally.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab")
    return parser.parse_args()


def _open_browser_later(host: str, port: int) -> None:
    def open_it() -> None:
        time.sleep(1.2)
        target_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
        webbrowser.open(f"http://{target_host}:{port}")

    threading.Thread(target=open_it, daemon=True).start()


def _run_server() -> None:
    try:
        import uvicorn

        from llm_topology.web.main import app
    except ImportError as exc:  # pragma: no cover - only hit with a broken environment
        raise SystemExit(
            "The environment is missing a dependency. Try:\n"
            f"  uv sync --project {ROOT}\n"
            f"  python3 {Path(__file__).name}"
        ) from exc

    args = _parse_args()
    url = f"http://{args.host}:{args.port}"
    print("", flush=True)
    print("  LLM Topology Lab", flush=True)
    print(f"  {url}", flush=True)
    print("  Press Ctrl+C to stop.", flush=True)
    print("", flush=True)
    if not args.no_browser:
        _open_browser_later(args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def main() -> None:
    if os.environ.get(BOOTSTRAP_FLAG) != "1":
        _reexec_inside_uv()
        return
    _run_server()


if __name__ == "__main__":
    main()
