#!/usr/bin/env python3
"""Bootstrap, check, and run the LLM Topology Lab.

System assumptions: python3 and uv. Everything else is provisioned by uv.
"""

from __future__ import annotations

import argparse
import os
import shutil
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
    return str(local) if local.exists() else None


def _uv_missing() -> None:
    raise SystemExit(
        "uv ist nicht installiert. uv wird zum Provisionieren der Umgebung benoetigt.\n\n"
        "Installiere uv mit einem der folgenden Befehle:\n"
        "  curl -LsSf https://astral.sh/uv/install.sh | sh\n"
        "  python3 -m pip install --user uv\n"
        "  brew install uv\n\n"
        "Danach erneut starten:\n"
        "  python3 app.py\n"
    )


def _reexec_inside_uv() -> None:
    uv = _find_uv()
    if not uv:
        _uv_missing()
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
    parser = argparse.ArgumentParser(
        description="LLM Topology Lab: GUI, headless training, or doctor.",
        epilog=(
            "Beispiele:\n"
            "  python3 app.py --doctor\n"
            "  python3 app.py\n"
            "  python3 app.py --config experiment.yaml --headless\n"
            "  python3 app.py --config experiment.yaml --headless --output runs/hpc\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="Web bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Web port (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab")
    parser.add_argument("--doctor", action="store_true", help="Run startup checks and exit")
    parser.add_argument("--config", help="YAML experiment config for headless/HPC mode")
    parser.add_argument(
        "--headless", action="store_true", help="Run without GUI (requires --config)"
    )
    parser.add_argument("--output", help="Override output directory for headless mode")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], help="Override device")
    return parser.parse_args()


def _open_browser_later(host: str, port: int) -> None:
    def open_it() -> None:
        time.sleep(1.2)
        target_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
        webbrowser.open(f"http://{target_host}:{port}")

    threading.Thread(target=open_it, daemon=True).start()


def _run_server(args: argparse.Namespace) -> None:
    try:
        import uvicorn

        from llm_topology.web.main import app
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Eine Python-Abhaengigkeit fehlt. Fuehre aus:\n"
            f"  uv sync --project {ROOT}\n"
            f"  python3 {Path(__file__).name}\n"
            f"Originalfehler: {exc}"
        ) from exc

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

    args = _parse_args()
    if args.headless and not args.config:
        raise SystemExit(
            "--headless braucht --config. Beispiel: python3 app.py --config experiment.yaml --headless"
        )

    from llm_topology.preflight import PreflightError, collect_checks, ensure_ready, render_checks

    mode = "doctor" if args.doctor else "cli" if args.config else "server"
    checks = collect_checks(mode=mode, config_path=args.config, host=args.host, port=args.port)
    render_checks(checks, title="LLM Topology Lab · preflight")
    if args.doctor:
        raise SystemExit(1 if any(check.status == "error" for check in checks) else 0)
    try:
        ensure_ready(checks)
    except PreflightError as exc:
        raise SystemExit(str(exc)) from exc

    if args.config:
        from llm_topology.cli import run_experiment

        try:
            run_experiment(
                args.config,
                output_override=args.output,
                device_override=args.device,
            )
        except KeyboardInterrupt:
            raise SystemExit(130) from None
        except SystemExit:
            raise
        except Exception as exc:
            raise SystemExit(
                f"Headless-Lauf fehlgeschlagen: {type(exc).__name__}: {exc}\n"
                "Pruefe die Config, den Dataset-Import und den Output-Pfad. "
                "Mit `python3 app.py --doctor` siehst du die Systemchecks."
            ) from exc
        return

    _run_server(args)


if __name__ == "__main__":
    main()
