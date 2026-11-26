from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.table import Table

Status = Literal["ok", "warn", "error"]


@dataclass
class Check:
    name: str
    status: Status
    message: str
    fix: str = ""


class PreflightError(RuntimeError):
    pass


def _status_icon(status: Status) -> str:
    return {"ok": "[green]OK[/green]", "warn": "[yellow]WARN[/yellow]", "error": "[red]FAIL[/red]"}[
        status
    ]


def _safe_import(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False


def _disk_free(path: Path) -> int:
    try:
        return shutil.disk_usage(path).free
    except Exception:
        return 0


def _meminfo() -> tuple[int, int]:
    total = available = 0
    path = Path("/proc/meminfo")
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("MemTotal:"):
                total = int(line.split()[1]) * 1024
            elif line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
    if not total:
        try:
            total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except Exception:
            pass
    return total, available


def _gpu_info() -> tuple[bool, list[str]]:
    if not _safe_import("torch"):
        return False, []
    try:
        import torch

        if not torch.cuda.is_available():
            return False, []
        names = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            names.append(f"{props.name} ({props.total_memory / 1024**3:.1f} GiB)")
        return True, names
    except Exception:
        return False, []


def _port_free(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def collect_checks(
    *,
    mode: Literal["server", "cli", "doctor"] = "server",
    config_path: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> list[Check]:
    checks: list[Check] = []
    py = sys.version_info
    if py < (3, 11):
        checks.append(
            Check(
                "Python",
                "error",
                f"Python {py.major}.{py.minor} ist zu alt.",
                "Installiere Python >= 3.11 oder nutze: uv python install 3.13",
            )
        )
    else:
        checks.append(Check("Python", "ok", f"{platform.python_version()} ({sys.executable})"))

    uv = shutil.which("uv")
    checks.append(
        Check(
            "uv",
            "ok" if uv else "error",
            uv or "uv wurde nicht gefunden.",
            "" if uv else "Installiere uv: curl -LsSf https://astral.sh/uv/install.sh | sh",
        )
    )

    required = {
        "numpy": "Numerik",
        "scipy": "Wissenschaftliche Berechnungen",
        "torch": "Custom Transformer",
        "gudhi": "Alpha-Komplex / PH",
        "ripser": "Vietoris-Rips",
        "fastapi": "Web-GUI",
        "uvicorn": "Web-Server",
        "rich": "CLI",
        "yaml": "YAML-Configs",
    }
    missing = [
        f"{module} ({label})" for module, label in required.items() if not _safe_import(module)
    ]
    checks.append(
        Check(
            "Python-Pakete",
            "error" if missing else "ok",
            "Alle benoetigten Pakete vorhanden." if not missing else "Fehlt: " + ", ".join(missing),
            "" if not missing else "Fuehre aus: uv sync --extra dev",
        )
    )

    total, available = _meminfo()
    if available == 0:
        checks.append(
            Check(
                "RAM",
                "warn",
                "RAM-Verfuegbarkeit konnte nicht ermittelt werden.",
                "Pruefe /proc/meminfo oder cgroup-Limits.",
            )
        )
    elif available < 512 * 1024**2:
        checks.append(
            Check(
                "RAM",
                "error",
                f"Nur {available / 1024**2:.0f} MiB RAM verfuegbar.",
                "Beende andere Prozesse oder reduziere Batch/Kontext.",
            )
        )
    elif available < 2 * 1024**3:
        checks.append(
            Check(
                "RAM",
                "warn",
                f"{available / 1024**3:.1f} GiB verfuegbar.",
                "Reduziere Modell/Batch oder nutze einen groesseren Knoten.",
            )
        )
    else:
        checks.append(
            Check(
                "RAM",
                "ok",
                f"{available / 1024**3:.1f} GiB verfuegbar von {total / 1024**3:.1f} GiB",
            )
        )

    free = _disk_free(Path.cwd())
    if free < 512 * 1024**2:
        checks.append(
            Check(
                "Speicherplatz",
                "error",
                f"Nur {free / 1024**2:.0f} MiB frei.",
                "Gib Speicher frei oder starte in einem anderen Verzeichnis.",
            )
        )
    elif free < 2 * 1024**3:
        checks.append(
            Check(
                "Speicherplatz",
                "warn",
                f"{free / 1024**3:.1f} GiB frei.",
                "Checkpoints/Datasets brauchen Platz.",
            )
        )
    else:
        checks.append(Check("Speicherplatz", "ok", f"{free / 1024**3:.1f} GiB frei"))

    has_gpu, gpu_names = _gpu_info()
    checks.append(
        Check(
            "GPU",
            "ok" if has_gpu else "warn",
            "GPU: " + ", ".join(gpu_names)
            if has_gpu
            else "Keine CUDA-GPU erkannt; Training laeuft auf CPU.",
            ""
            if has_gpu
            else "Das ist okay. Fuer GPU: CUDA-Treiber installieren und torch mit CUDA verwenden.",
        )
    )

    for directory in ("runs", "datasets", "state"):
        path = Path.cwd() / directory
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            checks.append(Check(directory, "ok", f"{path} ist schreibbar"))
        except Exception as exc:
            checks.append(
                Check(
                    directory,
                    "error",
                    f"{path} ist nicht schreibbar: {exc}",
                    f"chmod u+w {path} oder anderes Arbeitsverzeichnis waehlen.",
                )
            )

    if mode == "server":
        checks.append(
            Check(
                f"Port {port}",
                "ok" if _port_free(host, port) else "error",
                f"{host}:{port} ist frei"
                if _port_free(host, port)
                else f"{host}:{port} ist bereits belegt.",
                f"Starte mit einem anderen Port: python3 app.py --port {port + 1}",
            )
        )
        checks.append(
            Check(
                "Browser",
                "ok" if shutil.which("xdg-open") or shutil.which("open") else "warn",
                "Browser-Oeffnung ist optional.",
                "Nutze --no-browser und oeffne die URL manuell.",
            )
        )

    if config_path:
        path = Path(config_path)
        if not path.exists():
            checks.append(
                Check(
                    "Config",
                    "error",
                    f"Config nicht gefunden: {path}",
                    "Pruefe den Pfad oder exportiere die Config erneut aus der GUI.",
                )
            )
        else:
            try:
                from .experiment import ExperimentConfig

                cfg = ExperimentConfig.from_file(path)
                checks.append(Check("Config", "ok", f"{path.name}: {cfg.name}"))
                if cfg.dataset_id:
                    from .datasets import STORE

                    try:
                        STORE.get_meta(cfg.dataset_id)
                        checks.append(Check("Dataset", "ok", cfg.dataset_id))
                    except Exception as exc:
                        checks.append(
                            Check(
                                "Dataset",
                                "error",
                                f"Dataset {cfg.dataset_id!r} fehlt: {exc}",
                                "Importiere das Dataset-Bundle oder nutze eine Config mit data-Rezept.",
                            )
                        )
                else:
                    checks.append(Check("Dataset", "ok", "Wird aus der Config-Rezeptur gebaut."))
                out = Path(cfg.output_dir)
                out.mkdir(parents=True, exist_ok=True)
                checks.append(Check("Output", "ok", str(out.resolve())))
            except Exception as exc:
                checks.append(
                    Check(
                        "Config",
                        "error",
                        f"Config ungueltig: {exc}",
                        "Pruefe YAML-Syntax und Felder; exportiere sie erneut.",
                    )
                )
    elif mode == "cli":
        checks.append(
            Check(
                "Config",
                "error",
                "Headless-Modus braucht --config.",
                "Beispiel: python3 app.py --config experiment.yaml --headless",
            )
        )

    return checks


def render_checks(checks: list[Check], *, title: str = "Preflight") -> Console:
    console = Console(stderr=True, highlight=False)
    table = Table(title=title, border_style="cyan", show_lines=False)
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Details")
    for check in checks:
        table.add_row(check.name, _status_icon(check.status), check.message)
        if check.status != "ok" and check.fix:
            table.add_row("", "", f"[dim]→ {check.fix}[/dim]")
    console.print(table)
    return console


def ensure_ready(checks: list[Check]) -> None:
    errors = [check for check in checks if check.status == "error"]
    if errors:
        fixes = "\n".join(
            f"- {check.name}: {check.message}\n  Fix: {check.fix}" for check in errors
        )
        raise PreflightError("Startup-Pruefung fehlgeschlagen. Bitte behebe:\n" + fixes)
