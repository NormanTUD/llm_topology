#!/usr/bin/env python3
"""Low-resource GUI smoke test for the Docker app container."""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
import websockets
from rich.console import Console
from rich.table import Table

console = Console()
FAILURES: list[str] = []
EXCEPTIONS: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        FAILURES.append(name)
        console.print(f"[red]FAIL[/red] {name} {detail}")
    else:
        console.print(f"[green]PASS[/green] {name} {detail}")


def seed(base_url: str) -> str:
    spec = {
        "structure": "circle",
        "n_points": 40,
        "ambient_dim": 3,
        "precision": 5,
        "copies_min": 1,
        "copies_max": 1,
    }
    batch = httpx.post(
        f"{base_url}/api/generate/batch",
        json={"spec": spec, "count": 2, "mode": "single"},
        timeout=60,
    )
    batch.raise_for_status()
    httpx.post(
        f"{base_url}/api/ui-state",
        json={
            "lastBatch": batch.json()["items"],
            "selected": 0,
            "appView": "data",
            "uiMode": "simple",
            "targetBlocks": ["betti"],
            "batchMode": "single",
            "batchStructures": [],
        },
        timeout=30,
    ).raise_for_status()
    data = {
        "structures": ["circle", "sphere"],
        "n_train": 4,
        "n_val": 4,
        "n_test": 2,
        "n_points": 8,
        "max_seq_len": 512,
        "precision": 5,
        "augment": False,
        "val_transform_mode": "clean",
        "test_transform_mode": "clean",
        "task_mix": 0.5,
        "seed": 0,
    }
    meta = httpx.post(
        f"{base_url}/api/datasets",
        json={"name": "docker-smoke", "data_spec": data},
        timeout=60,
    )
    meta.raise_for_status()
    return meta.json()["id"]


def launch_chromium(base_url: str, profile: Path) -> subprocess.Popen:
    parsed = urlparse(base_url)
    host = socket.gethostbyname(parsed.hostname or "app")
    chrome_url = f"http://{host}:{parsed.port or 80}"
    command = [
        "chromium",
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        "--disable-webgl",
        "--disable-dev-shm-usage",
        "--disable-extensions",
        "--no-first-run",
        "--js-flags=--max-old-space-size=256",
        "--remote-debugging-port=9222",
        f"--user-data-dir={profile}",
        "--window-size=1440,1100",
        chrome_url,
    ]
    return subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


async def run_gui(base_url: str, dataset_id: str) -> None:
    deadline = time.time() + 20
    tab = None
    while time.time() < deadline:
        try:
            tabs = httpx.get("http://127.0.0.1:9222/json", timeout=2).json()
            tab = next((item for item in tabs if item.get("type") == "page"), None)
            if tab:
                break
        except Exception:
            pass
        await asyncio.sleep(0.5)
    if not tab:
        raise RuntimeError("Chromium DevTools endpoint did not become ready")

    ws = await websockets.connect(tab["webSocketDebuggerUrl"], max_size=30_000_000)
    counter = 0

    async def send(method: str, params: dict | None = None):
        nonlocal counter
        counter += 1
        await ws.send(json.dumps({"id": counter, "method": method, "params": params or {}}))
        while True:
            message = json.loads(await ws.recv())
            if message.get("id") == counter:
                return message
            if message.get("method") == "Runtime.exceptionThrown":
                details = message.get("params", {}).get("exceptionDetails", {})
                EXCEPTIONS.append(details.get("exception", {}).get("description", details.get("text", "")))

    async def evaluate(expression: str):
        result = await send(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
        )
        return result.get("result", {}).get("result", {}).get("value")

    async def wait_for(expression: str, timeout: float = 30.0):
        end = time.time() + timeout
        while time.time() < end:
            value = await evaluate(expression)
            if value:
                return value
            await asyncio.sleep(0.4)
        return None

    async def click(selector: str, wait: float = 0.25) -> str:
        value = await evaluate(
            f"""(() => {{
              const e = document.querySelector({json.dumps(selector)});
              if (!e) return "missing";
              if (e.disabled) return "disabled";
              e.click();
              return "clicked";
            }})()"""
        )
        await asyncio.sleep(wait)
        return value

    async def set_value(selector: str, value) -> str:
        return await evaluate(
            f"""(() => {{
              const e = document.querySelector({json.dumps(selector)});
              if (!e) return "missing";
              if (e.type === "checkbox") e.checked = Boolean({json.dumps(value)});
              else e.value = {json.dumps(str(value))};
              e.dispatchEvent(new Event("input", {{ bubbles: true }}));
              e.dispatchEvent(new Event("change", {{ bubbles: true }}));
              return e.value;
            }})()"""
        )

    await send("Runtime.enable")
    await send("Log.enable")
    await send("Page.enable")
    await asyncio.sleep(2)
    await wait_for('document.querySelector("#data-view") ? "yes" : ""', 15)
    await wait_for('document.body.classList.contains("simple-mode") ? "yes" : ""', 15)
    await wait_for('document.querySelectorAll(".cloud-card").length > 0 ? "yes" : ""', 20)

    check("Data Studio visible", bool(await evaluate('!document.querySelector("#data-view").classList.contains("hidden")')))
    check("seeded gallery rendered", int(await evaluate('document.querySelectorAll(".cloud-card").length') or 0) > 0)
    check("simple mode default", bool(await evaluate('document.body.classList.contains("simple-mode")')))
    check("default target betti", (await evaluate('Array.from(document.querySelectorAll(".target-check:checked")).map(e=>e.dataset.block)')) == ["betti"])
    check("components range visible", bool(await evaluate('getComputedStyle(document.querySelector("#copies-min").closest("label")).display !== "none"')))

    await click("#mode-toggle", 0.3)
    check("expert mode enabled", not bool(await evaluate('document.body.classList.contains("simple-mode")')))
    await click("#mode-toggle", 0.3)

    await set_value("#n-points", 40)
    await set_value("#batch-count", 2)
    await set_value("#batch-mode", "single")
    check("top generate button", await click("#top-generate", 0.5) == "clicked")
    await wait_for('document.querySelector("#footer-state").textContent === "ready"', 30)

    check("quick demo button", await click("#quick-demo", 0.5) == "clicked")
    await wait_for('document.querySelector("#footer-state").textContent === "ready"', 40)
    check("quick demo produced torus", "Torus" in str(await evaluate('document.querySelector("#metric-structure").textContent')))

    await set_value("#n-points", 40)
    await set_value("#copies-min", 1)
    await set_value("#copies-max", 1)
    await set_value("#batch-mode", "single")
    await click("#generate-btn", 0.5)
    await wait_for('document.querySelector("#footer-state").textContent === "ready"', 30)
    check("gallery cards present", int(await evaluate('document.querySelectorAll(".cloud-card").length') or 0) > 0)
    check("gallery card clickable", await click(".cloud-card", 0.3) == "clicked")

    await set_value("#dataset-name", "docker-built")
    await set_value("#dataset-n-train", 4)
    await set_value("#dataset-n-val", 4)
    await set_value("#dataset-n-test", 2)
    await set_value("#dataset-n-points", 8)
    await set_value("#dataset-context", 512)
    await set_value("#dataset-precision", 5)
    check("build dataset button", await click("#build-dataset", 0.5) == "clicked")
    saved = await wait_for('document.querySelector("#dataset-status").textContent.startsWith("saved") ? document.querySelector("#dataset-status").textContent : ""', 45)
    check("dataset saved", bool(saved), str(saved))

    await click("#nav-lab", 0.8)
    check("dataset selector populated", bool(await evaluate('document.querySelector("#dataset-select").value')))
    check("preview dataset button", await click("#preview-dataset", 0.5) == "clicked")
    check("preview cards rendered", int(await evaluate('document.querySelectorAll("#dataset-preview .prediction-card").length') or 0) > 0)
    for split in ("val", "test"):
        await set_value("#dataset-preview-split", split)
        await click("#preview-dataset", 0.4)
        check(f"preview {split}", int(await evaluate('document.querySelectorAll("#dataset-preview .prediction-card").length') or 0) > 0)

    for selector, value in (
        ("#n-layers", 1),
        ("#n-heads", 2),
        ("#d-model", 16),
        ("#d-ff", 32),
        ("#context-train", 512),
        ("#epochs", 1),
        ("#batch-size", 2),
        ("#max-new-tokens", 16),
        ("#patience", 2),
        ("#lr", 0.001),
    ):
        await set_value(selector, value)

    await click("#train-toggle", 0.5)
    check("train toggle starts", await wait_for('["queued","preparing","running","completed"].includes(document.querySelector("#run-status").textContent) ? "yes" : ""', 20) == "yes")
    final = await wait_for('["completed","early_stopped","stopped","failed"].includes(document.querySelector("#run-status").textContent) ? document.querySelector("#run-status").textContent : ""', 90)
    check("training completed", final == "completed", str(final))

    await click("#interpret-nav", 0.5)
    check("top analysis button", await click("#top-run-analysis", 0.5) == "clicked")
    check("analysis rendered", await wait_for('document.querySelector("#probe-plot").innerHTML.length > 1000 ? "yes" : ""', 60) == "yes")
    await set_value("#fresh-count", 1)
    await set_value("#fresh-points", 8)
    await set_value("#fresh-ambient", 3)
    check("top prediction button", await click("#top-run-predictions", 0.5) == "clicked")
    check("fresh predictions rendered", int(await wait_for('document.querySelectorAll("#fresh-predictions .prediction-card").length', 45) or 0) > 0)

    for selector in (
        "#export-config",
        "#export-run",
        "#export-dataset",
        "#refresh-runs",
        "#refresh-datasets",
        "#top-build-dataset",
        "#randomize-btn",
    ):
        clicked = await click(selector, 0.3)
        check(f"{selector} clickable", clicked == "clicked", clicked)
    await click(".lab-tab[data-tab=\"interpret\"]", 0.3)
    await click(".lab-tab[data-tab=\"train\"]", 0.3)
    await click("#go-data-studio", 0.5)
    check("go to data studio", bool(await evaluate('!document.querySelector("#data-view").classList.contains("hidden")')))
    await click("#nav-lab", 0.4)

    check("no JS exceptions", not EXCEPTIONS, "; ".join(EXCEPTIONS[:2]))
    await ws.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://app:8000")
    parser.add_argument("--keep-profile", action="store_true")
    args = parser.parse_args()

    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            if httpx.get(f"{args.base_url}/api/health", timeout=2).status_code == 200:
                break
        except Exception:
            time.sleep(1)
    else:
        raise SystemExit(f"App not reachable at {args.base_url}")

    dataset_id = seed(args.base_url)
    profile = Path(tempfile.mkdtemp(prefix="topology-chromium-"))
    chrome = launch_chromium(args.base_url, profile)
    try:
        asyncio.run(run_gui(args.base_url, dataset_id))
    finally:
        chrome.terminate()
        try:
            chrome.wait(timeout=5)
        except Exception:
            chrome.kill()
        if not args.keep_profile:
            shutil.rmtree(profile, ignore_errors=True)

    table = Table(title="Docker GUI smoke", border_style="cyan")
    table.add_column("Result")
    table.add_column("Count")
    table.add_row("Failures", str(len(FAILURES)))
    table.add_row("JS exceptions", str(len(EXCEPTIONS)))
    console.print(table)
    if FAILURES or EXCEPTIONS:
        sys.exit(1)
    console.print("[bold green]All Docker GUI checks passed.[/bold green]")


if __name__ == "__main__":
    main()
