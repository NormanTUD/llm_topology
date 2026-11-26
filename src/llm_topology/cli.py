from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from .datasets import STORE as DATASET_STORE
from .experiment import ExperimentConfig
from .model.config import TrainingSpec
from .model.train import Trainer, resolve_device
from .tokenizer import NumericTokenizer


def _next_run_dir(root: Path) -> Path:
    highest = 0
    if root.exists():
        for path in root.glob("exp-*"):
            suffix = path.name.removeprefix("exp-")
            if suffix.isdigit():
                highest = max(highest, int(suffix))
    return root / f"exp-{highest + 1:04d}"


def _write_job_snapshot(
    path: Path,
    *,
    run_id: str,
    spec: TrainingSpec,
    trainer: Trainer,
    status: str,
    started: float,
    finished: float,
    history: list[dict[str, Any]],
    loss_history: list[dict[str, Any]],
    result,
) -> None:
    dataset_info = {
        "train": len(trainer.bundle.train),
        "val": len(trainer.bundle.val),
        "test": len(trainer.bundle.test),
        "skipped": trainer.bundle.skipped,
        "max_dim": trainer.bundle.max_dim,
        "structures": [s.value for s in trainer.bundle.data_spec.structures],
    }
    snapshot = {
        "id": run_id,
        "status": status,
        "created_at": started,
        "started_at": started,
        "finished_at": finished,
        "error": None,
        "current": history[-1] if history else {},
        "samples": result.samples if result else {},
        "best_checkpoint": result.best_checkpoint if result else None,
        "best_val_loss": result.best_val_loss if result else None,
        "parameter_count": trainer.model.parameter_count(),
        "device": str(trainer.device),
        "dataset_info": dataset_info,
        "loss_history": loss_history,
        "history": history,
        "spec": spec.model_dump(mode="json"),
        "run_dir": str(path.parent),
    }
    path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")


def run_experiment(
    config_path: str | Path,
    *,
    output_override: str | None = None,
    device_override: str | None = None,
) -> Path:
    console = Console(stderr=True, highlight=False)
    cfg = ExperimentConfig.from_file(config_path)
    if device_override:
        cfg.training.device = device_override
    spec = cfg.to_training_spec()
    tokenizer = NumericTokenizer()

    dataset_meta = None
    if spec.dataset_id:
        try:
            dataset_meta = DATASET_STORE.get_meta(spec.dataset_id)
        except Exception as exc:
            raise SystemExit(f"Unknown dataset {spec.dataset_id!r}: {exc}") from exc
    elif cfg.data is not None:
        console.print(Panel("Building dataset from config recipe …", border_style="cyan"))
        dataset_meta = DATASET_STORE.save(cfg.name, cfg.data)
        spec.dataset_id = dataset_meta["id"]
    else:
        raise SystemExit("Config needs dataset_id or data recipe.")

    root = Path(output_override or cfg.output_dir)
    run_dir = _next_run_dir(root)
    run_id = run_dir.name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(cfg.to_yaml(), encoding="utf-8")
    (run_dir / "dataset.json").write_text(json.dumps(dataset_meta, indent=2), encoding="utf-8")

    device = resolve_device(cfg.training.device)
    console.print(
        Panel.fit(
            f"[bold cyan]{cfg.name}[/bold cyan]\n"
            f"run [bold]{run_id}[/bold]  device [bold]{device}[/bold]\n"
            f"dataset {spec.dataset_id}  context {cfg.model.max_seq_len}\n"
            f"output {run_dir}",
            title="LLM Topology Lab · headless",
            border_style="cyan",
        )
    )

    started = time.time()
    loss_history: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=32),
        MofNCompleteColumn(),
        TextColumn("· {task.fields[status]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    task_id = progress.add_task("training", total=None, status="preparing")

    def callback(values: dict[str, Any]) -> None:
        if values.get("type") == "batch":
            payload = {k: v for k, v in values.items() if k != "type"}
            loss_history.append(payload)
            total = payload.get("total_steps") or 1
            progress.update(
                task_id,
                total=total,
                completed=payload.get("step", 0),
                status=f"loss {payload.get('loss', 0):.4f}",
            )
            with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
        elif values.get("type") == "epoch":
            payload = {k: v for k, v in values.items() if k != "type"}
            history.append(payload)
            progress.update(
                task_id, status=f"epoch {payload.get('epoch')} val {payload.get('val_loss', 0):.4f}"
            )
            table = Table(title=f"Epoch {payload.get('epoch')}", border_style="cyan")
            table.add_column("Metric")
            table.add_column("Value", justify="right")
            for key in (
                "train_loss",
                "val_loss",
                "val_ce",
                "test_loss",
                "val_multiple_choice",
                "lr",
            ):
                if key in payload:
                    table.add_row(key, str(payload[key]))
            console.print(table)

    result = None
    trainer: Trainer | None = None
    with (
        (run_dir / "run.log").open("w", encoding="utf-8") as log_handle,
        contextlib.redirect_stdout(log_handle),
    ):
        try:
            trainer = Trainer(spec, tokenizer=tokenizer, checkpoint_dir=run_dir)
            with progress:
                result = trainer.run(callback=callback)
            status = result.status
        except Exception as exc:
            status = "failed"
            console.print(
                Panel(f"[red]{type(exc).__name__}: {exc}[/red]", title="failed", border_style="red")
            )
            if trainer is not None:
                _write_job_snapshot(
                    run_dir / "job.json",
                    run_id=run_id,
                    spec=spec,
                    trainer=trainer,
                    status=status,
                    started=started,
                    finished=time.time(),
                    history=history,
                    loss_history=loss_history,
                    result=None,
                )
            raise SystemExit(1) from exc

    _write_job_snapshot(
        run_dir / "job.json",
        run_id=run_id,
        spec=spec,
        trainer=trainer,
        status=status,
        started=started,
        finished=time.time(),
        history=history,
        loss_history=loss_history,
        result=result,
    )
    console.print(
        Panel.fit(
            f"status [bold]{status}[/bold]\n"
            f"best val [bold cyan]{result.best_val_loss:.4f}[/bold cyan]\n"
            f"checkpoint [bold]{result.best_checkpoint}[/bold]\n"
            f"run dir [bold]{run_dir}[/bold]",
            title="training finished",
            border_style="green" if status == "completed" else "yellow",
        )
    )
    return run_dir
