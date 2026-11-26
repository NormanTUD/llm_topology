from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from ..analysis.probes import analyze_checkpoint
from ..model.config import AnalysisSpec, TrainingSpec
from ..tokenizer import NumericTokenizer


class JobError(RuntimeError):
    pass


class TrainingJob:
    def __init__(self, spec: TrainingSpec, job_id: str | None = None) -> None:
        self.id = job_id or uuid.uuid4().hex[:12]
        self.spec = spec
        self.created_at = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.status = "queued"
        self.error: str | None = None
        self.current: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []
        self.loss_history: list[dict[str, Any]] = []
        self.samples: dict[str, str] = {}
        self.best_checkpoint: str | None = None
        self.best_val_loss: float | None = None
        self.parameter_count: int | None = None
        self.device: str | None = None
        self.dataset_info: dict[str, Any] = {}
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    @property
    def run_dir(self) -> Path:
        return Path.cwd() / self.spec.training.checkpoint_root / self.id

    def update(self, **values: Any) -> None:
        with self._lock:
            for key, value in values.items():
                setattr(self, key, value)

    def update_current(self, values: dict[str, Any]) -> None:
        with self._lock:
            self.current = values

    def add_batch(self, values: dict[str, Any]) -> None:
        with self._lock:
            self.current = values
            self.loss_history.append(values)
            if len(self.loss_history) > 20000:
                self.loss_history = self.loss_history[-20000:]

    def add_history(self, values: dict[str, Any]) -> None:
        with self._lock:
            self.history.append(values)
            self.current = values
            if "samples" in values:
                self.samples = values["samples"]
        self._write_result()

    def snapshot(self, *, include_history: bool = True) -> dict[str, Any]:
        with self._lock:
            data = {
                "id": self.id,
                "status": self.status,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "error": self.error,
                "current": self.current,
                "samples": self.samples,
                "best_checkpoint": self.best_checkpoint,
                "best_val_loss": self.best_val_loss,
                "parameter_count": self.parameter_count,
                "device": self.device,
                "dataset_info": self.dataset_info,
                "loss_history": list(self.loss_history),
                "spec": self.spec.model_dump(mode="json"),
                "run_dir": str(self.run_dir),
            }
            if include_history:
                data["history"] = list(self.history)
            return data

    def start(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._write_result()
        print(f"[{self.id}] job queued run_dir={self.run_dir}", flush=True)
        self._thread = threading.Thread(target=self._run, name=f"train-{self.id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def _run(self) -> None:
        self.update(status="preparing", started_at=time.time())
        print(f"[{self.id}] preparing dataset and model", flush=True)
        try:
            from ..model.train import Trainer

            tokenizer = NumericTokenizer()
            trainer = Trainer(self.spec, tokenizer=tokenizer, checkpoint_dir=self.run_dir)
            self.update(
                status="running",
                parameter_count=trainer.model.parameter_count(),
                device=str(trainer.device),
                dataset_info={
                    "train": len(trainer.bundle.train),
                    "val": len(trainer.bundle.val),
                    "skipped": trainer.bundle.skipped,
                    "max_dim": trainer.bundle.max_dim,
                    "structures": [structure.value for structure in self.spec.data.structures],
                },
            )

            print(
                f"[{self.id}] model params={trainer.model.parameter_count()} "
                f"device={trainer.device} train={len(trainer.bundle.train)} "
                f"val={len(trainer.bundle.val)}",
                flush=True,
            )

            def callback(values: dict[str, Any]) -> None:
                if values.get("type") == "batch":
                    payload = {key: value for key, value in values.items() if key != "type"}
                    self.add_batch(payload)
                elif values.get("type") == "epoch":
                    payload = {key: value for key, value in values.items() if key != "type"}
                    self.add_history(payload)
                    if "val_loss" in payload:
                        self.update(best_val_loss=float(payload["val_loss"]))
                    print(
                        f"[{self.id}] epoch {payload.get('epoch')} done "
                        f"train={payload.get('train_loss'):.4f} val={payload.get('val_loss'):.4f}",
                        flush=True,
                    )

            result = trainer.run(callback=callback, stop_event=self.stop_event)
            self.update(
                status=result.status,
                finished_at=time.time(),
                best_checkpoint=result.best_checkpoint,
                best_val_loss=result.best_val_loss,
                samples=result.samples,
                history=result.history,
                current=result.history[-1] if result.history else {},
            )
            self._write_result()
            print(
                f"[{self.id}] finished status={result.status} "
                f"best_val={result.best_val_loss:.4f} checkpoint={result.best_checkpoint}",
                flush=True,
            )
        except Exception as exc:  # pragma: no cover - surfaced through the API
            self.update(
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                finished_at=time.time(),
            )
            self._write_result()
            print(f"[{self.id}] FAILED {type(exc).__name__}: {exc}", flush=True)

    def _write_result(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / "job.json"
        path.write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, TrainingJob] = {}
        self._lock = threading.Lock()
        self._load_disk_runs()

    @staticmethod
    def _next_experiment_id(root: Path) -> str:
        highest = 0
        if root.exists():
            for path in root.glob("exp-*"):
                suffix = path.name.removeprefix("exp-")
                if suffix.isdigit():
                    highest = max(highest, int(suffix))
        return f"exp-{highest + 1:04d}"

    def _load_disk_runs(self) -> None:
        root = Path.cwd() / "runs"
        if not root.exists():
            return
        for path in sorted(root.glob("exp-*/job.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                spec = TrainingSpec.model_validate(data["spec"])
                job = TrainingJob(spec, job_id=data["id"])
                for key in (
                    "created_at",
                    "started_at",
                    "finished_at",
                    "error",
                    "current",
                    "samples",
                    "best_checkpoint",
                    "best_val_loss",
                    "parameter_count",
                    "device",
                    "dataset_info",
                    "history",
                    "loss_history",
                ):
                    if key in data:
                        setattr(job, key, data[key])
                job.status = data.get("status", "interrupted")
                if job.status in {"queued", "preparing", "running"}:
                    job.status = "interrupted"
                self._jobs[job.id] = job
            except Exception as exc:
                print(f"[jobs] could not load {path}: {exc}", flush=True)

    def start(self, spec: TrainingSpec) -> TrainingJob:
        root = Path.cwd() / spec.training.checkpoint_root
        with self._lock:
            job_id = self._next_experiment_id(root)
            while job_id in self._jobs:
                number = int(job_id.removeprefix("exp-")) + 1
                job_id = f"exp-{number:04d}"
            job = TrainingJob(spec, job_id=job_id)
            self._jobs[job.id] = job
        job.start()
        return job

    def get(self, job_id: str) -> TrainingJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise JobError(f"unknown job: {job_id}")
        return job

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        return [job.snapshot(include_history=False) for job in jobs]

    def analyze(self, job_id: str, spec: AnalysisSpec) -> dict[str, Any]:
        job = self.get(job_id)
        print(f"[{job_id}] analysis start", flush=True)
        checkpoint = job.best_checkpoint
        if not checkpoint or not Path(checkpoint).exists():
            raise JobError("this job has no best checkpoint yet")
        result = analyze_checkpoint(checkpoint, spec)
        output = job.run_dir / "analysis.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["analysis_path"] = str(output)
        print(f"[{job_id}] analysis saved to {output}", flush=True)
        return result


MANAGER = JobManager()
