from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llm_topology.config import DatasetSpec, StructureKind
from llm_topology.datasets import STORE as DATASET_STORE
from llm_topology.experiment import ExperimentConfig
from llm_topology.limits import estimate_training
from llm_topology.model.config import DataBuildSpec, ModelSpec, TrainConfig, TrainingSpec
from llm_topology.preflight import (
    Check,
    PreflightError,
    collect_checks,
    ensure_ready,
    system_snapshot,
)
from llm_topology.web.main import app

client = TestClient(app)


def test_system_snapshot_has_runtime_fields() -> None:
    snapshot = system_snapshot()
    assert snapshot["python"]
    assert snapshot["cpus"]
    assert snapshot["ram_total_bytes"] >= 0
    assert isinstance(snapshot["gpu"], bool)


def test_preflight_checks_are_actionable() -> None:
    checks = collect_checks(mode="doctor")
    assert any(check.name == "Python" for check in checks)
    assert any(check.name == "uv" for check in checks)
    assert all(check.fix for check in checks if check.status != "ok")


def test_preflight_blocks_on_error() -> None:
    with pytest.raises(PreflightError):
        ensure_ready([Check("broken", "error", "kaputt", "reparieren")])


def test_default_target_is_betti_only() -> None:
    spec = DatasetSpec(structure=StructureKind.TORUS, n_points=40)
    assert [block.name for block in spec.target_schema.blocks] == ["betti"]


def test_gui_contains_role_layout_and_component_range() -> None:
    html = client.get("/").text
    for marker in (
        'id="mode-toggle"',
        'id="quick-demo"',
        'id="copies-min"',
        'id="copies-max"',
        'id="train-view"',
        'id="dataset-preview"',
        'id="top-run-analysis"',
        'id="export-config"',
        'id="import-run"',
    ):
        assert marker in html


def test_config_import_rejects_invalid_yaml() -> None:
    response = client.post("/api/config/import", json={"yaml": "name: [broken"})
    assert response.status_code == 422
    assert "Config ungueltig" in response.json()["detail"]


def test_dataset_import_rejects_zip_slip() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../evil/meta.json", "{}")
    response = client.post(
        "/api/datasets/import",
        files={"file": ("evil.zip", buffer.getvalue(), "application/zip")},
    )
    assert response.status_code == 422
    assert "unsichere Pfade" in response.json()["detail"]


def test_cuda_request_is_blocked_without_gpu() -> None:
    import torch

    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this machine")
    spec = TrainingSpec(
        data=DataBuildSpec(max_seq_len=512),
        model=ModelSpec(max_seq_len=512, n_layers=1, n_heads=2, d_model=16, d_ff=32),
        training=TrainConfig(epochs=1, batch_size=2, device="cuda"),
    )
    result = estimate_training(spec)
    assert result["level"] == "block"
    assert any("CUDA" in warning for warning in result["warnings"])


def test_cli_config_and_run_layout(tmp_path: Path, monkeypatch) -> None:
    from llm_topology import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(DATASET_STORE, "root", tmp_path / "datasets")
    spec = TrainingSpec(
        data=DataBuildSpec(
            structures=["circle"],
            n_train=4,
            n_val=2,
            n_test=2,
            n_points=8,
            max_seq_len=512,
            precision=5,
            augment=False,
            val_transform_mode="clean",
            test_transform_mode="clean",
            seed=0,
        ),
        model=ModelSpec(max_seq_len=512, n_layers=1, n_heads=2, d_model=16, d_ff=32),
        training=TrainConfig(epochs=1, batch_size=2, warmup_steps=1, patience=2, device="cpu"),
    )
    config = ExperimentConfig(
        name="cli-test",
        data=spec.data,
        model=spec.model,
        training=spec.training,
        run_analysis=False,
        output_dir=str(tmp_path / "runs"),
    )
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(config.to_yaml(), encoding="utf-8")
    run_dir = cli.run_experiment(config_path)
    assert (run_dir / "job.json").exists()
    assert (run_dir / "history.json").exists()
    assert (run_dir / "run.log").exists()
    assert (run_dir / "metrics.jsonl").exists()
