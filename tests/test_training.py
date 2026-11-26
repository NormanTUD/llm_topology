from __future__ import annotations

from pathlib import Path

import pytest

from llm_topology.analysis.probes import analyze_checkpoint
from llm_topology.model.config import (
    AnalysisSpec,
    DataBuildSpec,
    ModelSpec,
    TrainConfig,
    TrainingSpec,
)
from llm_topology.model.train import Trainer
from llm_topology.tokenizer import NumericTokenizer


def _spec() -> TrainingSpec:
    return TrainingSpec(
        data=DataBuildSpec(
            structures=["circle", "sphere"],
            n_train=4,
            n_val=4,
            n_points=8,
            max_seq_len=512,
            precision=2,
            task_mix=0.5,
            augment=False,
        ),
        model=ModelSpec(
            max_seq_len=512,
            n_layers=1,
            n_heads=2,
            d_model=16,
            d_ff=32,
            dropout=0.0,
        ),
        training=TrainConfig(
            epochs=1,
            batch_size=2,
            lr=1e-3,
            warmup_steps=1,
            patience=2,
            device="cpu",
        ),
    )


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory) -> tuple[Path, object]:
    directory = tmp_path_factory.mktemp("topology-run")
    trainer = Trainer(
        _spec(),
        tokenizer=NumericTokenizer(),
        checkpoint_dir=directory / "run",
    )
    return directory, trainer.run()


def test_training_saves_checkpoint_and_history(trained_run) -> None:
    directory, result = trained_run
    assert result.history
    assert result.status in {"completed", "early_stopped"}
    assert Path(result.best_checkpoint).exists()
    assert (directory / "run" / "history.json").exists()
    assert "val_loss" in result.history[0]
    assert "samples" in result.history[0]


def test_analysis_from_checkpoint(trained_run) -> None:
    _directory, result = trained_run
    analysis = analyze_checkpoint(result.best_checkpoint, AnalysisSpec(max_examples=8, pca_dim=2))
    assert analysis["summary"]["layers"] >= 1
    assert analysis["probes"]
    assert analysis["projection"]["coordinates"]
    assert analysis["neuron_maps"]["normalized"]
    assert analysis["attention"]["tokens"]
