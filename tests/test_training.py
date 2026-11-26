from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from llm_topology.analysis.probes import analyze_checkpoint
from llm_topology.config import FiltrationSpec, TargetSchema, all_target_blocks
from llm_topology.datasets import DatasetStore
from llm_topology.model import train as train_module
from llm_topology.model.config import (
    AnalysisSpec,
    DataBuildSpec,
    ModelSpec,
    TrainConfig,
    TrainingSpec,
)
from llm_topology.model.data import build_dataset
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


def test_augmented_dataset_build_does_not_crash() -> None:
    data = _spec().data.model_copy(update={"augment": True, "augment_strength": 0.5})
    bundle = build_dataset(NumericTokenizer(), data)
    assert len(bundle.train) == data.n_train
    assert len(bundle.val) == data.n_val


def test_augmented_example_keeps_original_for_display() -> None:
    data = _spec().data.model_copy(update={"augment": True, "augment_strength": 0.5})
    bundle = build_dataset(NumericTokenizer(), data)
    example = bundle.train[0]
    assert example.original_points
    assert example.points != example.original_points


def test_val_test_splits_and_modes() -> None:
    data = _spec().data.model_copy(
        update={
            "n_test": 3,
            "val_transform_mode": "clean",
            "test_transform_mode": "heldout",
        }
    )
    bundle = build_dataset(NumericTokenizer(), data)
    assert len(bundle.test) == 3
    assert bundle.val[0].transform.get("rotation_deg") in (None, [0, 0, 0], (0, 0, 0))
    assert bundle.test[0].transform.get("twist") is not None


def test_manual_epsilon_is_encoded_in_inference_prompt() -> None:
    data = _spec().data.model_copy(
        update={
            "filtration": FiltrationSpec(method="alpha", epsilon=0.5),
            "target_schema": TargetSchema(blocks=all_target_blocks(max_dim=2)),
        }
    )
    bundle = build_dataset(NumericTokenizer(), data)
    example = next(item for item in bundle.train if item.task == "infer")
    epsilon_id = bundle.tokenizer.token_to_id["<epsilon>"]
    assert epsilon_id in example.input_ids
    assert example.target_groups[-1][2] == 0.5

    clean = _spec().data.model_copy(update={"augment": False, "val_transform_mode": "clean"})
    rotated = clean.model_copy(update={"val_transform_mode": "same", "val_augment_strength": 0.45})
    clean_bundle = build_dataset(NumericTokenizer(), clean)
    rotated_bundle = build_dataset(NumericTokenizer(), rotated)
    assert len(clean_bundle.val) == len(rotated_bundle.val)
    assert not np.allclose(clean_bundle.val[0].points, rotated_bundle.val[0].points)
    assert clean_bundle.val[0].betti == rotated_bundle.val[0].betti


def test_dataset_manual_filtration_is_preserved() -> None:
    data = _spec().data.model_copy(
        update={
            "filtration": FiltrationSpec(method="alpha", epsilon=0.5),
            "target_schema": TargetSchema(blocks=all_target_blocks(max_dim=2)),
        }
    )
    bundle = build_dataset(NumericTokenizer(), data)
    for example in [*bundle.train, *bundle.val]:
        assert example.target_groups[-1][2] == 0.5


def test_saved_dataset_roundtrip(tmp_path: Path) -> None:
    store = DatasetStore(root=tmp_path / "datasets")
    meta = store.save("tiny", _spec().data)
    assert meta["train_count"] == 4
    loaded = store.load(meta["id"])
    assert len(loaded.train) == 4
    assert loaded.data_spec.structures == _spec().data.structures


def test_training_target_order_matches_schema() -> None:
    spec = _spec()
    spec.data = spec.data.model_copy(
        update={"target_schema": TargetSchema(blocks=all_target_blocks(max_dim=2))}
    )
    bundle = build_dataset(NumericTokenizer(), spec.data)
    example = next(item for item in bundle.train if item.task == "infer")
    assert [block.name for block in spec.data.target_schema.blocks] == [
        "betti",
        "ph",
        "genus",
        "n_components",
        "filtration",
    ]
    assert example.target_groups[0] == [float(value) for value in example.betti]
    assert example.target_groups[1] == [float(value) for value in example.ph]


def test_trainer_can_train_from_saved_dataset(tmp_path: Path, monkeypatch) -> None:
    store = DatasetStore(root=tmp_path / "datasets")
    meta = store.save("tiny-train", _spec().data)
    monkeypatch.setattr(train_module, "STORE", store)
    spec = _spec().model_copy(update={"dataset_id": meta["id"]})
    trainer = Trainer(
        spec,
        tokenizer=NumericTokenizer(),
        checkpoint_dir=tmp_path / "run-from-dataset",
    )
    result = trainer.run()
    assert result.history
    assert Path(result.best_checkpoint).exists()


def test_training_saves_checkpoint_and_history(trained_run) -> None:
    directory, result = trained_run
    assert result.history
    assert result.status in {"completed", "early_stopped"}
    assert Path(result.best_checkpoint).exists()
    assert (directory / "run" / "history.json").exists()
    assert "val_loss" in result.history[0]
    assert "samples" in result.history[0]
    assert "test_loss" in result.history[0]


def test_analysis_from_checkpoint(trained_run) -> None:
    _directory, result = trained_run
    analysis = analyze_checkpoint(result.best_checkpoint, AnalysisSpec(max_examples=8, pca_dim=2))
    assert analysis["summary"]["layers"] >= 1
    assert analysis["probes"]
    assert analysis["projection"]["coordinates"]
    assert analysis["neuron_maps"]["normalized"]
    assert analysis["attention"]["tokens"]
