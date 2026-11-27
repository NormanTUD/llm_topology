from __future__ import annotations

from llm_topology.config import DatasetSpec, FiltrationMethod, FiltrationSpec
from llm_topology.limits import estimate_dataset, estimate_generation, estimate_training
from llm_topology.model.config import DataBuildSpec, ModelSpec, TrainConfig, TrainingSpec


def test_generation_estimate_warns_for_rips() -> None:
    result = estimate_generation(
        DatasetSpec(
            structure="torus",
            n_points=2000,
            filtration=FiltrationSpec(method=FiltrationMethod.RIPS),
        ),
        count=2,
    )
    assert result["level"] in {"warn", "block"}
    assert result["warnings"]


def test_dataset_estimate_blocks_context_overflow() -> None:
    result = estimate_dataset(
        DataBuildSpec(
            structures=["torus"],
            n_train=4,
            n_val=2,
            n_test=2,
            n_points=4000,
            max_seq_len=512,
            precision=5,
        )
    )
    assert result["level"] == "block"
    assert any("Kontext" in warning for warning in result["warnings"])


def test_training_estimate_blocks_huge_model() -> None:
    spec = TrainingSpec(
        model=ModelSpec(
            max_seq_len=32768,
            n_layers=24,
            n_heads=16,
            d_model=1024,
            d_ff=4096,
        ),
        training=TrainConfig(batch_size=64, epochs=1),
    )
    result = estimate_training(
        spec,
        dataset_meta={
            "train_count": 64,
            "max_example_tokens": 32768,
            "max_points": 240,
            "max_seq_len": 32768,
        },
    )
    assert result["level"] == "block"
    assert any("Attention" in warning for warning in result["warnings"])


def test_training_estimate_uses_dataset_tokens_and_batches() -> None:
    spec = TrainingSpec(
        data=DataBuildSpec(max_seq_len=2048),
        model=ModelSpec(max_seq_len=2048, n_layers=2, n_heads=4, d_model=64, d_ff=128),
        training=TrainConfig(batch_size=4, epochs=1),
    )
    meta = {
        "train_count": 10,
        "max_example_tokens": 300,
        "avg_example_tokens": 200.0,
        "max_points": 40,
        "avg_points": 30.0,
        "max_seq_len": 512,
    }
    result = estimate_training(spec, dataset_meta=meta)
    estimates = result["estimates"]
    assert estimates["seq_len"] == 300
    assert estimates["train_examples"] == 10
    assert estimates["batches_per_epoch"] == 3
    assert estimates["tokens_per_batch"] == 1200
    assert estimates["max_points_per_example"] == 40
    assert estimates["avg_tokens_per_example"] == 200.0


def test_training_estimate_small_is_ok() -> None:
    spec = TrainingSpec(
        data=DataBuildSpec(max_seq_len=512, n_points=8, precision=2),
        model=ModelSpec(max_seq_len=512, n_layers=1, n_heads=2, d_model=32, d_ff=64),
        training=TrainConfig(batch_size=2, epochs=1),
    )
    assert estimate_training(spec)["level"] == "ok"
