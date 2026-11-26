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
    result = estimate_training(spec)
    assert result["level"] == "block"
    assert any("Attention" in warning for warning in result["warnings"])


def test_training_estimate_small_is_ok() -> None:
    spec = TrainingSpec(
        data=DataBuildSpec(max_seq_len=512),
        model=ModelSpec(max_seq_len=512, n_layers=1, n_heads=2, d_model=32, d_ff=64),
        training=TrainConfig(batch_size=2, epochs=1),
    )
    assert estimate_training(spec)["level"] == "ok"
