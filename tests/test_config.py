from __future__ import annotations

from llm_topology.experiment import ExperimentConfig
from llm_topology.model.config import DataBuildSpec, ModelSpec, TrainConfig, TrainingSpec


def test_config_yaml_roundtrip() -> None:
    spec = TrainingSpec(
        data=DataBuildSpec(n_train=4, n_val=2, n_test=2, max_seq_len=512),
        model=ModelSpec(max_seq_len=512, n_layers=1, n_heads=2, d_model=16, d_ff=32),
        training=TrainConfig(epochs=1, batch_size=2, device="cpu"),
    )
    config = ExperimentConfig(
        name="roundtrip", data=spec.data, model=spec.model, training=spec.training
    )
    restored = ExperimentConfig.from_yaml(config.to_yaml())
    assert restored.name == "roundtrip"
    assert restored.model.d_model == 16
    assert restored.data is not None
