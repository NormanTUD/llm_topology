from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from .model.config import AnalysisSpec, DataBuildSpec, ModelSpec, TrainConfig, TrainingSpec


class ExperimentConfig(BaseModel):
    name: str = "experiment"
    notes: str = ""
    dataset_id: str | None = None
    data: DataBuildSpec | None = None
    model: ModelSpec = Field(default_factory=ModelSpec)
    training: TrainConfig = Field(default_factory=TrainConfig)
    analysis: AnalysisSpec | None = None
    run_analysis: bool = True
    output_dir: str = "runs"

    def to_training_spec(self) -> TrainingSpec:
        return TrainingSpec(
            dataset_id=self.dataset_id,
            data=self.data or DataBuildSpec(),
            model=self.model,
            training=self.training,
        )

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, text: str) -> ExperimentConfig:
        payload = yaml.safe_load(text)
        if not isinstance(payload, dict):
            raise ValueError("experiment config must be a YAML mapping")
        return cls.model_validate(payload)

    @classmethod
    def from_file(cls, path: str | Path) -> ExperimentConfig:
        return cls.from_yaml(Path(path).read_text(encoding="utf-8"))


def config_from_training_spec(
    spec: TrainingSpec,
    *,
    name: str = "experiment",
    notes: str = "",
    run_analysis: bool = True,
) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        notes=notes,
        dataset_id=spec.dataset_id,
        data=None if spec.dataset_id else spec.data,
        model=spec.model,
        training=spec.training,
        run_analysis=run_analysis,
    )
