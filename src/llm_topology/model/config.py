from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator

from ..config import FiltrationSpec, StructureKind, TargetSchema


class InitScheme(StrEnum):
    NORMAL = "normal"
    XAVIER_UNIFORM = "xavier_uniform"
    XAVIER_NORMAL = "xavier_normal"
    ORTHOGONAL = "orthogonal"
    UNIFORM = "uniform"
    ZERO = "zero"


class ActivationKind(StrEnum):
    GELU = "gelu"
    SILU = "silu"
    RELU = "relu"


class NormPlacement(StrEnum):
    PRE = "pre"
    POST = "post"


class PositionEncoding(StrEnum):
    LEARNED = "learned"
    ROPE = "rope"
    NONE = "none"


class InitSpec(BaseModel):
    scheme: InitScheme = InitScheme.NORMAL
    std: float = Field(default=0.02, gt=0)
    mean: float = 0.0
    seed: int = 0

    @field_validator("std")
    @classmethod
    def _positive_std(cls, value: float) -> float:
        return value


class ModelSpec(BaseModel):
    max_seq_len: int = Field(default=1024, ge=32, le=32768)
    n_layers: int = Field(default=3, ge=1, le=24)
    n_heads: int = Field(default=4, ge=1, le=16)
    d_model: int = Field(default=96, ge=8, le=1024)
    d_ff: int = Field(default=384, ge=8, le=4096)
    dropout: float = Field(default=0.0, ge=0, le=0.8)
    activation: ActivationKind = ActivationKind.SILU
    norm_placement: NormPlacement = NormPlacement.PRE
    position_encoding: PositionEncoding = PositionEncoding.LEARNED
    tie_embeddings: bool = True
    use_bias: bool = True
    init: InitSpec = Field(default_factory=InitSpec)

    @field_validator("d_model")
    @classmethod
    def _model_size(cls, value: int) -> int:
        if value % 2 != 0:
            raise ValueError("d_model must be even for the attention head split")
        return value

    def validate_heads(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

    @property
    def head_dim(self) -> int:
        self.validate_heads()
        return self.d_model // self.n_heads


class DataBuildSpec(BaseModel):
    structures: list[StructureKind] = Field(
        default_factory=lambda: [
            StructureKind.CIRCLE,
            StructureKind.SPHERE,
            StructureKind.TORUS,
            StructureKind.GENUS_SURFACE,
        ]
    )
    n_train: int = Field(default=64, ge=4, le=5000)
    n_val: int = Field(default=24, ge=2, le=1000)
    n_points: int = Field(default=32, ge=8, le=512)
    ambient_dim: int = Field(default=3, ge=2, le=12)
    copies: int = Field(default=1, ge=1, le=32)
    genus: int = Field(default=2, ge=1, le=16)
    precision: int = Field(default=3, ge=1, le=8)
    max_seq_len: int = Field(default=1024, ge=64, le=32768)
    task_mix: float = Field(default=0.5, ge=0, le=1)
    augment: bool = True
    augment_strength: float = Field(default=0.35, ge=0, le=1)
    seed: int = 0
    filtration: FiltrationSpec = Field(default_factory=FiltrationSpec)
    target_schema: TargetSchema = Field(default_factory=TargetSchema)

    @field_validator("structures")
    @classmethod
    def _has_structures(cls, value: list[StructureKind]) -> list[StructureKind]:
        if not value:
            raise ValueError("select at least one structure")
        return value


class TrainConfig(BaseModel):
    epochs: int = Field(default=6, ge=1, le=1000)
    batch_size: int = Field(default=8, ge=1, le=512)
    lr: float = Field(default=3e-4, gt=0, le=10)
    weight_decay: float = Field(default=0.01, ge=0, le=10)
    optimizer: str = Field(default="adamw", pattern="^(adamw|adam|sgd)$")
    warmup_steps: int = Field(default=20, ge=0, le=100000)
    grad_clip: float = Field(default=1.0, ge=0, le=1000)
    patience: int = Field(default=4, ge=1, le=1000)
    seed: int = 7
    aux_weight: float = Field(default=0.2, ge=0, le=10)
    ranking_weight: float = Field(default=0.05, ge=0, le=10)
    ranking_margin: float = Field(default=0.5, ge=0, le=100)
    mc_negatives: int = Field(default=2, ge=1, le=16)
    device: str = "auto"
    checkpoint_root: str = "runs"
    max_new_tokens: int = Field(default=80, ge=1, le=2048)
    eval_every_batches: int = Field(default=20, ge=1, le=10000)

    @field_validator("device")
    @classmethod
    def _device(cls, value: str) -> str:
        if value not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda, or mps")
        return value


class TrainingSpec(BaseModel):
    data: DataBuildSpec = Field(default_factory=DataBuildSpec)
    model: ModelSpec = Field(default_factory=ModelSpec)
    training: TrainConfig = Field(default_factory=TrainConfig)

    @model_validator(mode="after")
    def _validate_shapes(self) -> TrainingSpec:
        self.model.validate_heads()
        if self.model.max_seq_len < self.data.max_seq_len:
            raise ValueError("model.max_seq_len must be at least data.max_seq_len")
        return self


class AnalysisSpec(BaseModel):
    max_examples: int = Field(default=96, ge=8, le=1024)
    probe_alpha: float = Field(default=1.0, ge=0, le=10000)
    attention_example: int = Field(default=0, ge=0, le=1024)
    pca_dim: int = Field(default=3, ge=2, le=3)
    layer: int | None = Field(default=None, ge=0, le=64)
    top_neurons: int = Field(default=12, ge=1, le=128)
