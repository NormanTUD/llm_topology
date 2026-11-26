from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class StructureKind(StrEnum):
    CIRCLE = "circle"
    SPHERE = "sphere"
    TORUS = "torus"
    GENUS_SURFACE = "genus_surface"
    DISJOINT = "disjoint"
    LINKED_RINGS = "linked_rings"
    FIGURE_EIGHT = "figure_eight"
    KLEIN_BOTTLE = "klein_bottle"
    MOBIUS_STRIP = "mobius_strip"
    ANNULUS = "annulus"
    NOISE = "noise"
    BALL = "ball"


class FiltrationMethod(StrEnum):
    ALPHA = "alpha"
    RIPS = "rips"


class CoefficientField(StrEnum):
    Z2 = "Z2"
    Z3 = "Z3"


METHOD_CODES = {FiltrationMethod.ALPHA: 0, FiltrationMethod.RIPS: 1}
FIELD_CODES = {CoefficientField.Z2: 2, CoefficientField.Z3: 3}
PRIME_BY_FIELD = {CoefficientField.Z2: 2, CoefficientField.Z3: 3}


class TransformSpec(BaseModel):
    rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    stretch: tuple[float, float, float] = (1.0, 1.0, 1.0)
    twist: float = 0.0
    shear: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bend: float = 0.0
    warp: float = 0.0
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    noise: float = 0.0
    permute: bool = False
    seed: int = 0

    @field_validator("stretch")
    @classmethod
    def _positive_stretch(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if any(v <= 0 for v in value):
            raise ValueError("stretch values must be positive")
        return value

    @field_validator("noise")
    @classmethod
    def _non_negative_noise(cls, value: float) -> float:
        if value < 0:
            raise ValueError("noise must be non-negative")
        return value


class FiltrationSpec(BaseModel):
    method: FiltrationMethod = FiltrationMethod.ALPHA
    max_dim: int = Field(default=2, ge=0, le=6)
    field: CoefficientField = CoefficientField.Z2
    epsilon: float | None = Field(default=None, ge=0)
    max_edge_length: float | None = Field(default=None, gt=0)
    rips_threshold_quantile: float = Field(default=0.35, gt=0, le=1)


class TargetBlock(BaseModel):
    name: Literal[
        "betti",
        "ph",
        "genus",
        "n_components",
        "filtration",
        "structure_id",
        "n_points",
        "ambient_dim",
    ]
    fields: list[str]


def default_target_blocks(max_dim: int = 2) -> list[TargetBlock]:
    betti_fields = [f"b{i}" for i in range(max_dim + 1)]
    return [
        TargetBlock(name="betti", fields=betti_fields),
        TargetBlock(name="ph", fields=betti_fields),
        TargetBlock(name="genus", fields=["genus"]),
        TargetBlock(name="n_components", fields=["n_components"]),
        TargetBlock(
            name="filtration",
            fields=["method_code", "maxdim", "epsilon", "field_code"],
        ),
    ]


def optional_target_blocks() -> list[TargetBlock]:
    return [
        TargetBlock(name="structure_id", fields=["structure_id"]),
        TargetBlock(name="n_points", fields=["n_points"]),
        TargetBlock(name="ambient_dim", fields=["ambient_dim"]),
    ]


class TargetSchema(BaseModel):
    blocks: list[TargetBlock] = Field(default_factory=default_target_blocks)

    @field_validator("blocks")
    @classmethod
    def _unique_blocks(cls, blocks: list[TargetBlock]) -> list[TargetBlock]:
        names = [block.name for block in blocks]
        if len(names) != len(set(names)):
            raise ValueError("target blocks must be unique")
        return blocks

    @property
    def names(self) -> list[str]:
        return [block.name for block in self.blocks]

    @property
    def flat_fields(self) -> list[str]:
        return [
            f"{block.name}.{field_name}" for block in self.blocks for field_name in block.fields
        ]


class DatasetSpec(BaseModel):
    structure: StructureKind = StructureKind.CIRCLE
    n_points: int = Field(default=240, ge=4)
    ambient_dim: int = Field(default=3, ge=1, le=64)
    copies: int = Field(default=1, ge=1, le=1000)
    seed: int = 0
    genus: int = Field(default=2, ge=1, le=32)
    params: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    transform: TransformSpec = Field(default_factory=TransformSpec)
    filtration: FiltrationSpec = Field(default_factory=FiltrationSpec)
    target_schema: TargetSchema = Field(default_factory=TargetSchema)

    @field_validator("ambient_dim")
    @classmethod
    def _ambient_fits_known_structures(cls, value: int) -> int:
        if value < 2:
            raise ValueError("ambient_dim must be at least 2 for the current catalog")
        return value


class GeneratedPointCloud(BaseModel):
    points: list[list[float]]
    structure: StructureKind
    structure_id: int
    structure_name: str
    theoretical_betti: list[int]
    theoretical_betti_q: list[int]
    genus: int
    n_components: int
    n_points: int
    ambient_dim: int
    metadata: dict[str, float | int | str | bool | None] = Field(default_factory=dict)


class PersistenceSummary(BaseModel):
    method: FiltrationMethod
    field: CoefficientField
    max_dim: int
    epsilon: float
    betti: list[int]
    diagrams: list[list[list[float | None]]]
    finite_values: list[float]
    epsilon_grid: list[float] = Field(default_factory=list)
    betti_curve: list[list[int]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DatasetResponse(BaseModel):
    sample: GeneratedPointCloud
    persistence: PersistenceSummary
    target_groups: list[list[float]]
    target_text: str
    sequence: str
    target_schema: TargetSchema
    projection: list[list[float]] = Field(default_factory=list)
    projection_method: str = "first-3"
    reference_projection: list[list[float]] = Field(default_factory=list)
    reference_structure_name: str | None = None
    reference_betti: list[int] = Field(default_factory=list)
    topology_guard: bool = False
    topology_guard_label: str = "unknown"
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class BatchDatasetResponse(BaseModel):
    items: list[DatasetResponse]
    warnings: list[str] = Field(default_factory=list)
    count: int
    mode: str = "single"
    structures: list[str] = Field(default_factory=list)
