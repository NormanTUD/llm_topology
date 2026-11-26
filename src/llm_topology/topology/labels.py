from __future__ import annotations

from collections.abc import Sequence

from ..config import (
    FIELD_CODES,
    METHOD_CODES,
    DatasetSpec,
    GeneratedPointCloud,
    PersistenceSummary,
    TargetSchema,
)
from .generators import GeneratorError


class TargetBuildError(ValueError):
    pass


def _betti_for_field(sample: GeneratedPointCloud, field: str) -> list[int]:
    if field == "Z2":
        return list(sample.theoretical_betti)
    return list(sample.theoretical_betti_q)


def _value_for_field(
    block_name: str,
    field_name: str,
    sample: GeneratedPointCloud,
    persistence: PersistenceSummary,
    spec: DatasetSpec,
) -> float:
    if block_name == "betti":
        values = _betti_for_field(sample, persistence.field.value)
        try:
            index = int(field_name.removeprefix("b"))
        except ValueError as exc:
            raise TargetBuildError(f"unknown betti field: {field_name!r}") from exc
        if not (0 <= index < len(values)):
            raise TargetBuildError(f"betti field {field_name!r} exceeds the configured max_dim")
        return float(values[index])
    if block_name == "ph":
        try:
            index = int(field_name.removeprefix("b"))
        except ValueError as exc:
            raise TargetBuildError(f"unknown ph field: {field_name!r}") from exc
        if not (0 <= index < len(persistence.betti)):
            raise TargetBuildError(f"ph field {field_name!r} exceeds the configured max_dim")
        return float(persistence.betti[index])
    if block_name == "genus":
        return float(sample.genus)
    if block_name == "n_components":
        return float(sample.n_components)
    if block_name == "structure_id":
        return float(sample.structure_id)
    if block_name == "n_points":
        return float(sample.n_points)
    if block_name == "ambient_dim":
        return float(sample.ambient_dim)
    if block_name == "filtration":
        mapping = {
            "method_code": float(METHOD_CODES[persistence.method]),
            "maxdim": float(persistence.max_dim),
            "epsilon": float(persistence.epsilon),
            "field_code": float(FIELD_CODES[persistence.field]),
        }
        if field_name not in mapping:
            raise TargetBuildError(f"unknown filtration field: {field_name!r}")
        return mapping[field_name]
    raise TargetBuildError(f"unknown target block: {block_name!r}")


def build_target_groups(
    sample: GeneratedPointCloud,
    persistence: PersistenceSummary,
    schema: TargetSchema,
    spec: DatasetSpec,
) -> list[list[float]]:
    groups: list[list[float]] = []
    for block in schema.blocks:
        group = [
            _value_for_field(block.name, field_name, sample, persistence, spec)
            for field_name in block.fields
        ]
        groups.append(group)
    return groups


def schema_for_max_dim(schema: TargetSchema, max_dim: int) -> TargetSchema:
    """Return a copy whose betti/ph blocks match the selected homology dimension."""

    blocks = []
    for block in schema.blocks:
        if block.name in {"betti", "ph"}:
            fields = [f"b{i}" for i in range(max_dim + 1)]
            blocks.append(block.model_copy(update={"fields": fields}))
        else:
            blocks.append(block.model_copy())
    return TargetSchema(blocks=blocks)


def target_blocks_hint(schema: TargetSchema) -> str:
    return " | ".join(
        f"{block.name}: {', '.join(block.fields)}" for block in schema.blocks
    )


def build_dataset_response(
    sample: GeneratedPointCloud,
    persistence: PersistenceSummary,
    schema: TargetSchema,
    spec: DatasetSpec,
    tokenizer,
) -> tuple[list[list[float]], str, str]:
    groups = build_target_groups(sample, persistence, schema, spec)
    target_text = tokenizer.target_text(groups)
    sequence = tokenizer.sequence(sample.points, groups)
    return groups, target_text, sequence
