from __future__ import annotations

import pytest

from llm_topology.config import (
    DatasetSpec,
    FiltrationMethod,
    FiltrationSpec,
    StructureKind,
    TargetBlock,
    TargetSchema,
)
from llm_topology.pipeline import PipelineError, generate_dataset
from llm_topology.tokenizer import DEFAULT_BOUNDARY


def test_default_pipeline_returns_all_pieces() -> None:
    response = generate_dataset(DatasetSpec(structure=StructureKind.CIRCLE, n_points=100, seed=1))
    assert response.sample.n_points == 100
    assert len(response.target_groups) == len(response.target_schema.blocks)
    assert response.target_text.count(";") == len(response.target_schema.blocks) - 1
    assert DEFAULT_BOUNDARY in response.sequence
    assert (
        response.sequence.startswith("0.")
        or response.sequence.startswith("-0.")
        or "," in response.sequence
    )
    assert response.persistence.betti[0] == 1


def test_custom_schema_and_max_dim() -> None:
    schema = TargetSchema(
        blocks=[
            TargetBlock(name="ph", fields=["b0", "b1", "b2", "b3"]),
            TargetBlock(name="genus", fields=["genus"]),
        ]
    )
    response = generate_dataset(
        DatasetSpec(
            structure=StructureKind.TORUS,
            n_points=120,
            target_schema=schema,
            filtration=FiltrationSpec(method=FiltrationMethod.ALPHA, max_dim=3),
        )
    )
    assert response.target_schema.blocks[0].fields == ["b0", "b1", "b2", "b3"]
    assert len(response.target_groups[0]) == 4
    assert len(response.target_groups[1]) == 1


def test_optional_target_groups() -> None:
    schema = TargetSchema(
        blocks=[
            TargetBlock(name="structure_id", fields=["structure_id"]),
            TargetBlock(name="n_points", fields=["n_points"]),
            TargetBlock(name="ambient_dim", fields=["ambient_dim"]),
        ]
    )
    response = generate_dataset(
        DatasetSpec(structure=StructureKind.CIRCLE, n_points=40, target_schema=schema)
    )
    assert [group[0] for group in response.target_groups] == [0.0, 40.0, 3.0]


def test_mismatch_warning_is_present_for_noisy_sample() -> None:
    response = generate_dataset(
        DatasetSpec(
            structure=StructureKind.TORUS,
            n_points=160,
            seed=2,
            transform={"noise": 0.08, "seed": 3},
        )
    )
    assert any("differs" in warning for warning in response.warnings)


def test_invalid_generator_is_reported_as_pipeline_error() -> None:
    with pytest.raises(PipelineError):
        generate_dataset(DatasetSpec(structure=StructureKind.SPHERE, n_points=50, ambient_dim=2))


def test_invalid_schema_is_reported_as_pipeline_error() -> None:
    schema = TargetSchema(blocks=[TargetBlock(name="betti", fields=["b0", "b1", "b2", "b9"])])
    with pytest.raises(PipelineError):
        generate_dataset(
            DatasetSpec(structure=StructureKind.CIRCLE, n_points=60, target_schema=schema)
        )
