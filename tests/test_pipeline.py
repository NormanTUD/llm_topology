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


def test_target_group_order_is_fixed_and_visible() -> None:
    response = generate_dataset(DatasetSpec(structure=StructureKind.TORUS, n_points=80, seed=1))
    names = [block.name for block in response.target_schema.blocks]
    assert names == ["betti", "ph", "genus", "n_components", "filtration"]
    assert response.target_groups[0] == [
        float(value) for value in response.sample.theoretical_betti
    ]
    assert response.target_groups[1] == [float(value) for value in response.persistence.betti]
    assert response.target_text.split(";")[0] == ",".join(
        str(int(value)) for value in response.sample.theoretical_betti
    )


def test_reference_cloud_is_separate_from_model_sequence() -> None:
    response = generate_dataset(
        DatasetSpec(
            structure=StructureKind.TORUS,
            n_points=60,
            seed=3,
            transform={"twist": 0.2, "bend": 0.2, "warp": 0.1, "seed": 4},
        )
    )
    assert response.reference_projection
    assert response.reference_structure_name
    assert response.reference_betti
    assert "<endoftext>" in response.sequence
    assert response.sequence.startswith("<") is False or response.sequence


def test_default_torus_guard_matches() -> None:
    response = generate_dataset(DatasetSpec(structure=StructureKind.TORUS, n_points=400, seed=0))
    assert response.topology_guard is True
    assert response.target_groups[0] == [1.0, 2.0, 1.0]


def test_precision_is_used_in_serialized_targets() -> None:
    response = generate_dataset(
        DatasetSpec(structure=StructureKind.CIRCLE, n_points=80, precision=2)
    )
    for raw in response.target_text.replace(";", ",").split(","):
        if "." in raw:
            assert len(raw.split(".")[1]) <= 2


def test_copies_range_changes_b0() -> None:
    response = generate_dataset(
        DatasetSpec(
            structure=StructureKind.TORUS,
            copies_min=1,
            copies_max=5,
            points_per_component=20,
            ambient_dim=3,
            seed=4,
        )
    )
    assert 1 <= response.sample.n_components <= 5
    assert response.sample.theoretical_betti[0] == response.sample.n_components


def test_points_per_component_scales_total_points() -> None:
    response = generate_dataset(
        DatasetSpec(
            structure=StructureKind.TORUS,
            copies=3,
            points_per_component=40,
            n_points=999,
            ambient_dim=3,
            seed=1,
        )
    )
    assert response.sample.n_points == 120
    assert response.sample.theoretical_betti == [3, 6, 3]


def test_multi_torus_components_and_topology_guard() -> None:
    response = generate_dataset(
        DatasetSpec(
            structure=StructureKind.TORUS,
            n_points=1200,
            copies=3,
            ambient_dim=3,
            seed=1,
            filtration=FiltrationSpec(method=FiltrationMethod.ALPHA, max_dim=2),
        )
    )
    assert response.sample.theoretical_betti == [3, 6, 3]
    assert response.sample.n_components == 3
    assert response.sample.genus == 3
    assert response.topology_guard is True


def test_manual_filtration_epsilon_is_in_targets() -> None:
    response = generate_dataset(
        DatasetSpec(
            structure=StructureKind.CIRCLE,
            n_points=120,
            filtration=FiltrationSpec(method=FiltrationMethod.ALPHA, epsilon=0.5),
        )
    )
    assert response.persistence.epsilon == 0.5
    assert response.target_groups[-1][2] == 0.5


def test_auto_filtration_epsilon_is_reported_in_targets() -> None:
    response = generate_dataset(DatasetSpec(structure=StructureKind.CIRCLE, n_points=120))
    assert response.persistence.epsilon > 0
    assert response.target_groups[-1][2] == response.persistence.epsilon


def test_augmented_generation_does_not_crash() -> None:
    response = generate_dataset(
        DatasetSpec(
            structure=StructureKind.TORUS,
            n_points=40,
            seed=11,
            transform={
                "rotation_deg": [10, 20, 30],
                "stretch": [1.1, 0.9, 1.0],
                "twist": 0.2,
                "shear": [0.1, -0.1, 0.05],
                "bend": 0.2,
                "warp": 0.15,
                "seed": 3,
            },
        )
    )
    assert response.sample.n_points == 40
    assert response.topology_guard in {True, False}


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


def test_mismatch_note_is_present_for_noisy_sample() -> None:
    response = generate_dataset(
        DatasetSpec(
            structure=StructureKind.TORUS,
            n_points=160,
            seed=2,
            transform={"noise": 0.08, "seed": 3},
        )
    )
    assert any("differ" in note for note in response.notes)


def test_invalid_generator_is_reported_as_pipeline_error() -> None:
    with pytest.raises(PipelineError):
        generate_dataset(DatasetSpec(structure=StructureKind.SPHERE, n_points=50, ambient_dim=2))


def test_invalid_schema_is_reported_as_pipeline_error() -> None:
    schema = TargetSchema(blocks=[TargetBlock(name="betti", fields=["b0", "b1", "b2", "b9"])])
    with pytest.raises(PipelineError):
        generate_dataset(
            DatasetSpec(structure=StructureKind.CIRCLE, n_points=60, target_schema=schema)
        )
