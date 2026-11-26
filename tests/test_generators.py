from __future__ import annotations

import numpy as np
import pytest

from llm_topology.config import DatasetSpec, StructureKind, TransformSpec
from llm_topology.topology.generators import (
    STRUCTURE_IDS,
    GeneratorError,
    generate_point_cloud,
    theoretical_topology,
)


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (StructureKind.CIRCLE, [1, 1, 0]),
        (StructureKind.SPHERE, [1, 0, 1]),
        (StructureKind.TORUS, [1, 2, 1]),
        (StructureKind.GENUS_SURFACE, [1, 4, 1]),
        (StructureKind.LINKED_RINGS, [2, 2, 0]),
        (StructureKind.FIGURE_EIGHT, [1, 2, 0]),
        (StructureKind.KLEIN_BOTTLE, [1, 2, 1]),
        (StructureKind.MOBIUS_STRIP, [1, 1, 0]),
        (StructureKind.ANNULUS, [1, 1, 0]),
        (StructureKind.NOISE, [1, 0, 0]),
        (StructureKind.BALL, [1, 0, 0]),
    ],
)
def test_theoretical_labels(kind: StructureKind, expected: list[int]) -> None:
    sample = generate_point_cloud(
        DatasetSpec(
            structure=kind,
            n_points=80,
            ambient_dim=4 if kind is StructureKind.KLEIN_BOTTLE else 3,
            genus=2,
            seed=7,
        )
    )
    assert sample.theoretical_betti == expected
    assert sample.structure_id == STRUCTURE_IDS[kind]


def test_generation_is_deterministic() -> None:
    spec = DatasetSpec(structure=StructureKind.TORUS, n_points=64, seed=42)
    first = generate_point_cloud(spec)
    second = generate_point_cloud(spec)
    assert np.allclose(first.points, second.points)


@pytest.mark.parametrize("kind", list(StructureKind))
def test_all_structures_generate_finite_points(kind: StructureKind) -> None:
    ambient = 4 if kind is StructureKind.KLEIN_BOTTLE else 3
    spec = DatasetSpec(
        structure=kind,
        n_points=48,
        ambient_dim=ambient,
        genus=2,
        copies=1,
        seed=3,
        params={"base_structure": "torus"} if kind is StructureKind.DISJOINT else {},
    )
    sample = generate_point_cloud(spec)
    points = np.asarray(sample.points)
    assert points.shape == (48, ambient)
    assert np.all(np.isfinite(points))
    assert sample.n_points == 48


def test_copies_multiply_topology() -> None:
    spec = DatasetSpec(
        structure=StructureKind.TORUS,
        n_points=90,
        copies=3,
        ambient_dim=3,
        seed=1,
    )
    sample = generate_point_cloud(spec)
    assert sample.n_components == 3
    assert sample.theoretical_betti == [3, 6, 3]
    assert len(sample.points) == 90


def test_disjoint_uses_base_structure() -> None:
    spec = DatasetSpec(
        structure=StructureKind.DISJOINT,
        n_points=90,
        copies=3,
        ambient_dim=3,
        seed=1,
        params={"base_structure": "sphere"},
    )
    sample = generate_point_cloud(spec)
    assert sample.metadata["base_structure"] == "sphere"
    assert sample.n_components == 3
    assert sample.theoretical_betti == [3, 0, 3]


def test_rotation_does_not_change_topology_labels() -> None:
    spec = DatasetSpec(
        structure=StructureKind.TORUS,
        n_points=60,
        seed=5,
        transform=TransformSpec(rotation_deg=(20, 35, 10), seed=2),
    )
    sample = generate_point_cloud(spec)
    assert sample.theoretical_betti == [1, 2, 1]


def test_permutation_changes_order_not_shape() -> None:
    base = generate_point_cloud(DatasetSpec(structure=StructureKind.CIRCLE, n_points=40, seed=4))
    permuted = generate_point_cloud(
        DatasetSpec(
            structure=StructureKind.CIRCLE,
            n_points=40,
            seed=4,
            transform=TransformSpec(permute=True, seed=9),
        )
    )
    assert not np.allclose(base.points, permuted.points)
    assert np.allclose(
        np.sort(np.asarray(base.points), axis=0),
        np.sort(np.asarray(permuted.points), axis=0),
    )


def test_noise_is_applied() -> None:
    base = generate_point_cloud(DatasetSpec(structure=StructureKind.CIRCLE, n_points=30, seed=4))
    noisy = generate_point_cloud(
        DatasetSpec(
            structure=StructureKind.CIRCLE,
            n_points=30,
            seed=4,
            transform=TransformSpec(noise=0.1, seed=9),
        )
    )
    assert not np.allclose(base.points, noisy.points)


def test_ambient_dimension_validation() -> None:
    with pytest.raises(GeneratorError):
        generate_point_cloud(
            DatasetSpec(structure=StructureKind.SPHERE, n_points=30, ambient_dim=2)
        )
    with pytest.raises(GeneratorError):
        generate_point_cloud(
            DatasetSpec(structure=StructureKind.KLEIN_BOTTLE, n_points=30, ambient_dim=3)
        )


def test_invalid_disjoint_base() -> None:
    with pytest.raises(GeneratorError):
        generate_point_cloud(
            DatasetSpec(
                structure=StructureKind.DISJOINT,
                n_points=30,
                ambient_dim=3,
                params={"base_structure": "not_a_structure"},
            )
        )


def test_theoretical_topology_klein_q_variant() -> None:
    topology = theoretical_topology(StructureKind.KLEIN_BOTTLE)
    assert topology.betti_z2 == (1, 2, 1)
    assert topology.betti_q == (1, 1, 0)
