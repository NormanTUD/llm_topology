from __future__ import annotations

import math

import numpy as np
import pytest

from llm_topology.config import (
    CoefficientField,
    DatasetSpec,
    FiltrationMethod,
    FiltrationSpec,
    StructureKind,
)
from llm_topology.topology.filtration import (
    FiltrationError,
    betti_at,
    compute_persistence,
)
from llm_topology.topology.generators import generate_point_cloud


def _sample(kind: StructureKind, n_points: int, seed: int, ambient_dim: int = 3):
    return generate_point_cloud(
        DatasetSpec(
            structure=kind,
            n_points=n_points,
            ambient_dim=ambient_dim,
            seed=seed,
        )
    )


def test_circle_alpha_recovers_h1() -> None:
    sample = _sample(StructureKind.CIRCLE, 180, 2)
    result = compute_persistence(
        sample.points,
        FiltrationSpec(method=FiltrationMethod.ALPHA, max_dim=2, field=CoefficientField.Z2),
    )
    assert result.betti[0] == 1
    assert result.betti[1] == 1
    assert result.betti[2] == 0
    assert 0 < result.epsilon < 10


def test_sphere_alpha_recovers_h2() -> None:
    sample = _sample(StructureKind.SPHERE, 240, 2)
    result = compute_persistence(
        sample.points,
        FiltrationSpec(method=FiltrationMethod.ALPHA, max_dim=2),
    )
    assert result.betti[0] == 1
    assert result.betti[1] == 0
    assert result.betti[2] == 1


def test_torus_alpha_recovers_h1_and_h2() -> None:
    sample = _sample(StructureKind.TORUS, 500, 2)
    result = compute_persistence(
        sample.points,
        FiltrationSpec(method=FiltrationMethod.ALPHA, max_dim=2),
    )
    assert result.betti == [1, 2, 1]


def test_z3_field_runs() -> None:
    sample = _sample(StructureKind.CIRCLE, 120, 4)
    result = compute_persistence(
        sample.points,
        FiltrationSpec(method=FiltrationMethod.ALPHA, field=CoefficientField.Z3),
    )
    assert result.field is CoefficientField.Z3
    assert result.betti[0] == 1
    assert result.betti[1] >= 1


def test_betti_at_counts_active_intervals() -> None:
    diagrams = [
        [(0.0, None), (0.0, 1.0)],
        [(0.5, 1.5), (0.2, 0.3)],
        [(2.0, None)],
    ]
    assert betti_at(diagrams, 0.0) == [2, 0, 0]
    assert betti_at(diagrams, 0.6) == [2, 1, 0]
    assert betti_at(diagrams, 1.1) == [1, 1, 0]
    assert betti_at(diagrams, 2.1) == [1, 0, 1]


def test_rips_circle_runs() -> None:
    sample = _sample(StructureKind.CIRCLE, 90, 3)
    result = compute_persistence(
        sample.points,
        FiltrationSpec(
            method=FiltrationMethod.RIPS,
            max_dim=1,
            max_edge_length=1.2,
        ),
    )
    assert result.betti[0] == 1
    assert result.betti[1] >= 1
    assert result.epsilon_grid
    assert len(result.epsilon_grid) == len(result.betti_curve)


def test_manual_epsilon_is_used() -> None:
    sample = _sample(StructureKind.CIRCLE, 140, 3)
    result = compute_persistence(
        sample.points,
        FiltrationSpec(method=FiltrationMethod.ALPHA, epsilon=0.5),
    )
    assert math.isclose(result.epsilon, 0.5, rel_tol=0, abs_tol=1e-12)


def test_invalid_points_are_rejected() -> None:
    with pytest.raises(FiltrationError):
        compute_persistence([[0.0, 0.0]])
    with pytest.raises(FiltrationError):
        compute_persistence([[0.0, 0.0], [np.nan, 1.0]])
