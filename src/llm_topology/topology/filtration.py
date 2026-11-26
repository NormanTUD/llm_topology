from __future__ import annotations

import math
from collections.abc import Sequence

import gudhi
import numpy as np
from ripser import ripser

from ..config import (
    PRIME_BY_FIELD,
    CoefficientField,
    FiltrationMethod,
    FiltrationSpec,
    PersistenceSummary,
)


class FiltrationError(ValueError):
    pass


def _as_points(points: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2:
        raise FiltrationError("points must be a two-dimensional array")
    if array.shape[0] < 2:
        raise FiltrationError("at least two points are required")
    if not np.all(np.isfinite(array)):
        raise FiltrationError("points must contain only finite coordinates")
    return array


def _clean_diagram(diagram: np.ndarray) -> list[tuple[float, float | None]]:
    cleaned: list[tuple[float, float | None]] = []
    for birth, death in diagram:
        birth = float(birth)
        if math.isfinite(float(death)):
            death_value: float | None = float(death)
        else:
            death_value = None
        cleaned.append((birth, death_value))
    return cleaned


def _alpha_diagrams(
    points: np.ndarray,
    max_dim: int,
    field: CoefficientField,
) -> tuple[list[list[tuple[float, float | None]]], float, list[str]]:
    warnings: list[str] = []
    try:
        st = gudhi.AlphaComplex(points=points).create_simplex_tree()
    except Exception as exc:  # pragma: no cover - depends on native geometry failures
        raise FiltrationError(f"alpha complex construction failed: {exc}") from exc
    try:
        pairs = st.persistence(homology_coeff_field=PRIME_BY_FIELD[field])
    except Exception as exc:  # pragma: no cover - native failures
        raise FiltrationError(f"alpha persistence failed: {exc}") from exc

    filtration_values = [float(value) for _, value in st.get_filtration()]
    max_filtration = max(filtration_values) if filtration_values else 0.0
    bbox = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    raw_max_epsilon = math.sqrt(max(max_filtration, 0.0))
    max_epsilon = max(min(raw_max_epsilon, 1.5 * max(bbox, 1e-9)), 1e-9)

    diagrams: list[list[tuple[float, float | None]]] = [[] for _ in range(max_dim + 1)]
    for dimension, (birth, death) in pairs:
        if dimension > max_dim:
            continue
        birth_radius = math.sqrt(max(float(birth), 0.0))
        death_radius = (
            None if not math.isfinite(float(death)) else math.sqrt(max(float(death), 0.0))
        )
        diagrams[dimension].append((birth_radius, death_radius))
    for dimension, diagram in enumerate(diagrams):
        diagrams[dimension] = sorted(diagram, key=lambda item: item[0])

    if field is CoefficientField.Z3:
        warnings.append(
            "Z3 is used as a Q-like comparison field; it is not an exact Q computation."
        )
    if any(
        item[1] is None for dimension in range(1, len(diagrams)) for item in diagrams[dimension]
    ):
        warnings.append(
            "Some positive-dimensional classes are infinite; inspect the filtration range."
        )
    return diagrams, max_epsilon, warnings


def _rips_diagrams(
    points: np.ndarray,
    max_dim: int,
    field: CoefficientField,
    spec: FiltrationSpec,
) -> tuple[list[list[tuple[float, float | None]]], float, list[str]]:
    warnings: list[str] = []
    if len(points) > 512:
        warnings.append(
            "Vietoris-Rips with more than 512 points can be very expensive; "
            "reduce points or use alpha filtration."
        )
    if spec.max_edge_length is not None:
        threshold = float(spec.max_edge_length)
    else:
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        upper = distances[np.triu_indices(len(points), k=1)]
        threshold = float(np.quantile(upper, spec.rips_threshold_quantile))
        if threshold <= 0:
            threshold = float(np.max(upper)) if upper.size else 1.0
    bbox = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    threshold = min(threshold, 1.5 * max(bbox, 1e-9))
    try:
        result = ripser(
            points,
            maxdim=max_dim,
            thresh=threshold,
            coeff=PRIME_BY_FIELD[field],
            distance_matrix=False,
        )
    except Exception as exc:  # pragma: no cover - native failures
        raise FiltrationError(f"ripser failed: {exc}") from exc

    diagrams: list[list[tuple[float, float | None]]] = [[] for _ in range(max_dim + 1)]
    for dimension, diagram in enumerate(result["dgms"]):
        if dimension > max_dim:
            continue
        diagrams[dimension] = sorted(_clean_diagram(diagram), key=lambda item: item[0])

    if field is CoefficientField.Z3:
        warnings.append(
            "Z3 is used as a Q-like comparison field; it is not an exact Q computation."
        )
    if any(
        item[1] is None for dimension in range(1, len(diagrams)) for item in diagrams[dimension]
    ):
        warnings.append(
            "Some positive-dimensional Rips classes are infinite at the chosen "
            "max_edge_length; increase the threshold for finite lifetimes."
        )
    return diagrams, threshold, warnings


def betti_at(
    diagrams: Sequence[Sequence[tuple[float, float | None]]],
    epsilon: float,
    *,
    tolerance: float = 1e-9,
    min_lifetime: float = 0.0,
    filter_from_dim: int = 1,
) -> list[int]:
    values: list[int] = []
    for dimension, diagram in enumerate(diagrams):
        count = 0
        for birth, death in diagram:
            if birth - tolerance <= epsilon and (death is None or epsilon < death - tolerance):
                lifetime = float("inf") if death is None else death - birth
                if dimension >= filter_from_dim and lifetime < min_lifetime:
                    continue
                count += 1
        values.append(count)
    return values


def _event_values(
    diagrams: Sequence[Sequence[tuple[float, float | None]]],
    max_epsilon: float,
) -> np.ndarray:
    events = {0.0, max_epsilon}
    for diagram in diagrams:
        for birth, death in diagram:
            if math.isfinite(birth):
                events.add(float(birth))
            if death is not None and math.isfinite(death):
                events.add(float(death))
    return np.asarray(sorted(value for value in events if value <= max_epsilon + 1e-12))


def _candidate_epsilon_values(
    diagrams: Sequence[Sequence[tuple[float, float | None]]],
    max_epsilon: float,
) -> np.ndarray:
    ordered = _event_values(diagrams, max_epsilon)
    if len(ordered) == 1:
        return np.array([0.0, max_epsilon])
    midpoints = (ordered[:-1] + ordered[1:]) / 2
    return np.unique(np.clip(np.concatenate([ordered, midpoints]), 0, max_epsilon))


def select_epsilon(
    diagrams: Sequence[Sequence[tuple[float, float | None]]],
    max_epsilon: float,
) -> float:
    if max_epsilon <= 0:
        return 0.0

    lifetimes: list[tuple[int, float, float, float]] = []
    for dimension in range(1, len(diagrams)):
        for birth, death in diagrams[dimension]:
            if death is None:
                lifetime = max(max_epsilon - birth, 0.0)
                midpoint = min(birth + lifetime / 2, max_epsilon)
            else:
                lifetime = max(death - birth, 0.0)
                midpoint = (birth + death) / 2
            if lifetime > 0:
                lifetimes.append((dimension, lifetime, birth, midpoint))

    if not lifetimes:
        return float(0.5 * max_epsilon)

    max_lifetime = max(item[1] for item in lifetimes)
    significance = max(0.03 * max_lifetime, 1e-9)

    for dimension in range(len(diagrams) - 1, 0, -1):
        candidates = [
            item for item in lifetimes if item[0] == dimension and item[1] >= significance
        ]
        if candidates:
            return float(max(candidates, key=lambda item: item[1])[3])

    candidates = [item for item in lifetimes if item[1] >= significance]
    if candidates:
        return float(max(candidates, key=lambda item: item[1])[3])

    return float(0.5 * max_epsilon)


def _curve_grid(
    diagrams: Sequence[Sequence[tuple[float, float | None]]],
    max_epsilon: float,
    count: int = 64,
) -> np.ndarray:
    if max_epsilon <= 0:
        return np.array([0.0])
    events = _candidate_epsilon_values(diagrams, max_epsilon)
    if len(events) > count:
        indices = np.linspace(0, len(events) - 1, count).astype(int)
        events = events[np.unique(indices)]
    linear = np.linspace(0, max_epsilon, max(8, count // 2))
    grid = np.unique(np.concatenate([events, linear]))
    if len(grid) > count:
        indices = np.linspace(0, len(grid) - 1, count).astype(int)
        grid = grid[np.unique(indices)]
    return grid


def compute_persistence(
    points: Sequence[Sequence[float]],
    spec: FiltrationSpec | None = None,
) -> PersistenceSummary:
    spec = spec or FiltrationSpec()
    array = _as_points(points)
    if spec.method is FiltrationMethod.ALPHA:
        diagrams, max_epsilon, warnings = _alpha_diagrams(array, spec.max_dim, spec.field)
    else:
        diagrams, max_epsilon, warnings = _rips_diagrams(array, spec.max_dim, spec.field, spec)

    epsilon = (
        float(spec.epsilon) if spec.epsilon is not None else select_epsilon(diagrams, max_epsilon)
    )
    epsilon = min(max(epsilon, 0.0), max_epsilon)
    max_lifetime = max(
        (
            (death - birth) if death is not None else max_epsilon - birth
            for dimension in range(1, len(diagrams))
            for birth, death in diagrams[dimension]
        ),
        default=0.0,
    )
    significance = max(spec.significance_fraction * max_lifetime, 1e-9)
    betti = betti_at(diagrams, epsilon, min_lifetime=significance)
    grid = _curve_grid(diagrams, max_epsilon)
    curve = [betti_at(diagrams, float(value), min_lifetime=significance) for value in grid]

    finite_values = sorted(
        {
            float(value)
            for diagram in diagrams
            for birth, death in diagram
            for value in (birth, death)
            if value is not None and math.isfinite(value)
        }
    )
    if len(finite_values) > 256:
        indices = np.linspace(0, len(finite_values) - 1, 256).astype(int)
        finite_values = [finite_values[i] for i in indices]

    if not any(any(diagram) for diagram in diagrams[1:]):
        warnings.append("No positive-dimensional finite persistence pairs were found.")
    if spec.epsilon is not None and not (0 <= spec.epsilon <= max_epsilon):
        warnings.append(
            f"Requested epsilon={spec.epsilon:g} lies outside [0, {max_epsilon:g}]; it was clipped."
        )

    return PersistenceSummary(
        method=spec.method,
        field=spec.field,
        max_dim=spec.max_dim,
        epsilon=epsilon,
        betti=betti,
        diagrams=[[[birth, death] for birth, death in diagram] for diagram in diagrams],
        finite_values=finite_values,
        epsilon_grid=[float(value) for value in grid],
        betti_curve=curve,
        significance_threshold=significance,
        warnings=warnings,
    )
