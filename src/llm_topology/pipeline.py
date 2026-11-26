from __future__ import annotations

import numpy as np

from .config import DatasetResponse, DatasetSpec, TransformSpec
from .tokenizer import NumericTokenizer
from .topology.filtration import FiltrationError, compute_persistence
from .topology.generators import GeneratorError, generate_point_cloud
from .topology.labels import build_dataset_response


class PipelineError(ValueError):
    pass


_TOKENIZER = NumericTokenizer()


def generate_dataset(spec: DatasetSpec) -> DatasetResponse:
    if spec.n_points > 20_000:
        raise PipelineError("n_points above 20000 is not supported")
    schema = spec.target_schema
    try:
        sample = generate_point_cloud(spec)
    except GeneratorError as exc:
        raise PipelineError(str(exc)) from exc

    try:
        persistence = compute_persistence(sample.points, spec.filtration)
    except FiltrationError as exc:
        raise PipelineError(str(exc)) from exc

    warnings: list[str] = []
    notes: list[str] = []
    warnings.extend(persistence.warnings)
    if spec.n_points > 2000:
        warnings.append(
            "Large point clouds can make spatial plots, Alpha complexes, and Rips filtrations slow."
        )
    if spec.ambient_dim > 12:
        warnings.append("Ambient dimensions above 12 are mostly useful for stress tests.")
    if spec.filtration.max_dim > 3:
        warnings.append("Homology dimensions above 3 are computationally expensive.")
    if sample.theoretical_betti != persistence.betti:
        notes.append(
            "Generator and PH labels differ because the finite sample and filtration "
            "see a different scale. Compare both; this is part of the experiment."
        )
    if spec.structure.value in {"klein_bottle", "mobius_strip"}:
        warnings.append(
            "Non-orientable structures use Z2 as the default reference field; "
            "the Z3 comparison is a Q-like proxy, not an exact Q computation."
        )

    try:
        groups, target_text, sequence = build_dataset_response(
            sample,
            persistence,
            schema,
            spec,
            _TOKENIZER,
        )
    except (ValueError, KeyError) as exc:
        raise PipelineError(str(exc)) from exc

    projection, projection_method = project_for_display(sample.points)
    reference_projection, reference_name, reference_betti = _reference_cloud(spec, sample)
    guard = list(sample.theoretical_betti[: len(persistence.betti)]) == list(persistence.betti)

    return DatasetResponse(
        sample=sample,
        persistence=persistence,
        target_groups=groups,
        target_text=target_text,
        sequence=sequence,
        target_schema=schema,
        projection=projection,
        projection_method=projection_method,
        reference_projection=reference_projection,
        reference_structure_name=reference_name,
        reference_betti=reference_betti,
        topology_guard=guard,
        topology_guard_label="match" if guard else "differs",
        warnings=warnings,
        notes=notes,
    )


def _reference_cloud(
    spec: DatasetSpec,
    sample,
) -> tuple[list[list[float]], str | None, list[int]]:
    """Canonical comparison cloud for the UI only; never fed to the model."""
    try:
        reference_spec = spec.model_copy(
            update={
                "seed": 0,
                "transform": TransformSpec(seed=0),
            }
        )
        reference = generate_point_cloud(reference_spec)
        projection, _ = project_for_display(reference.points)
        return projection, reference.structure_name, list(reference.theoretical_betti)
    except Exception:
        return [], None, list(sample.theoretical_betti)


def project_for_display(
    points: list[list[float]],
) -> tuple[list[list[float]], str]:
    array = np.asarray(points, dtype=float)
    if array.shape[1] > 3:
        centered = array - array.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ vt[:3].T
        return projected.tolist(), "pca-3d"
    if array.shape[1] < 3:
        projected = np.zeros((array.shape[0], 3))
        projected[:, : array.shape[1]] = array
        return projected.tolist(), "padded-3d"
    return array.tolist(), "native-3d"
