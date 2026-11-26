from __future__ import annotations

from .config import DatasetResponse, DatasetSpec
from .tokenizer import NumericTokenizer
from .topology.filtration import FiltrationError, compute_persistence
from .topology.generators import GeneratorError, generate_point_cloud
from .topology.labels import build_dataset_response, schema_for_max_dim


class PipelineError(ValueError):
    pass


_TOKENIZER = NumericTokenizer()


def generate_dataset(spec: DatasetSpec) -> DatasetResponse:
    if spec.n_points > 20_000:
        raise PipelineError("n_points above 20000 is not supported")
    if spec.n_points > 1024 or spec.ambient_dim > 6:
        pass

    schema = schema_for_max_dim(spec.target_schema, spec.filtration.max_dim)
    try:
        sample = generate_point_cloud(spec)
    except GeneratorError as exc:
        raise PipelineError(str(exc)) from exc

    try:
        persistence = compute_persistence(sample.points, spec.filtration)
    except FiltrationError as exc:
        raise PipelineError(str(exc)) from exc

    warnings: list[str] = []
    warnings.extend(persistence.warnings)
    if spec.n_points > 512:
        warnings.append("Large point clouds can make spatial plots and Rips filtration slow.")
    if spec.ambient_dim > 6:
        warnings.append("Ambient dimensions above 6 are mostly useful for stress tests.")
    if sample.theoretical_betti != persistence.betti:
        warnings.append(
            "The computed PH label differs from the theoretical generator label. "
            "This is expected for finite noisy samples and is part of the experiment."
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

    return DatasetResponse(
        sample=sample,
        persistence=persistence,
        target_groups=groups,
        target_text=target_text,
        sequence=sequence,
        target_schema=schema,
        warnings=warnings,
    )
