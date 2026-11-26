from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import DatasetSpec, GeneratedPointCloud, StructureKind


class GeneratorError(ValueError):
    pass


STRUCTURE_IDS: dict[StructureKind, int] = {
    StructureKind.CIRCLE: 0,
    StructureKind.SPHERE: 1,
    StructureKind.TORUS: 2,
    StructureKind.GENUS_SURFACE: 3,
    StructureKind.DISJOINT: 4,
    StructureKind.LINKED_RINGS: 5,
    StructureKind.FIGURE_EIGHT: 6,
    StructureKind.KLEIN_BOTTLE: 7,
    StructureKind.MOBIUS_STRIP: 8,
    StructureKind.ANNULUS: 9,
    StructureKind.NOISE: 10,
    StructureKind.BALL: 11,
}

STRUCTURE_LABELS: dict[StructureKind, str] = {
    StructureKind.CIRCLE: "Circle",
    StructureKind.SPHERE: "Sphere",
    StructureKind.TORUS: "Torus",
    StructureKind.GENUS_SURFACE: "Genus surface",
    StructureKind.DISJOINT: "Disjoint union",
    StructureKind.LINKED_RINGS: "Linked rings",
    StructureKind.FIGURE_EIGHT: "Figure eight",
    StructureKind.KLEIN_BOTTLE: "Klein bottle",
    StructureKind.MOBIUS_STRIP: "Mobius strip",
    StructureKind.ANNULUS: "Annulus",
    StructureKind.NOISE: "Noise cloud",
    StructureKind.BALL: "Solid ball",
}


@dataclass(frozen=True)
class TheoreticalTopology:
    betti_z2: tuple[int, ...]
    betti_q: tuple[int, ...]
    genus: int
    n_components: int


def _normalize_betti(values: list[int] | tuple[int, ...], max_dim: int = 2) -> tuple[int, ...]:
    values = tuple(int(v) for v in values)
    if len(values) < max_dim + 1:
        values = values + (0,) * (max_dim + 1 - len(values))
    return values[: max_dim + 1]


def theoretical_topology(
    kind: StructureKind,
    *,
    genus: int = 2,
    copies: int = 1,
    params: dict[str, float | int | str | bool | None] | None = None,
    max_dim: int = 2,
) -> TheoreticalTopology:
    params = params or {}
    if kind is StructureKind.DISJOINT:
        base_raw = str(params.get("base_structure", StructureKind.TORUS.value))
        try:
            base_kind = StructureKind(base_raw)
        except ValueError as exc:
            raise GeneratorError(f"unknown base_structure: {base_raw!r}") from exc
        base = theoretical_topology(
            base_kind,
            genus=genus,
            copies=1,
            params=params,
            max_dim=max_dim,
        )
        return TheoreticalTopology(
            betti_z2=tuple(v * copies for v in base.betti_z2),
            betti_q=tuple(v * copies for v in base.betti_q),
            genus=base.genus * copies,
            n_components=base.n_components * copies,
        )

    if kind is StructureKind.CIRCLE:
        base = TheoreticalTopology((1, 1, 0), (1, 1, 0), 0, 1)
    elif kind is StructureKind.SPHERE:
        base = TheoreticalTopology((1, 0, 1), (1, 0, 1), 0, 1)
    elif kind is StructureKind.TORUS:
        base = TheoreticalTopology((1, 2, 1), (1, 2, 1), 1, 1)
    elif kind is StructureKind.GENUS_SURFACE:
        base = TheoreticalTopology((1, 2 * genus, 1), (1, 2 * genus, 1), genus, 1)
    elif kind is StructureKind.LINKED_RINGS:
        base = TheoreticalTopology((2, 2, 0), (2, 2, 0), 0, 2)
    elif kind is StructureKind.FIGURE_EIGHT:
        base = TheoreticalTopology((1, 2, 0), (1, 2, 0), 0, 1)
    elif kind is StructureKind.KLEIN_BOTTLE:
        base = TheoreticalTopology((1, 2, 1), (1, 1, 0), 1, 1)
    elif kind is StructureKind.MOBIUS_STRIP or kind is StructureKind.ANNULUS:
        base = TheoreticalTopology((1, 1, 0), (1, 1, 0), 0, 1)
    elif kind in {StructureKind.NOISE, StructureKind.BALL}:
        base = TheoreticalTopology((1, 0, 0), (1, 0, 0), 0, 1)
    else:
        raise GeneratorError(f"unsupported structure: {kind.value!r}")

    return TheoreticalTopology(
        betti_z2=_normalize_betti([v * copies for v in base.betti_z2], max_dim),
        betti_q=_normalize_betti([v * copies for v in base.betti_q], max_dim),
        genus=base.genus * copies,
        n_components=base.n_components * copies,
    )


def _unit_vector(ambient_dim: int, rng: np.random.Generator) -> np.ndarray:
    vector = rng.normal(size=ambient_dim)
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        vector = np.zeros(ambient_dim)
        vector[0] = 1.0
        return vector
    return vector / norm


def _embed_orthonormal(
    points: np.ndarray, ambient_dim: int, rng: np.random.Generator
) -> np.ndarray:
    if points.ndim != 2:
        raise GeneratorError("points must be a two-dimensional array")
    intrinsic_dim = points.shape[1]
    if intrinsic_dim > ambient_dim:
        raise GeneratorError(
            f"structure needs at least {intrinsic_dim} ambient dimensions, got {ambient_dim}"
        )
    if intrinsic_dim == ambient_dim:
        return points
    basis, _ = np.linalg.qr(rng.normal(size=(ambient_dim, intrinsic_dim)))
    return points @ basis.T


def _center(points: np.ndarray) -> np.ndarray:
    return points - points.mean(axis=0, keepdims=True)


def _area_uniform_torus_v(
    n: int, rng: np.random.Generator, major: float, minor: float
) -> np.ndarray:
    accepted: list[float] = []
    max_density = major + minor
    while len(accepted) < n:
        candidate = rng.uniform(-math.pi, math.pi, size=max(16, n - len(accepted)))
        density = major + minor * np.cos(candidate)
        mask = rng.uniform(0, max_density, size=candidate.shape) < density
        accepted.extend(candidate[mask].tolist())
    return np.asarray(accepted[:n])


def _circle(n: int, rng: np.random.Generator, radius: float = 1.0) -> np.ndarray:
    theta = rng.uniform(0, 2 * math.pi, n)
    return np.c_[radius * np.cos(theta), radius * np.sin(theta)]


def _sphere(n: int, rng: np.random.Generator, radius: float = 1.0) -> np.ndarray:
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return direction * radius


def _torus(
    n: int,
    rng: np.random.Generator,
    major: float = 2.0,
    minor: float = 0.7,
) -> np.ndarray:
    u = rng.uniform(0, 2 * math.pi, n)
    v = _area_uniform_torus_v(n, rng, major, minor)
    radial = major + minor * np.cos(v)
    return np.c_[radial * np.cos(u), radial * np.sin(u), minor * np.sin(v)]


def _distance_to_circle(points: np.ndarray, center: np.ndarray, major: float) -> np.ndarray:
    planar = points[:, :2] - center[:2]
    rho = np.linalg.norm(planar, axis=1)
    return np.sqrt((rho - major) ** 2 + points[:, 2] ** 2)


def _genus_surface(
    n: int,
    genus: int,
    rng: np.random.Generator,
    major: float = 1.0,
    minor: float = 0.34,
) -> np.ndarray:
    if genus < 1:
        raise GeneratorError("genus must be at least 1")
    spacing = 1.35 * major
    centers = np.c_[
        (np.arange(genus) - (genus - 1) / 2) * spacing,
        np.zeros(genus),
        np.zeros(genus),
    ]
    accepted: list[np.ndarray] = []
    attempts = 0
    max_attempts = 80
    while sum(len(chunk) for chunk in accepted) < n and attempts < max_attempts:
        attempts += 1
        batch_per_loop = max(32, int(n * 1.6 / genus))
        for loop_index, center in enumerate(centers):
            u = rng.uniform(0, 2 * math.pi, batch_per_loop)
            v = _area_uniform_torus_v(batch_per_loop, rng, major, minor)
            radial = major + minor * np.cos(v)
            points = np.c_[
                center[0] + radial * np.cos(u),
                center[1] + radial * np.sin(u),
                center[2] + minor * np.sin(v),
            ]
            keep = np.ones(points.shape[0], dtype=bool)
            for other_index, other_center in enumerate(centers):
                if other_index == loop_index:
                    continue
                keep &= _distance_to_circle(points, other_center, major) >= minor
            accepted.append(points[keep])
    if not accepted:
        raise GeneratorError("could not sample genus surface")
    points = np.vstack(accepted)
    if len(points) < n:
        raise GeneratorError(f"could not sample enough genus-surface points ({len(points)} < {n})")
    indices = rng.choice(len(points), size=n, replace=False)
    return points[indices]


def _linked_rings(n: int, rng: np.random.Generator, radius: float = 1.0) -> np.ndarray:
    n_first = max(1, n // 2)
    n_second = n - n_first
    first_theta = rng.uniform(0, 2 * math.pi, n_first)
    second_theta = rng.uniform(0, 2 * math.pi, n_second)
    first = np.c_[
        radius * np.cos(first_theta),
        radius * np.sin(first_theta),
        np.zeros(n_first),
    ]
    second = np.c_[
        np.zeros(n_second),
        radius + radius * np.cos(second_theta),
        radius * np.sin(second_theta),
    ]
    return np.vstack([first, second])


def _figure_eight(n: int, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    theta = rng.uniform(0, 2 * math.pi, n)
    return np.c_[
        scale * np.sin(theta),
        scale * 0.7 * np.sin(theta) * np.cos(theta),
    ]


def _klein_bottle(
    n: int,
    rng: np.random.Generator,
    major: float = 1.4,
    minor: float = 0.55,
) -> np.ndarray:
    u = rng.uniform(0, 2 * math.pi, n)
    v = rng.uniform(0, 2 * math.pi, n)
    radial = major + minor * np.cos(v)
    return np.c_[
        radial * np.cos(u),
        radial * np.sin(u),
        minor * np.sin(v) * np.cos(u / 2),
        minor * np.sin(v) * np.sin(u / 2),
    ]


def _mobius_strip(
    n: int,
    rng: np.random.Generator,
    radius: float = 1.6,
    half_width: float = 0.55,
) -> np.ndarray:
    u = rng.uniform(0, 2 * math.pi, n)
    s = rng.uniform(-half_width, half_width, n)
    radial = radius + s * np.cos(u / 2)
    return np.c_[
        radial * np.cos(u),
        radial * np.sin(u),
        s * np.sin(u / 2),
    ]


def _annulus(
    n: int,
    rng: np.random.Generator,
    inner: float = 0.55,
    outer: float = 1.25,
) -> np.ndarray:
    radius = np.sqrt(rng.uniform(inner**2, outer**2, n))
    theta = rng.uniform(0, 2 * math.pi, n)
    return np.c_[radius * np.cos(theta), radius * np.sin(theta)]


def _noise(n: int, ambient_dim: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1, 1, size=(n, ambient_dim))


def _ball(n: int, ambient_dim: int, rng: np.random.Generator) -> np.ndarray:
    direction = rng.normal(size=(n, ambient_dim))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    radius = rng.uniform(0, 1, n) ** (1 / ambient_dim)
    return direction * radius[:, None]


def _natural_dimension(kind: StructureKind) -> int:
    if kind in {StructureKind.CIRCLE, StructureKind.FIGURE_EIGHT, StructureKind.ANNULUS}:
        return 2
    if kind in {
        StructureKind.SPHERE,
        StructureKind.TORUS,
        StructureKind.GENUS_SURFACE,
        StructureKind.LINKED_RINGS,
        StructureKind.MOBIUS_STRIP,
    }:
        return 3
    if kind is StructureKind.KLEIN_BOTTLE:
        return 4
    if kind in {StructureKind.NOISE, StructureKind.BALL, StructureKind.DISJOINT}:
        return -1
    return 3


def _single_cloud(
    spec: DatasetSpec,
    kind: StructureKind,
    n_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    params = spec.params
    if kind is StructureKind.CIRCLE:
        intrinsic = _circle(n_points, rng, float(params.get("radius", 1.0)))
    elif kind is StructureKind.SPHERE:
        intrinsic = _sphere(n_points, rng, float(params.get("radius", 1.0)))
    elif kind is StructureKind.TORUS:
        intrinsic = _torus(
            n_points,
            rng,
            float(params.get("major_radius", 2.0)),
            float(params.get("minor_radius", 0.7)),
        )
    elif kind is StructureKind.GENUS_SURFACE:
        intrinsic = _genus_surface(
            n_points,
            spec.genus,
            rng,
            float(params.get("major_radius", 1.0)),
            float(params.get("minor_radius", 0.34)),
        )
    elif kind is StructureKind.LINKED_RINGS:
        intrinsic = _linked_rings(n_points, rng, float(params.get("radius", 1.0)))
    elif kind is StructureKind.FIGURE_EIGHT:
        intrinsic = _figure_eight(n_points, rng, float(params.get("scale", 1.0)))
    elif kind is StructureKind.KLEIN_BOTTLE:
        intrinsic = _klein_bottle(
            n_points,
            rng,
            float(params.get("major_radius", 1.4)),
            float(params.get("minor_radius", 0.55)),
        )
    elif kind is StructureKind.MOBIUS_STRIP:
        intrinsic = _mobius_strip(
            n_points,
            rng,
            float(params.get("radius", 1.6)),
            float(params.get("half_width", 0.55)),
        )
    elif kind is StructureKind.ANNULUS:
        intrinsic = _annulus(
            n_points,
            rng,
            float(params.get("inner_radius", 0.55)),
            float(params.get("outer_radius", 1.25)),
        )
    elif kind is StructureKind.NOISE:
        intrinsic = _noise(n_points, spec.ambient_dim, rng)
    elif kind is StructureKind.BALL:
        intrinsic = _ball(n_points, spec.ambient_dim, rng)
    else:
        raise GeneratorError(f"cannot generate a single cloud for {kind.value!r}")

    if kind in {StructureKind.NOISE, StructureKind.BALL}:
        return _center(intrinsic)

    natural_dim = _natural_dimension(kind)
    if natural_dim > spec.ambient_dim:
        raise GeneratorError(
            f"{STRUCTURE_LABELS[kind]} needs at least {natural_dim} ambient dimensions, "
            f"got {spec.ambient_dim}"
        )
    intrinsic = _center(intrinsic)
    return _embed_orthonormal(intrinsic, spec.ambient_dim, rng)


def _separate_chunks(
    chunks: list[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    if len(chunks) == 1:
        return chunks[0]
    extent = max(
        float(np.max(np.linalg.norm(chunk - chunk.mean(axis=0), axis=1))) for chunk in chunks
    )
    separation = max(2.5 * extent, 0.5)
    direction = _unit_vector(chunks[0].shape[1], rng)
    separated = []
    for index, chunk in enumerate(chunks):
        shift = direction * separation * (index - (len(chunks) - 1) / 2)
        separated.append(chunk + shift)
    return np.vstack(separated)


def _apply_transform(points: np.ndarray, spec: DatasetSpec) -> np.ndarray:
    transform = spec.transform
    rng = np.random.default_rng(transform.seed)
    out = points.copy()

    if any(abs(angle) > 1e-12 for angle in transform.rotation_deg):
        if out.shape[1] < 2:
            raise GeneratorError("rotation requires at least 2 ambient dimensions")
        ax, ay, az = np.deg2rad(transform.rotation_deg)
        rx = np.array(
            [[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]]
        )
        ry = np.array(
            [[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]]
        )
        rz = np.array(
            [[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]]
        )
        rotation = rz[:2, :2] if out.shape[1] == 2 else rz @ ry @ rx
        dim = min(3, out.shape[1])
        out[:, :dim] = out[:, :dim] @ rotation[:dim, :dim].T

    stretch = np.asarray(transform.stretch, dtype=float)
    dim = min(3, out.shape[1])
    out[:, :dim] *= stretch[:dim]

    if out.shape[1] >= 2:
        sx, sy, sz = transform.shear
        x = out[:, 0].copy()
        y = out[:, 1].copy()
        if out.shape[1] >= 3:
            z = out[:, 2].copy()
            out[:, 0] = x + sx * y + sy * z
            out[:, 1] = y + sz * z
        else:
            out[:, 0] = x + sx * y

    if abs(transform.twist) > 1e-12 and out.shape[1] >= 3:
        z = out[:, 2]
        scale = np.std(z)
        if scale > 1e-12:
            angle = transform.twist * z / scale
            x = out[:, 0].copy()
            y = out[:, 1].copy()
            out[:, 0] = x * np.cos(angle) - y * np.sin(angle)
            out[:, 1] = x * np.sin(angle) + y * np.cos(angle)

    if abs(transform.bend) > 1e-12 and out.shape[1] >= 2:
        x = out[:, 0]
        scale = float(np.std(x))
        if scale > 1e-12:
            out[:, 1] = out[:, 1] + transform.bend * scale * np.sin(x / scale)

    if abs(transform.warp) > 1e-12 and out.shape[1] >= 2:
        scale = max(float(np.std(out)), 1e-12)
        amplitude = 0.18 * transform.warp * scale
        phase = transform.seed * 0.017
        x = out[:, 0].copy()
        y = out[:, 1].copy()
        out[:, 0] = x + amplitude * np.sin(2 * math.pi * y / scale + phase)
        out[:, 1] = y + amplitude * np.sin(2 * math.pi * x / scale + phase)
        if out.shape[1] >= 3:
            z = out[:, 2].copy()
            out[:, 2] = z + amplitude * np.sin(2 * math.pi * x / scale + phase)

    translation = np.asarray(transform.translation, dtype=float)
    dim = min(3, out.shape[1])
    out[:, :dim] += translation[:dim]

    if transform.noise > 0:
        sigma = transform.noise * max(float(np.std(out)), 1e-9)
        out = out + rng.normal(scale=sigma, size=out.shape)

    if transform.permute:
        out = out[rng.permutation(out.shape[0])]

    return out


def generate_point_cloud(spec: DatasetSpec) -> GeneratedPointCloud:
    rng = np.random.default_rng(spec.seed)
    if spec.n_points < spec.copies:
        raise GeneratorError("n_points must be at least as large as copies")

    base_kind = spec.structure
    if spec.structure is StructureKind.DISJOINT:
        raw_base = str(spec.params.get("base_structure", StructureKind.TORUS.value))
        try:
            base_kind = StructureKind(raw_base)
        except ValueError as exc:
            raise GeneratorError(f"unknown base_structure: {raw_base!r}") from exc
        if base_kind is StructureKind.DISJOINT:
            raise GeneratorError("disjoint base_structure cannot itself be disjoint")

    topology = theoretical_topology(
        base_kind if spec.structure is StructureKind.DISJOINT else spec.structure,
        genus=spec.genus,
        copies=spec.copies,
        params=spec.params,
        max_dim=spec.filtration.max_dim,
    )

    if spec.copies == 1:
        points = _single_cloud(spec, base_kind, spec.n_points, rng)
    else:
        counts = np.full(spec.copies, spec.n_points // spec.copies, dtype=int)
        counts[: spec.n_points % spec.copies] += 1
        chunks = [_single_cloud(spec, base_kind, int(count), rng) for count in counts]
        points = _separate_chunks(chunks, rng)

    points = _center(points)
    points = _apply_transform(points, spec)
    if not np.all(np.isfinite(points)):
        raise GeneratorError("generated point cloud contains non-finite coordinates")

    structure_name = STRUCTURE_LABELS[
        base_kind if spec.structure is StructureKind.DISJOINT else spec.structure
    ]
    if spec.copies > 1:
        structure_name = f"{spec.copies} x {structure_name}"
    if spec.structure is StructureKind.DISJOINT:
        structure_name = f"{spec.copies} x {STRUCTURE_LABELS[base_kind]}"

    metadata: dict[str, float | int | str | bool | None] = {
        "base_structure": base_kind.value,
        "copies": spec.copies,
        "seed": spec.seed,
        "transform_seed": spec.transform.seed,
        "nonorientable": base_kind in {StructureKind.KLEIN_BOTTLE, StructureKind.MOBIUS_STRIP},
    }
    if spec.structure is StructureKind.GENUS_SURFACE:
        metadata["genus"] = spec.genus

    return GeneratedPointCloud(
        points=points.tolist(),
        structure=spec.structure,
        structure_id=STRUCTURE_IDS[spec.structure],
        structure_name=structure_name,
        theoretical_betti=list(topology.betti_z2),
        theoretical_betti_q=list(topology.betti_q),
        genus=topology.genus,
        n_components=topology.n_components,
        n_points=len(points),
        ambient_dim=spec.ambient_dim,
        metadata=metadata,
    )
