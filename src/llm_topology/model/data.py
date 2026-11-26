from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import DatasetSpec, StructureKind, TransformSpec
from ..pipeline import PipelineError, generate_dataset
from ..tokenizer import DEFAULT_BOUNDARY, NumericTokenizer, serialize_points
from .config import DataBuildSpec


class DataBuildError(ValueError):
    pass


@dataclass
class TokenExample:
    task: str
    input_ids: list[int]
    loss_mask: list[bool]
    boundary_index: int
    prefix_ids: list[int]
    target_ids: list[int]
    structure_id: int
    structure_name: str
    genus: float
    n_components: float
    betti: list[float]
    ph: list[float]
    points: list[list[float]]
    target_text: str
    target_groups: list[list[float]]
    seq_len: int

    def aux_targets(self, max_dim: int) -> dict[str, float | int | list[float]]:
        return {
            "structure_id": self.structure_id,
            "genus": self.genus,
            "n_components": self.n_components,
            "betti": _pad(self.betti, max_dim + 1),
            "ph": _pad(self.ph, max_dim + 1),
        }


@dataclass
class DatasetBundle:
    train: list[TokenExample]
    val: list[TokenExample]
    tokenizer: NumericTokenizer
    data_spec: DataBuildSpec
    max_dim: int
    structure_names: list[str]
    skipped: int = 0


def _pad(values: list[float], size: int) -> list[float]:
    values = list(values[:size])
    return values + [0.0] * (size - len(values))


def _random_transform(spec: DataBuildSpec, seed: int) -> TransformSpec:
    if not spec.augment:
        return TransformSpec(seed=seed)
    rng = np.random.default_rng(seed)
    strength = spec.augment_strength
    return TransformSpec(
        rotation_deg=tuple(rng.uniform(-180, 180, size=3).tolist()),
        stretch=tuple(rng.uniform(1 - 0.22 * strength, 1 + 0.22 * strength, size=3).tolist()),
        twist=float(rng.uniform(-0.8, 0.8) * strength),
        shear=tuple(rng.uniform(-0.45, 0.45, size=3).tolist() * strength),
        bend=float(rng.uniform(-0.55, 0.55) * strength),
        warp=float(rng.uniform(-0.6, 0.6) * strength),
        noise=float(rng.uniform(0, 0.015) * strength),
        permute=True,
        seed=seed + 13,
    )


def _make_spec(
    data_spec: DataBuildSpec,
    structure: StructureKind,
    index: int,
    *,
    augment: bool,
) -> DatasetSpec:
    seed = data_spec.seed + index + (1_000_000 if not augment else 0)
    transform_spec = _random_transform(data_spec, seed)
    if not augment:
        transform_spec = TransformSpec(seed=seed)
    return DatasetSpec(
        structure=structure,
        n_points=data_spec.n_points,
        ambient_dim=data_spec.ambient_dim,
        copies=data_spec.copies,
        genus=data_spec.genus,
        seed=seed,
        params={},
        transform=transform_spec,
        filtration=data_spec.filtration,
        target_schema=data_spec.target_schema,
    )


def _make_example(
    tokenizer: NumericTokenizer,
    response,
    task: str,
    data_spec: DataBuildSpec,
) -> TokenExample | None:
    sample = response.sample
    point_text = serialize_points(sample.points, precision=data_spec.precision)
    point_ids = tokenizer.encode(point_text)
    target_text = response.target_text
    target_ids = tokenizer.encode(target_text)
    eot_id = tokenizer.token_to_id[DEFAULT_BOUNDARY]
    eos_id = tokenizer.eos_id

    if task == "infer":
        prefix_ids = [tokenizer.token_to_id["<infer>"], *point_ids, eot_id]
        answer_ids = [*target_ids, eos_id]
    else:
        prefix_ids = [tokenizer.token_to_id["<generate>"], *target_ids, eot_id]
        answer_ids = [*point_ids, eos_id]

    input_ids = [*prefix_ids, *answer_ids]
    if len(input_ids) > data_spec.max_seq_len:
        return None
    boundary_index = len(prefix_ids) - 1
    loss_mask = [False] * len(input_ids)
    for index in range(boundary_index, len(input_ids) - 1):
        loss_mask[index] = True

    max_dim = data_spec.filtration.max_dim
    return TokenExample(
        task=task,
        input_ids=input_ids,
        loss_mask=loss_mask,
        boundary_index=boundary_index,
        prefix_ids=prefix_ids,
        target_ids=target_ids,
        structure_id=sample.structure_id,
        structure_name=sample.structure_name,
        genus=float(sample.genus),
        n_components=float(sample.n_components),
        betti=_pad([float(value) for value in sample.theoretical_betti], max_dim + 1),
        ph=_pad([float(value) for value in response.persistence.betti], max_dim + 1),
        points=sample.points,
        target_text=target_text,
        target_groups=response.target_groups,
        seq_len=len(input_ids),
    )


def _build_split(
    tokenizer: NumericTokenizer,
    data_spec: DataBuildSpec,
    count: int,
    *,
    augment: bool,
    start_index: int,
) -> tuple[list[TokenExample], int]:
    examples: list[TokenExample] = []
    skipped = 0
    attempts = 0
    max_attempts = count * 3 + 8
    while len(examples) < count and attempts < max_attempts:
        index = start_index + attempts
        structure = data_spec.structures[index % len(data_spec.structures)]
        attempts += 1
        spec = _make_spec(data_spec, structure, index, augment=augment)
        try:
            response = generate_dataset(spec)
        except PipelineError:
            skipped += 1
            continue
        task = "generate" if attempts <= round(count * data_spec.task_mix) else "infer"
        example = _make_example(tokenizer, response, task, data_spec)
        if example is None:
            skipped += 1
            continue
        examples.append(example)
    if not examples:
        raise DataBuildError(
            "No training examples fit into max_seq_len. Reduce n_points, precision, "
            "or increase max_seq_len."
        )
    return examples, skipped


def build_dataset(tokenizer: NumericTokenizer, data_spec: DataBuildSpec) -> DatasetBundle:
    train, skipped_train = _build_split(
        tokenizer,
        data_spec,
        data_spec.n_train,
        augment=data_spec.augment,
        start_index=0,
    )
    val, skipped_val = _build_split(
        tokenizer,
        data_spec,
        data_spec.n_val,
        augment=False,
        start_index=100_000,
    )
    names = [structure.value for structure in data_spec.structures]
    return DatasetBundle(
        train=train,
        val=val,
        tokenizer=tokenizer,
        data_spec=data_spec,
        max_dim=data_spec.filtration.max_dim,
        structure_names=names,
        skipped=skipped_train + skipped_val,
    )


class TokenDataset(Dataset):
    def __init__(self, examples: list[TokenExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TokenExample:
        return self.examples[index]


def collate_examples(
    batch: list[TokenExample],
    pad_id: int,
) -> dict[str, torch.Tensor | list[list[int]] | list[int]]:
    max_len = max(len(example.input_ids) for example in batch)
    max_dim = max(len(example.betti) for example in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    loss_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    aux_structure = torch.zeros(len(batch), dtype=torch.long)
    aux_genus = torch.zeros(len(batch), dtype=torch.float)
    aux_components = torch.zeros(len(batch), dtype=torch.float)
    aux_betti = torch.zeros(len(batch), max_dim, dtype=torch.float)
    aux_ph = torch.zeros(len(batch), max_dim, dtype=torch.float)
    boundary = torch.zeros(len(batch), dtype=torch.long)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    for row, example in enumerate(batch):
        length = len(example.input_ids)
        input_ids[row, :length] = torch.tensor(example.input_ids, dtype=torch.long)
        loss_mask[row, :length] = torch.tensor(example.loss_mask, dtype=torch.bool)
        boundary[row] = example.boundary_index
        lengths[row] = length
        aux_structure[row] = example.structure_id
        aux_genus[row] = example.genus
        aux_components[row] = example.n_components
        aux_betti[row, : len(example.betti)] = torch.tensor(example.betti)
        aux_ph[row, : len(example.ph)] = torch.tensor(example.ph)
    return {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "boundary_index": boundary,
        "lengths": lengths,
        "aux_structure": aux_structure,
        "aux_genus": aux_genus,
        "aux_components": aux_components,
        "aux_betti": aux_betti,
        "aux_ph": aux_ph,
        "prefix_ids": [example.prefix_ids for example in batch],
        "target_ids": [example.target_ids for example in batch],
        "tasks": [example.task for example in batch],
    }
