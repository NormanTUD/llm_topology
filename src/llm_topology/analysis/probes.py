from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score

from ..model.config import AnalysisSpec, DataBuildSpec
from ..model.data import DatasetBundle, TokenExample, build_dataset
from ..model.train import load_checkpoint, resolve_device
from ..model.transformer import TopologyTransformer
from ..tokenizer import NumericTokenizer


class AnalysisError(RuntimeError):
    pass


CONTINUOUS_TARGETS = [
    "betti.0",
    "betti.1",
    "betti.2",
    "ph.0",
    "ph.1",
    "ph.2",
    "genus",
    "n_components",
]


@dataclass
class ActivationSet:
    activations: list[np.ndarray]
    examples: list[TokenExample]


def _inference_examples(examples: list[TokenExample], limit: int) -> list[TokenExample]:
    selected = [example for example in examples if example.task == "infer"]
    return selected[:limit]


@torch.no_grad()
def collect_activations(
    model: TopologyTransformer,
    examples: list[TokenExample],
    *,
    device: torch.device,
    limit: int,
) -> ActivationSet:
    selected = _inference_examples(examples, limit)
    if not selected:
        raise AnalysisError(
            "Keine Inferenz-Beispiele fuer Probes gefunden. "
            "Erhoehe n_val oder senke task_mix, damit mindestens ein Inferenz-Sample im Split liegt."
        )
    rows: list[list[np.ndarray]] = []
    for example in selected:
        input_ids = torch.tensor([example.input_ids], dtype=torch.long, device=device)
        output = model(input_ids)
        boundary = example.boundary_index
        rows.append(
            [
                hidden[0, boundary].detach().cpu().numpy().astype(np.float64)
                for hidden in output.hidden_states
            ]
        )
    layers = len(rows[0])
    activations = [np.stack([row[layer] for row in rows], axis=0) for layer in range(layers)]
    return ActivationSet(activations=activations, examples=selected)


def _target_matrix(examples: list[TokenExample]) -> dict[str, np.ndarray]:
    return {
        "betti.0": np.asarray([example.betti[0] for example in examples], dtype=float),
        "betti.1": np.asarray([example.betti[1] for example in examples], dtype=float),
        "betti.2": np.asarray([example.betti[2] for example in examples], dtype=float),
        "ph.0": np.asarray([example.ph[0] for example in examples], dtype=float),
        "ph.1": np.asarray([example.ph[1] for example in examples], dtype=float),
        "ph.2": np.asarray([example.ph[2] for example in examples], dtype=float),
        "genus": np.asarray([example.genus for example in examples], dtype=float),
        "n_components": np.asarray([example.n_components for example in examples], dtype=float),
    }


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    if np.allclose(y_true, y_true[0]):
        return float("nan")
    return float(r2_score(y_true, y_pred))


def _probe_layer(
    x_train: np.ndarray,
    x_val: np.ndarray,
    train_examples: list[TokenExample],
    val_examples: list[TokenExample],
    *,
    alpha: float,
    top_neurons: int,
) -> dict[str, Any]:
    train_targets = _target_matrix(train_examples)
    val_targets = _target_matrix(val_examples)
    r2: dict[str, float] = {}
    weights: list[list[float]] = []
    top: dict[str, list[dict[str, float | int]]] = {}
    for target in CONTINUOUS_TARGETS:
        if target not in train_targets or target not in val_targets:
            continue
        estimator = Ridge(alpha=alpha)
        estimator.fit(x_train, train_targets[target])
        prediction = estimator.predict(x_val)
        r2[target] = _safe_r2(val_targets[target], prediction)
        coef = estimator.coef_.astype(float)
        weights.append(coef.tolist())
        order = np.argsort(-np.abs(coef))[:top_neurons]
        top[target] = [{"index": int(index), "weight": float(coef[index])} for index in order]

    structure_accuracy = float("nan")
    labels_train = np.asarray([example.structure_id for example in train_examples])
    labels_val = np.asarray([example.structure_id for example in val_examples])
    if len(np.unique(labels_train)) > 1:
        try:
            classifier = LogisticRegression(max_iter=2000, n_jobs=None)
            classifier.fit(x_train, labels_train)
            structure_accuracy = float(accuracy_score(labels_val, classifier.predict(x_val)))
        except Exception:
            structure_accuracy = float("nan")

    return {
        "r2": r2,
        "structure_accuracy": structure_accuracy,
        "weights": weights,
        "top_neurons": top,
    }


@torch.no_grad()
def _attention_and_ffn(
    model: TopologyTransformer,
    tokenizer: NumericTokenizer,
    examples: list[TokenExample],
    *,
    device: torch.device,
    index: int,
    max_tokens: int = 320,
) -> dict[str, Any]:
    if not examples:
        return {}
    example = examples[index % len(examples)]
    input_ids = torch.tensor([example.input_ids], dtype=torch.long, device=device)
    output = model(input_ids, return_attentions=True, return_ffn=True)
    if not output.attentions:
        return {}
    boundary = example.boundary_index
    attention_rows: list[list[list[float]]] = []
    entropy: list[list[float]] = []
    for layer_attention in output.attentions:
        layer_rows: list[list[float]] = []
        layer_entropy: list[float] = []
        for head in range(layer_attention.shape[1]):
            row = layer_attention[0, head, boundary].detach().cpu().numpy()
            layer_rows.append(row.tolist())
            clipped = np.clip(row, 1e-12, 1.0)
            layer_entropy.append(float(-(clipped * np.log(clipped)).sum()))
        attention_rows.append(layer_rows)
        entropy.append(layer_entropy)

    token_labels = [tokenizer.id_to_token[int(token_id)] for token_id in example.input_ids]
    # Downsample the key axis for the browser if needed.
    if len(token_labels) > max_tokens:
        indices = np.linspace(0, len(token_labels) - 1, max_tokens).astype(int)
        token_labels = [token_labels[int(i)] for i in indices]
        attention_rows = [
            [[row[int(i)] for i in indices] for row in layer] for layer in attention_rows
        ]
        boundary_display = int(np.searchsorted(indices, boundary))
    else:
        boundary_display = boundary

    ffn_layers: list[list[float]] = []
    if output.ffn_activations:
        for layer_ffn in output.ffn_activations:
            ffn_layers.append(layer_ffn[0, boundary].detach().cpu().numpy().tolist())
    return {
        "tokens": token_labels,
        "boundary_index": boundary_display,
        "attention": attention_rows,
        "attention_entropy": entropy,
        "ffn_activation": ffn_layers,
        "target_text": example.target_text,
        "structure_name": example.structure_name,
    }


@torch.no_grad()
def _pca_projection(
    activations: list[np.ndarray],
    examples: list[TokenExample],
    *,
    dim: int,
) -> dict[str, Any]:
    coordinates: list[list[list[float]]] = []
    variances: list[list[float]] = []
    for layer in activations:
        if layer.shape[0] < 2:
            coordinates.append([[0.0] * dim for _ in range(layer.shape[0])])
            variances.append([0.0] * dim)
            continue
        count = min(dim, layer.shape[0], layer.shape[1])
        pca = PCA(n_components=count)
        reduced = pca.fit_transform(layer)
        padded = np.zeros((reduced.shape[0], dim))
        padded[:, :count] = reduced
        coordinates.append(padded.tolist())
        variances.append(pca.explained_variance_ratio_.tolist() + [0.0] * (dim - count))
    return {
        "coordinates": coordinates,
        "explained_variance": variances,
        "structure_names": [example.structure_name for example in examples],
        "structure_ids": [example.structure_id for example in examples],
        "betti": [example.betti for example in examples],
        "genus": [example.genus for example in examples],
    }


@torch.no_grad()
def _neuron_maps(activations: list[np.ndarray]) -> dict[str, Any]:
    mean_abs: list[list[float]] = []
    normalized: list[list[float]] = []
    for layer in activations:
        values = np.abs(layer).mean(axis=0)
        mean_abs.append(values.tolist())
        maximum = float(values.max()) if values.size else 0.0
        normalized.append((values / maximum).tolist() if maximum > 0 else values.tolist())
    return {"mean_abs": mean_abs, "normalized": normalized}


def analyze_model(
    model: TopologyTransformer,
    tokenizer: NumericTokenizer,
    bundle: DatasetBundle,
    spec: AnalysisSpec,
    *,
    device: torch.device | None = None,
) -> dict[str, Any]:
    device = device or torch.device("cpu")
    model.to(device).eval()
    train_set = collect_activations(model, bundle.train, device=device, limit=spec.max_examples)
    val_examples = _inference_examples(bundle.val, max(spec.max_examples // 2, 8))
    if len(val_examples) < 2:
        val_examples.extend(_inference_examples(bundle.test, max(spec.max_examples // 2, 8)))
    if len(val_examples) < 2:
        val_examples.extend(_inference_examples(bundle.train, max(spec.max_examples // 2, 8)))
    val_set = collect_activations(
        model, val_examples, device=device, limit=max(spec.max_examples // 2, 8)
    )
    probe_results = []
    for layer_index, (x_train, x_val) in enumerate(
        zip(train_set.activations, val_set.activations, strict=True)
    ):
        layer_result = _probe_layer(
            x_train,
            x_val,
            train_set.examples,
            val_set.examples,
            alpha=spec.probe_alpha,
            top_neurons=spec.top_neurons,
        )
        layer_result["layer"] = layer_index
        probe_results.append(layer_result)

    attention = _attention_and_ffn(
        model,
        tokenizer,
        val_set.examples,
        device=device,
        index=spec.attention_example,
    )
    projection = _pca_projection(
        val_set.activations,
        val_set.examples,
        dim=spec.pca_dim,
    )
    neuron_maps = _neuron_maps(val_set.activations)
    summary = {
        "best_structure_accuracy": max(
            (
                item["structure_accuracy"]
                for item in probe_results
                if not np.isnan(item["structure_accuracy"])
            ),
            default=float("nan"),
        ),
        "best_betti_r2": max(
            (
                value
                for item in probe_results
                for key, value in item["r2"].items()
                if key.startswith("betti") and not np.isnan(value)
            ),
            default=float("nan"),
        ),
        "layers": len(probe_results),
        "examples_train": len(train_set.examples),
        "examples_val": len(val_set.examples),
        "parameter_count": model.parameter_count(),
    }
    return {
        "summary": summary,
        "probes": probe_results,
        "targets": CONTINUOUS_TARGETS,
        "attention": attention,
        "projection": projection,
        "neuron_maps": neuron_maps,
    }


def analyze_checkpoint(
    checkpoint_path: str | Path,
    spec: AnalysisSpec,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    resolved = resolve_device(device)
    model, tokenizer, checkpoint = load_checkpoint(checkpoint_path, device=str(resolved))
    data_spec = DataBuildSpec.model_validate(checkpoint["data_spec"])
    bundle = build_dataset(tokenizer, data_spec)
    result = analyze_model(model, tokenizer, bundle, spec, device=resolved)
    result["checkpoint"] = str(checkpoint_path)
    result["model_spec"] = checkpoint["model_spec"]
    result["data_spec"] = checkpoint["data_spec"]
    return result
