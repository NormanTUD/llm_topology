from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

from .config import DatasetSpec
from .model.config import DataBuildSpec, TrainingSpec


def _read_int(path: str) -> int | None:
    try:
        return int(Path(path).read_text().strip())
    except Exception:
        return None


def available_memory_bytes() -> int:
    candidates: list[int] = []
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                candidates.append(int(line.split()[1]) * 1024)
                break
    with contextlib.suppress(Exception):
        candidates.append(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    cgroup_v2 = _read_int("/sys/fs/cgroup/memory.max")
    cgroup_v2_current = _read_int("/sys/fs/cgroup/memory.current")
    if cgroup_v2 and cgroup_v2 > 0:
        candidates.append(max(0, cgroup_v2 - (cgroup_v2_current or 0)))
    cgroup_v1 = _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    cgroup_v1_current = _read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    if cgroup_v1 and cgroup_v1 < 10**15:
        candidates.append(max(0, cgroup_v1 - (cgroup_v1_current or 0)))
    return min(candidates) if candidates else 2 * 1024**3


def _level(ratio: float) -> str:
    if ratio >= 0.85:
        return "block"
    if ratio >= 0.55:
        return "warn"
    return "ok"


def _points_at_max(spec: DatasetSpec) -> int:
    copies = spec.copies
    if spec.copies_min is not None or spec.copies_max is not None:
        copies = spec.copies_max or spec.copies_min or copies
    if spec.points_per_component is not None:
        return spec.points_per_component * copies
    return spec.n_points


def estimate_generation(spec: DatasetSpec, *, count: int) -> dict[str, Any]:
    points = _points_at_max(spec)
    total_points = points * count
    tokens = total_points * spec.ambient_dim * (spec.precision + 3) + 128 * count
    available = available_memory_bytes()
    bytes_needed = total_points * 64 + tokens * 8
    if spec.filtration.method.value == "rips":
        bytes_needed += total_points * total_points * 16
    ratio = bytes_needed / max(available, 1)
    warnings: list[str] = []
    if total_points > 5000:
        warnings.append(
            f"{total_points} Punkte in einer Batch sind sehr viel fuer interaktive Plots."
        )
    if spec.filtration.method.value == "rips" and total_points > 700:
        warnings.append(
            "Vietoris-Rips skaliert schlecht: mehr als ~700 Punkte koennen sehr lange dauern."
        )
    if spec.precision > 6:
        warnings.append("Sehr hohe String-Praezision erzeugt lange Tokenketten.")
    if tokens > 12000:
        warnings.append(f"Geschaetzte Tokenzahl ~{tokens:,}; das ist fuer kleine Kontexte zu viel.")
    if ratio >= 0.55:
        warnings.append(
            f"Geschaetzter Speicherbedarf ~{bytes_needed / 1024**2:.0f} MiB "
            f"bei ~{available / 1024**2:.0f} MiB verfuegbar."
        )
    level = _level(ratio)
    if spec.filtration.method.value == "rips" and total_points > 1200:
        level = "block"
    return {
        "level": level,
        "warnings": warnings,
        "estimates": {
            "points_per_cloud": points,
            "total_points": total_points,
            "tokens": tokens,
            "bytes": int(bytes_needed),
            "available_bytes": available,
        },
    }


def estimate_dataset(data_spec: DataBuildSpec) -> dict[str, Any]:
    examples = data_spec.n_train + data_spec.n_val + data_spec.n_test
    points = data_spec.n_points
    if data_spec.points_per_component is not None:
        copies = data_spec.copies_max or data_spec.copies_min or data_spec.copies
        points = data_spec.points_per_component * copies
    tokens_per_example = points * data_spec.ambient_dim * (data_spec.precision + 3) + 96
    total_tokens = examples * tokens_per_example
    bytes_needed = total_tokens * 4 + examples * 4096
    available = available_memory_bytes()
    ratio = bytes_needed / max(available, 1)
    warnings: list[str] = []
    if tokens_per_example > data_spec.max_seq_len:
        warnings.append(
            f"Eine Sequenz braucht ~{tokens_per_example} Tokens, der Kontext ist "
            f"{data_spec.max_seq_len}. Erhoehe den Kontext oder reduziere Punkte/Precision."
        )
    if examples > 5000:
        warnings.append(f"{examples} Beispiele werden sehr gross und langsam zu bauen.")
    if ratio >= 0.55:
        warnings.append(
            f"Geschaetzter Dataset-Speicher ~{bytes_needed / 1024**2:.0f} MiB "
            f"bei ~{available / 1024**2:.0f} MiB verfuegbar."
        )
    return {
        "level": _level(ratio) if tokens_per_example <= data_spec.max_seq_len else "block",
        "warnings": warnings,
        "estimates": {
            "examples": examples,
            "points_per_example": points,
            "tokens_per_example": tokens_per_example,
            "total_tokens": total_tokens,
            "bytes": int(bytes_needed),
            "available_bytes": available,
        },
    }


def device_memory(device: str) -> tuple[int, str]:
    if device in {"auto", "cuda"}:
        try:
            import torch

            if torch.cuda.is_available():
                free, _total = torch.cuda.mem_get_info()
                name = torch.cuda.get_device_name(0)
                return int(free), f"GPU {name}"
        except Exception:
            pass
    return available_memory_bytes(), "CPU RAM"


def estimate_training(
    spec: TrainingSpec,
    *,
    dataset_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model = spec.model
    seq_len = model.max_seq_len
    if dataset_meta:
        seq_len = int(dataset_meta.get("max_seq_len", seq_len))
    seq_len = min(seq_len, model.max_seq_len)
    batch = spec.training.batch_size
    vocab = 64
    attention = batch * model.n_heads * seq_len * seq_len
    activation = batch * seq_len * model.n_layers * (model.d_model * 4 + model.d_ff * 2)
    logits = batch * seq_len * vocab
    params = model.n_layers * (12 * model.d_model**2 + 2 * model.d_model * model.d_ff)
    params += vocab * model.d_model * 2
    bytes_needed = (attention + activation + logits) * 4 + params * 16
    available, memory_source = device_memory(spec.training.device)
    ratio = bytes_needed / max(available, 1)
    warnings: list[str] = []
    cuda_unavailable = False
    if spec.training.device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                cuda_unavailable = True
                warnings.append("CUDA ist konfiguriert, aber keine CUDA-GPU/Torch-CUDA verfuegbar.")
        except Exception:
            cuda_unavailable = True
            warnings.append("CUDA ist konfiguriert, aber torch konnte nicht geprueft werden.")
    if dataset_meta and dataset_meta.get("max_seq_len", 0) > model.max_seq_len:
        warnings.append(
            f"Dataset-Kontext {dataset_meta['max_seq_len']} > Modell-Kontext {model.max_seq_len}."
        )
    if attention > 1_000_000_000:
        warnings.append(
            f"Attention-Tensor ~{attention / 1e9:.2f} Mrd. Werte; das ist sehr speicherintensiv."
        )
    if params > 100_000_000:
        warnings.append(f"Modell hat ~{params / 1e6:.0f} M Parameter.")
    if ratio >= 0.55:
        warnings.append(
            f"Geschaetzter Trainingsbedarf ~{bytes_needed / 1024**2:.0f} MiB "
            f"bei ~{available / 1024**2:.0f} MiB verfuegbar ({memory_source})."
        )
    level = _level(ratio)
    if cuda_unavailable:
        level = "block"
    if attention > 2_000_000_000:
        level = "block"
    if dataset_meta and dataset_meta.get("max_seq_len", 0) > model.max_seq_len:
        level = "block"
    return {
        "level": level,
        "warnings": warnings,
        "estimates": {
            "seq_len": seq_len,
            "batch_size": batch,
            "attention_elements": attention,
            "activation_elements": activation,
            "parameters": params,
            "bytes": int(bytes_needed),
            "available_bytes": available,
            "memory_source": memory_source,
        },
    }
