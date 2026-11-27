from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .model.config import DataBuildSpec
from .model.data import DatasetBundle, TokenExample, build_dataset
from .pipeline import project_for_display
from .tokenizer import NumericTokenizer


class DatasetStoreError(ValueError):
    pass


class DatasetStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or (Path.cwd() / "datasets")

    def _next_id(self) -> str:
        highest = 0
        if self.root.exists():
            for path in self.root.glob("ds-*"):
                suffix = path.name.removeprefix("ds-")
                if suffix.isdigit():
                    highest = max(highest, int(suffix))
        return f"ds-{highest + 1:04d}"

    def save(self, name: str, data_spec: DataBuildSpec) -> dict[str, Any]:
        tokenizer = NumericTokenizer()
        bundle = build_dataset(tokenizer, data_spec)
        dataset_id = self._next_id()
        directory = self.root / dataset_id
        directory.mkdir(parents=True, exist_ok=True)
        all_examples = [*bundle.train, *bundle.val, *bundle.test]
        token_lengths = [example.seq_len for example in all_examples]
        point_counts = [len(example.points) for example in all_examples]
        train_tokens = [example.seq_len for example in bundle.train]
        meta = {
            "id": dataset_id,
            "name": name.strip() or dataset_id,
            "created_at": time.time(),
            "train_count": len(bundle.train),
            "val_count": len(bundle.val),
            "test_count": len(bundle.test),
            "skipped": bundle.skipped,
            "max_dim": bundle.max_dim,
            "max_seq_len": data_spec.max_seq_len,
            "max_example_tokens": max(token_lengths, default=0),
            "avg_example_tokens": (sum(token_lengths) / len(token_lengths)) if token_lengths else 0,
            "max_train_tokens": max(train_tokens, default=0),
            "max_points": max(point_counts, default=0),
            "avg_points": (sum(point_counts) / len(point_counts)) if point_counts else 0,
            "precision": data_spec.precision,
            "structures": [structure.value for structure in data_spec.structures],
            "target_schema": data_spec.target_schema.model_dump(mode="json"),
            "data_spec": data_spec.model_dump(mode="json"),
            "path": str(directory),
        }
        data = {
            "train": [asdict(example) for example in bundle.train],
            "val": [asdict(example) for example in bundle.val],
            "test": [asdict(example) for example in bundle.test],
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (directory / "data.json").write_text(json.dumps(data), encoding="utf-8")
        print(
            f"[datasets] saved {dataset_id} name={meta['name']!r} "
            f"train={meta['train_count']} val={meta['val_count']} test={meta['test_count']}",
            flush=True,
        )
        return meta

    def list(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        items: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("ds-*/meta.json")):
            try:
                items.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return items

    def get_meta(self, dataset_id: str) -> dict[str, Any]:
        path = self.root / dataset_id / "meta.json"
        if not path.exists():
            raise DatasetStoreError(f"unknown dataset: {dataset_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def preview(self, dataset_id: str, split: str = "train", limit: int = 6) -> dict[str, Any]:
        bundle = self.load(dataset_id)
        examples = getattr(bundle, split)
        items = []
        for example in examples[:limit]:
            projection, _ = project_for_display(example.points)
            original_projection, _ = project_for_display(example.original_points)
            transform = example.transform or {}
            summary_parts = []
            for key in ("rotation_deg", "stretch", "twist", "shear", "bend", "warp", "noise"):
                value = transform.get(key)
                if value in (None, 0, 0.0, (0, 0, 0), [0, 0, 0], (1, 1, 1), [1, 1, 1]):
                    continue
                summary_parts.append(f"{key}={value}")
            items.append(
                {
                    "structure_name": example.structure_name,
                    "task": example.task,
                    "seq_len": example.seq_len,
                    "projection": projection,
                    "original_projection": original_projection,
                    "points": example.points,
                    "original_points": example.original_points,
                    "true_groups": example.target_groups,
                    "target_text": example.target_text,
                    "transform": transform,
                    "transform_summary": ", ".join(summary_parts) or "clean",
                }
            )
        return {"dataset_id": dataset_id, "split": split, "total": len(examples), "items": items}

    def load(self, dataset_id: str) -> DatasetBundle:
        directory = self.root / dataset_id
        meta_path = directory / "meta.json"
        data_path = directory / "data.json"
        if not meta_path.exists() or not data_path.exists():
            raise DatasetStoreError(f"unknown dataset: {dataset_id}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        data = json.loads(data_path.read_text(encoding="utf-8"))
        data_spec = DataBuildSpec.model_validate(meta["data_spec"])

        def _example(item: dict[str, Any]) -> TokenExample:
            item.setdefault("transform", {})
            item.setdefault("original_points", item.get("points", []))
            return TokenExample(**item)

        train = [_example(item) for item in data.get("train", [])]
        val = [_example(item) for item in data.get("val", [])]
        test = [_example(item) for item in data.get("test", [])]
        return DatasetBundle(
            train=train,
            val=val,
            test=test,
            tokenizer=NumericTokenizer(),
            data_spec=data_spec,
            max_dim=int(meta["max_dim"]),
            structure_names=list(meta.get("structures", [])),
            skipped=int(meta.get("skipped", 0)),
        )


STORE = DatasetStore()
