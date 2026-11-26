from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .model.config import DataBuildSpec
from .model.data import DatasetBundle, TokenExample, build_dataset
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
        meta = {
            "id": dataset_id,
            "name": name.strip() or dataset_id,
            "created_at": time.time(),
            "train_count": len(bundle.train),
            "val_count": len(bundle.val),
            "skipped": bundle.skipped,
            "max_dim": bundle.max_dim,
            "max_seq_len": data_spec.max_seq_len,
            "precision": data_spec.precision,
            "structures": [structure.value for structure in data_spec.structures],
            "target_schema": data_spec.target_schema.model_dump(mode="json"),
            "data_spec": data_spec.model_dump(mode="json"),
            "path": str(directory),
        }
        data = {
            "train": [asdict(example) for example in bundle.train],
            "val": [asdict(example) for example in bundle.val],
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (directory / "data.json").write_text(json.dumps(data), encoding="utf-8")
        print(
            f"[datasets] saved {dataset_id} name={meta['name']!r} "
            f"train={meta['train_count']} val={meta['val_count']}",
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

    def load(self, dataset_id: str) -> DatasetBundle:
        directory = self.root / dataset_id
        meta_path = directory / "meta.json"
        data_path = directory / "data.json"
        if not meta_path.exists() or not data_path.exists():
            raise DatasetStoreError(f"unknown dataset: {dataset_id}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        data = json.loads(data_path.read_text(encoding="utf-8"))
        data_spec = DataBuildSpec.model_validate(meta["data_spec"])
        train = [TokenExample(**item) for item in data.get("train", [])]
        val = [TokenExample(**item) for item in data.get("val", [])]
        return DatasetBundle(
            train=train,
            val=val,
            tokenizer=NumericTokenizer(),
            data_spec=data_spec,
            max_dim=int(meta["max_dim"]),
            structure_names=list(meta.get("structures", [])),
            skipped=int(meta.get("skipped", 0)),
        )


STORE = DatasetStore()
