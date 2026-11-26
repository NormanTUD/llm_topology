from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

STATE_PATH = Path.cwd() / "state" / "ui_state.json"


class UIStateStore:
    def __init__(self, path: Path = STATE_PATH) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            self._state = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self._state = {}

    def get(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def save(self, patch: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._state.update(patch)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._state), encoding="utf-8")
            return dict(self._state)


STORE = UIStateStore()
