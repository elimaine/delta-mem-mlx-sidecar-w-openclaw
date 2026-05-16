from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from delta_mem_sidecar.runtime import DeltaRuntime, RuntimeState


@dataclass(frozen=True)
class StateMetadata:
    state_key_hash: str
    created_at: datetime
    updated_at: datetime
    updates: int


@dataclass
class _StateEntry:
    state: RuntimeState
    created_at: datetime
    updated_at: datetime


class StateKeyResolver:
    def __init__(self, namespace: str = "delta-mem-sidecar-v0") -> None:
        self._namespace = namespace

    def storage_key(self, logical_key: str) -> str:
        scoped = f"{self._namespace}:{logical_key}".encode("utf-8")
        return hashlib.sha256(scoped).hexdigest()


class InMemoryStateStore:
    """Thread-safe session state registry keyed by hashed IDs."""

    def __init__(
        self,
        runtime: DeltaRuntime,
        resolver: StateKeyResolver | None = None,
        persistence_dir: str | Path | None = None,
    ) -> None:
        self._runtime = runtime
        self._resolver = resolver or StateKeyResolver()
        self._states: dict[str, _StateEntry] = {}
        self._persistence_dir = Path(persistence_dir) if persistence_dir else None
        self._lock = RLock()

    def get_or_create(self, logical_key: str) -> RuntimeState:
        storage_key = self._resolver.storage_key(logical_key)
        with self._lock:
            entry = self._states.get(storage_key)
            if entry is None:
                entry = self._load_entry(storage_key) or self._fresh_entry()
                self._states[storage_key] = entry
            return entry.state

    def mark_updated(self, logical_key: str) -> StateMetadata:
        storage_key = self._resolver.storage_key(logical_key)
        with self._lock:
            entry = self._states[storage_key]
            entry.updated_at = datetime.now(UTC)
            self._persist_entry(storage_key, entry)
            return self._metadata_for(storage_key, entry)

    def reset(self, logical_key: str) -> StateMetadata:
        storage_key = self._resolver.storage_key(logical_key)
        with self._lock:
            now = datetime.now(UTC)
            entry = _StateEntry(
                state=self._runtime.fresh_state(),
                created_at=now,
                updated_at=now,
            )
            self._states[storage_key] = entry
            self._remove_persisted_entry(storage_key)
            self._persist_entry(storage_key, entry)
            return self._metadata_for(storage_key, entry)

    def metadata(self, logical_key: str) -> StateMetadata | None:
        storage_key = self._resolver.storage_key(logical_key)
        with self._lock:
            entry = self._states.get(storage_key)
            if entry is None:
                entry = self._load_entry(storage_key)
                if entry is None:
                    return None
                self._states[storage_key] = entry
            return self._metadata_for(storage_key, entry)

    def _fresh_entry(self) -> _StateEntry:
        now = datetime.now(UTC)
        return _StateEntry(
            state=self._runtime.fresh_state(),
            created_at=now,
            updated_at=now,
        )

    def _entry_dir(self, storage_key: str) -> Path | None:
        if self._persistence_dir is None:
            return None
        return self._persistence_dir / storage_key

    def _load_entry(self, storage_key: str) -> _StateEntry | None:
        path = self._entry_dir(storage_key)
        if path is None:
            return None
        metadata_path = path / "metadata.json"
        load_state = getattr(self._runtime, "load_state", None)
        if not metadata_path.exists() or not callable(load_state):
            return None
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        return _StateEntry(
            state=load_state(path),
            created_at=_parse_datetime(data["created_at"]),
            updated_at=_parse_datetime(data["updated_at"]),
        )

    def _persist_entry(self, storage_key: str, entry: _StateEntry) -> None:
        path = self._entry_dir(storage_key)
        save_state = getattr(self._runtime, "save_state", None)
        if path is None or not callable(save_state):
            return
        path.mkdir(parents=True, exist_ok=True)
        (path / "metadata.json").write_text(
            json.dumps(
                {
                    "state_key_hash": storage_key,
                    "created_at": entry.created_at.isoformat(),
                    "updated_at": entry.updated_at.isoformat(),
                }
            ),
            encoding="utf-8",
        )
        save_state(entry.state, path)

    def _remove_persisted_entry(self, storage_key: str) -> None:
        path = self._entry_dir(storage_key)
        if path is not None and path.exists():
            shutil.rmtree(path)

    def _metadata_for(self, storage_key: str, entry: _StateEntry) -> StateMetadata:
        return StateMetadata(
            state_key_hash=storage_key,
            created_at=entry.created_at,
            updated_at=entry.updated_at,
            updates=entry.state.updates,
        )


def _parse_datetime(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
