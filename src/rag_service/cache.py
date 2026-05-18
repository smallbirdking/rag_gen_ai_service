from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Optional


@dataclass
class CacheItem:
    value: Dict[str, Any]
    expires_at: float


class TTLCache:
    def __init__(self, ttl_seconds: int = 600):
        self.ttl_seconds = ttl_seconds
        self._store: Dict[str, CacheItem] = {}
        self.hits = 0
        self.misses = 0
        self._lock = Lock()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            item = self._store.get(key)
            now = time.time()
            if item and item.expires_at > now:
                self.hits += 1
                return item.value
            if item:
                self._store.pop(key, None)
            self.misses += 1
            return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        with self._lock:
            self._store[key] = CacheItem(value=value, expires_at=time.time() + self.ttl_seconds)

    @property
    def hit_rate(self) -> float:
        with self._lock:
            total = self.hits + self.misses
            return self.hits / total if total else 0.0
