import copy
import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np
from scipy import sparse
from schrodinger.numerical.grid import SpatialGrid
from schrodinger.potentials.base import Potential


@dataclass
class CacheStats:
    """Tracks cache usage and effectiveness."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    invalidations: int = 0
    current_entries: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return float(self.hits / total) if total > 0 else 0.0


class ComputationCache:
    """
    Thread-safe, bounded LRU cache for reusable quantum computations.
    Supports grid metadata, potential evaluations, and Hamiltonian matrices
    with automatic parameter-fingerprint invalidation.
    """

    def __init__(self, max_entries: int = 128, copy_on_get: bool = True):
        if max_entries <= 0:
            raise ValueError("max_entries must be a positive integer.")
        self._max_entries = int(max_entries)
        self._copy_on_get = bool(copy_on_get)
        self._store: "OrderedDict[Tuple[str, str], Any]" = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()

    @staticmethod
    def _serialize_value(val: Any) -> Any:
        if isinstance(val, (int, float, str, bool)) or val is None:
            return val
        if isinstance(val, np.generic):
            return val.item()
        if isinstance(val, np.ndarray):
            return {
                "__ndarray__": True,
                "dtype": str(val.dtype),
                "shape": list(val.shape),
                "sha256": hashlib.sha256(val.tobytes()).hexdigest()
            }
        if isinstance(val, (list, tuple)):
            return [ComputationCache._serialize_value(v) for v in val]
        if isinstance(val, dict):
            return {str(k): ComputationCache._serialize_value(v) for k, v in sorted(val.items())}
        return repr(val)

    @classmethod
    def make_fingerprint(cls, payload: Dict[str, Any]) -> str:
        """Creates a deterministic SHA-256 fingerprint from structured parameters."""
        normalized = cls._serialize_value(payload)
        encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def grid_fingerprint(cls, grid: SpatialGrid) -> str:
        """Generates a unique fingerprint capturing all immutable grid properties."""
        return cls.make_fingerprint({
            "x_min": grid.x_min,
            "x_max": grid.x_max,
            "points": grid.points,
            "dtype": str(grid.grid.dtype),
            "boundary_condition": grid.boundary_condition.name
        })

    @classmethod
    def potential_fingerprint(cls, potential: Potential) -> str:
        """Generates a fingerprint from the potential class and its current attributes."""
        attrs = {
            k: v for k, v in vars(potential).items()
            if not k.startswith("__")
        }
        return cls.make_fingerprint({
            "class": f"{potential.__class__.__module__}.{potential.__class__.__qualname__}",
            "attributes": attrs
        })

    @classmethod
    def hamiltonian_fingerprint(
        cls,
        grid: SpatialGrid,
        potential: Potential,
        mass: float,
        hbar: float,
        unit_system: str,
        accuracy_order: int,
        interior_only: bool,
        representation: str = "sparse"
    ) -> str:
        """Generates a fingerprint for a fully specified discretized Hamiltonian."""
        return cls.make_fingerprint({
            "grid": cls.grid_fingerprint(grid),
            "potential": cls.potential_fingerprint(potential),
            "mass": float(mass),
            "hbar": float(hbar),
            "unit_system": unit_system,
            "accuracy_order": int(accuracy_order),
            "interior_only": bool(interior_only),
            "representation": representation
        })

    @staticmethod
    def _safe_copy(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.copy()
        if sparse.issparse(value):
            return value.copy()
        if isinstance(value, tuple):
            return tuple(ComputationCache._safe_copy(v) for v in value)
        if isinstance(value, dict):
            return {k: ComputationCache._safe_copy(v) for k, v in value.items()}
        return copy.deepcopy(value)

    def get(self, namespace: str, key: str) -> Optional[Any]:
        """Retrieves an item if present and updates LRU ordering."""
        composite = (namespace, key)
        with self._lock:
            if composite not in self._store:
                self._stats.misses += 1
                return None
            self._store.move_to_end(composite)
            self._stats.hits += 1
            val = self._store[composite]
            return self._safe_copy(val) if self._copy_on_get else val

    def set(self, namespace: str, key: str, value: Any) -> None:
        """Stores an item in the cache, evicting the oldest entry if full."""
        composite = (namespace, key)
        stored_val = self._safe_copy(value) if self._copy_on_get else value
        with self._lock:
            if composite in self._store:
                self._store.move_to_end(composite)
                self._store[composite] = stored_val
            else:
                if len(self._store) >= self._max_entries:
                    self._store.popitem(last=False)
                    self._stats.evictions += 1
                self._store[composite] = stored_val
            self._stats.current_entries = len(self._store)

    def get_or_compute(self, namespace: str, key: str, compute_fn: Callable[[], Any]) -> Any:
        """Returns cached result if available; otherwise computes, stores, and returns it."""
        cached = self.get(namespace, key)
        if cached is not None:
            return cached
        result = compute_fn()
        self.set(namespace, key, result)
        return self._safe_copy(result) if self._copy_on_get else result

    def invalidate(self, namespace: str, key: Optional[str] = None) -> int:
        """
        Invalidates a specific (namespace, key) entry or all entries in a namespace.
        Returns the number of removed entries.
        """
        with self._lock:
            if key is not None:
                composite = (namespace, key)
                if composite in self._store:
                    del self._store[composite]
                    self._stats.invalidations += 1
                    self._stats.current_entries = len(self._store)
                    return 1
                return 0

            to_remove = [k for k in self._store if k[0] == namespace]
            for k in to_remove:
                del self._store[k]
            self._stats.invalidations += len(to_remove)
            self._stats.current_entries = len(self._store)
            return len(to_remove)

    def clear(self) -> None:
        """Clears all cached entries."""
        with self._lock:
            removed = len(self._store)
            self._store.clear()
            self._stats.invalidations += removed
            self._stats.current_entries = 0

    @property
    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                invalidations=self._stats.invalidations,
                current_entries=len(self._store)
            )
