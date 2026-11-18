import json
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Iterator, Optional


@dataclass
class StageMetrics:
    """Timing and memory metrics for a single execution stage."""
    name: str
    elapsed_seconds: float
    memory_current_bytes: int
    memory_peak_bytes: int


@dataclass
class PerformanceReport:
    """Structured performance report across solver execution stages."""
    total_time_seconds: float
    peak_memory_bytes: int
    matrix_construction_time: float
    eigenvalue_computation_time: float
    wavefunction_reconstruction_time: float
    stages: Dict[str, StageMetrics] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def peak_memory_mb(self) -> float:
        return self.peak_memory_bytes / (1024.0 * 1024.0)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["peak_memory_mb"] = self.peak_memory_mb
        return data

    def to_json(self, indent: int = 4) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class PerformanceMonitor:
    """
    Lightweight profiler measuring stage-level wall-clock execution time
    and Python/NumPy heap memory peaks via tracemalloc.
    """

    def __init__(self, track_memory: bool = True):
        self.track_memory = track_memory
        self._stages: Dict[str, StageMetrics] = {}
        self._metadata: Dict[str, Any] = {}
        self._start_time: Optional[float] = None
        self._total_time: float = 0.0
        self._overall_peak_bytes: int = 0
        self._owns_tracemalloc: bool = False

    def start(self) -> None:
        """Starts overall profiling session."""
        self._stages.clear()
        self._overall_peak_bytes = 0
        if self.track_memory:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
                self._owns_tracemalloc = True
            else:
                self._owns_tracemalloc = False
            tracemalloc.reset_peak()
        self._start_time = time.perf_counter()

    def stop(self) -> None:
        """Stops overall profiling session and records totals."""
        if self._start_time is not None:
            self._total_time = time.perf_counter() - self._start_time
            self._start_time = None
        if self.track_memory and tracemalloc.is_tracing():
            _, peak = tracemalloc.get_traced_memory()
            self._overall_peak_bytes = max(self._overall_peak_bytes, int(peak))
            if self._owns_tracemalloc:
                tracemalloc.stop()
                self._owns_tracemalloc = False

    def add_metadata(self, key: str, value: Any) -> None:
        """Attaches contextual metadata (e.g., grid size, solver method) to the report."""
        self._metadata[key] = value

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Context manager that times and measures memory for a named stage."""
        started_local_trace = False
        if self.track_memory:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
                started_local_trace = True
            mem_before, _ = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()
        else:
            mem_before = 0

        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            if self.track_memory and tracemalloc.is_tracing():
                mem_after, mem_peak = tracemalloc.get_traced_memory()
                mem_delta = max(0, int(mem_after - mem_before))
                peak_bytes = int(mem_peak)
                self._overall_peak_bytes = max(self._overall_peak_bytes, peak_bytes)
                if started_local_trace:
                    tracemalloc.stop()
            else:
                mem_delta = 0
                peak_bytes = 0

            self._stages[name] = StageMetrics(
                name=name,
                elapsed_seconds=elapsed,
                memory_current_bytes=mem_delta,
                memory_peak_bytes=peak_bytes
            )

    def build_report(self) -> PerformanceReport:
        """Compiles recorded stages into a structured PerformanceReport."""
        total = self._total_time
        if total == 0.0 and self._stages:
            total = sum(s.elapsed_seconds for s in self._stages.values())

        def _stage_time(key: str) -> float:
            return self._stages[key].elapsed_seconds if key in self._stages else 0.0

        return PerformanceReport(
            total_time_seconds=total,
            peak_memory_bytes=self._overall_peak_bytes,
            matrix_construction_time=_stage_time("matrix_construction"),
            eigenvalue_computation_time=_stage_time("eigenvalue_computation"),
            wavefunction_reconstruction_time=_stage_time("wavefunction_reconstruction"),
            stages=dict(self._stages),
            metadata=dict(self._metadata)
        )
