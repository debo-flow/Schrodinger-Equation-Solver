from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class ParallelExecutionEngine:
    """
    Optional thread-based parallel executor for independent numerical workloads.
    Automatically falls back to sequential execution for small batches to avoid
    scheduling overhead, and preserves deterministic input ordering.
    """

    def __init__(self, max_workers: Optional[int] = None, min_tasks_for_parallel: int = 2):
        if max_workers is not None and max_workers < 1:
            raise ValueError("max_workers must be >= 1 or None.")
        if min_tasks_for_parallel < 1:
            raise ValueError("min_tasks_for_parallel must be >= 1.")

        self.max_workers = max_workers
        self.min_tasks_for_parallel = min_tasks_for_parallel

    def should_parallelize(self, num_tasks: int) -> bool:
        """Determines whether parallel execution is worthwhile for the batch size."""
        if self.max_workers == 1:
            return False
        return num_tasks >= self.min_tasks_for_parallel

    def map_ordered(self, fn: Callable[[T], R], items: Sequence[T]) -> List[R]:
        """
        Applies `fn` across `items`, guaranteeing output order matches input order.
        """
        if not items:
            return []

        if not self.should_parallelize(len(items)):
            return [fn(item) for item in items]

        workers = self.max_workers if self.max_workers is not None else min(len(items), 4)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(fn, items))
