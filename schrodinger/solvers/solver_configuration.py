from dataclasses import dataclass
from enum import Enum
from typing import Optional

class SolverMethod(Enum):
    AUTO = "auto"
    DENSE = "dense"
    SPARSE = "sparse"

@dataclass
class SolverConfig:
    """Configuration parameters for the eigenvalue solvers."""
    num_eigenvalues: int = 5
    tolerance: float = 1e-10
    max_iterations: Optional[int] = None
    method: SolverMethod = SolverMethod.AUTO
    sparse_threshold: int = 500  # N > threshold uses sparse by default
    degeneracy_tolerance: float = 1e-8
