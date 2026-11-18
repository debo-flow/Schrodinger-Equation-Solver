import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from scipy import sparse
from schrodinger.potentials.base import Potential
from schrodinger.numerical.grid import SpatialGrid, Precision, BoundaryCondition


@dataclass
class StabilityDiagnostics:
    """Eigenpair convergence and floating-point health diagnostics."""
    max_residual: float
    max_normalized_residual: float
    orthogonality_error: float
    all_finite: bool
    converged: bool


@dataclass
class PrecisionSensitivityResult:
    """Comparison between float32 and float64 solutions for precision sensitivity."""
    max_energy_difference: float
    max_relative_energy_difference: Optional[float]
    float32_acceptable: bool
    recommended_precision: str


@dataclass
class StabilityReport:
    """Comprehensive numerical stability and recovery report."""
    matrix_dimension: int
    dtype: str
    is_finite: bool
    gershgorin_lower_bound: float
    gershgorin_upper_bound: float
    dynamic_range: float
    condition_estimate: Optional[float]
    diagnostics: Optional[StabilityDiagnostics]
    warnings: List[str]
    recovery_recommendations: List[str]
    is_stable: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 4) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class NumericalStabilityAnalyzer:
    """
    Analyzes Hamiltonian conditioning, floating-point dynamic range,
    eigenpair residual convergence, precision sensitivity, and recovery actions.
    """

    @staticmethod
    def gershgorin_bounds(matrix: Union[np.ndarray, sparse.spmatrix]) -> Tuple[float, float]:
        """
        Computes guaranteed lower and upper bounds on all real eigenvalues
        using the Gershgorin Circle Theorem in O(nnz) time.
        """
        if sparse.issparse(matrix):
            diag = np.real(matrix.diagonal()).astype(np.float64)
            abs_mat = abs(matrix).tocsr()
            row_abs_sum = np.asarray(abs_mat.sum(axis=1)).ravel()
            radii = np.maximum(0.0, row_abs_sum - np.abs(diag))
        else:
            diag = np.real(np.diag(matrix)).astype(np.float64)
            row_abs_sum = np.sum(np.abs(matrix), axis=1)
            radii = np.maximum(0.0, row_abs_sum - np.abs(diag))

        lower = float(np.min(diag - radii))
        upper = float(np.max(diag + radii))
        return lower, upper

    @staticmethod
    def estimate_condition_number(
        matrix: Union[np.ndarray, sparse.spmatrix],
        exact_threshold: int = 600
    ) -> Optional[float]:
        """
        Computes exact 2-norm condition number for small matrices, or a fast
        spectral-bound estimate for large sparse matrices.
        """
        n = matrix.shape[0]
        if n == 0:
            return None

        if n <= exact_threshold:
            dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
            if not np.all(np.isfinite(dense)):
                return float("inf")
            return float(np.linalg.cond(dense))

        lower, upper = NumericalStabilityAnalyzer.gershgorin_bounds(matrix)
        max_mag = max(abs(lower), abs(upper))
        min_mag = min(abs(lower), abs(upper))
        if min_mag < 1e-12:
            # Matrix may have near-zero eigenvalues; use diagonal scale ratio as proxy
            diag = np.abs(matrix.diagonal()) if sparse.issparse(matrix) else np.abs(np.diag(matrix))
            pos_diag = diag[diag > 1e-15]
            if len(pos_diag) == 0:
                return float("inf")
            return float(max_mag / np.min(pos_diag))
        return float(max_mag / min_mag)

    @staticmethod
    def compute_dynamic_range(matrix: Union[np.ndarray, sparse.spmatrix]) -> float:
        """Computes ratio of largest to smallest non-zero absolute entry."""
        data = matrix.data if sparse.issparse(matrix) else np.asarray(matrix).ravel()
        abs_data = np.abs(data)
        nonzero = abs_data[(abs_data > 0) & np.isfinite(abs_data)]
        if len(nonzero) == 0:
            return 0.0
        return float(np.max(nonzero) / np.min(nonzero))

    @staticmethod
    def diagnose_eigenpairs(
        matrix: Union[np.ndarray, sparse.spmatrix],
        energies: np.ndarray,
        eigenvectors: np.ndarray,
        residual_tolerance: float = 1e-6
    ) -> StabilityDiagnostics:
        """Evaluates residual norms and eigenvector orthogonality."""
        all_finite = bool(np.all(np.isfinite(energies)) and np.all(np.isfinite(eigenvectors)))
        if not all_finite:
            return StabilityDiagnostics(
                max_residual=float("inf"),
                max_normalized_residual=float("inf"),
                orthogonality_error=float("inf"),
                all_finite=False,
                converged=False
            )

        hv = matrix.dot(eigenvectors) if sparse.issparse(matrix) else matrix @ eigenvectors
        ev = eigenvectors * energies[np.newaxis, :]
        res_norms = np.linalg.norm(hv - ev, axis=0)
        norm_res = res_norms / (np.abs(energies) + 1.0)

        # Discrete eigenvector orthogonality: ||V^H V - I||_max
        gram = np.conj(eigenvectors.T) @ eigenvectors
        ortho_err = float(np.max(np.abs(gram - np.eye(len(energies)))))

        max_res = float(np.max(res_norms)) if len(res_norms) > 0 else 0.0
        max_norm_res = float(np.max(norm_res)) if len(norm_res) > 0 else 0.0
        converged = (max_norm_res <= residual_tolerance) and (ortho_err <= max(residual_tolerance * 10, 1e-5))

        return StabilityDiagnostics(
            max_residual=max_res,
            max_normalized_residual=max_norm_res,
            orthogonality_error=ortho_err,
            all_finite=True,
            converged=converged
        )

    @classmethod
    def analyze(
        cls,
        matrix: Union[np.ndarray, sparse.spmatrix],
        energies: Optional[np.ndarray] = None,
        eigenvectors: Optional[np.ndarray] = None,
        residual_tolerance: float = 1e-6,
        condition_warning_threshold: float = 1e10
    ) -> StabilityReport:
        """Generates a full numerical stability and recovery diagnostic report."""
        data = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
        is_finite = bool(np.all(np.isfinite(data)))
        dtype_str = str(matrix.dtype)

        warnings: List[str] = []
        recommendations: List[str] = []

        if not is_finite:
            warnings.append("Matrix contains NaN or Inf entries.")
            recommendations.append("Inspect potential domain/boundaries and ensure interior_only=True for infinite wells.")
            return StabilityReport(
                matrix_dimension=int(matrix.shape[0]),
                dtype=dtype_str,
                is_finite=False,
                gershgorin_lower_bound=float("nan"),
                gershgorin_upper_bound=float("nan"),
                dynamic_range=float("inf"),
                condition_estimate=float("inf"),
                diagnostics=None,
                warnings=warnings,
                recovery_recommendations=recommendations,
                is_stable=False
            )

        lower, upper = cls.gershgorin_bounds(matrix)
        dyn_range = cls.compute_dynamic_range(matrix)
        cond_est = cls.estimate_condition_number(matrix)

        if "float32" in dtype_str or "complex64" in dtype_str:
            warnings.append(f"Single-precision dtype ({dtype_str}) may suffer rounding error on fine grids.")
            recommendations.append("Switch SpatialGrid precision to Precision.FLOAT64.")

        if cond_est is not None and cond_est > condition_warning_threshold:
            warnings.append(f"High matrix condition number estimate ({cond_est:.2e}).")
            recommendations.append("Use float64 precision, reduce excess grid density, or apply shift-invert sparse solving.")

        if dyn_range > 1e12:
            warnings.append(f"Large floating-point dynamic range ({dyn_range:.2e}) detected.")
            recommendations.append("Rescale to dimensionless/atomic units to improve floating-point conditioning.")

        diagnostics: Optional[StabilityDiagnostics] = None
        if energies is not None and eigenvectors is not None:
            diagnostics = cls.diagnose_eigenpairs(
                matrix, energies, eigenvectors, residual_tolerance=residual_tolerance
            )
            if not diagnostics.converged:
                warnings.append(
                    f"Eigenpair convergence check failed (max_norm_res={diagnostics.max_normalized_residual:.2e}, "
                    f"ortho_err={diagnostics.orthogonality_error:.2e})."
                )
                recommendations.append("Fallback to direct tridiagonal/dense solver or tighten solver tolerance.")

        is_stable = (
            is_finite
            and (diagnostics is None or diagnostics.converged)
            and (cond_est is None or cond_est <= condition_warning_threshold * 1e2)
        )

        return StabilityReport(
            matrix_dimension=int(matrix.shape[0]),
            dtype=dtype_str,
            is_finite=is_finite,
            gershgorin_lower_bound=lower,
            gershgorin_upper_bound=upper,
            dynamic_range=dyn_range,
            condition_estimate=cond_est,
            diagnostics=diagnostics,
            warnings=warnings,
            recovery_recommendations=recommendations,
            is_stable=is_stable
        )

    @staticmethod
    def analyze_precision_sensitivity(
        potential: Potential,
        x_min: float,
        x_max: float,
        points: int,
        num_states: int = 3,
        mass: float = 1.0,
        hbar: float = 1.0,
        tolerance: float = 1e-4
    ) -> PrecisionSensitivityResult:
        """
        Compares float32 vs float64 energy eigenvalues to quantify precision sensitivity.
        """
        from .optimization_engine import OptimizationEngine, OptimizationConfig

        engine = OptimizationEngine(OptimizationConfig(enable_cache=False, monitor_performance=False))

        grid_32 = SpatialGrid(x_min, x_max, points, precision=Precision.FLOAT32, boundary_condition=BoundaryCondition.DIRICHLET)
        grid_64 = SpatialGrid(x_min, x_max, points, precision=Precision.FLOAT64, boundary_condition=BoundaryCondition.DIRICHLET)

        res_32 = engine.solve_system(grid_32, potential, num_eigenvalues=num_states, mass=mass, hbar=hbar)
        res_64 = engine.solve_system(grid_64, potential, num_eigenvalues=num_states, mass=mass, hbar=hbar)

        diff = np.abs(res_32.energies - res_64.energies)
        max_diff = float(np.max(diff))
        ref_mag = np.abs(res_64.energies)
        valid = ref_mag > 1e-12
        max_rel = float(np.max(diff[valid] / ref_mag[valid])) if np.any(valid) else None

        acceptable = (max_rel is not None and max_rel <= tolerance) or (max_rel is None and max_diff <= tolerance)
        return PrecisionSensitivityResult(
            max_energy_difference=max_diff,
            max_relative_energy_difference=max_rel,
            float32_acceptable=acceptable,
            recommended_precision="float32" if acceptable else "float64"
        )
