from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
from scipy import integrate, linalg, sparse
from scipy.sparse.linalg import eigsh

from schrodinger.numerical.grid import SpatialGrid, BoundaryCondition
from schrodinger.numerical.finite_difference import FiniteDifferenceEngine
from schrodinger.potentials.base import Potential
from schrodinger.operators.kinetic_operator import KineticEnergyOperator
from schrodinger.operators.potential_operator import PotentialEnergyOperator
from schrodinger.operators.hamiltonian import Hamiltonian
from schrodinger.operators.operator_validation import OperatorValidator
from schrodinger.wavefunctions.wavefunction import Wavefunction

from .computation_cache import ComputationCache
from .performance_monitor import PerformanceMonitor, PerformanceReport
from .solver_selector import OptimizedSolverMethod, SolverSelector, SolverSelectionCriteria
from .numerical_stability import NumericalStabilityAnalyzer, StabilityReport


@dataclass
class OptimizationConfig:
    """Configuration for the optimized Schrödinger solver pipeline."""
    accuracy_order: int = 2
    tolerance: float = 1e-10
    max_iterations: Optional[int] = None
    available_memory_mb: float = 512.0
    forced_method: Optional[OptimizedSolverMethod] = None
    enable_cache: bool = True
    monitor_performance: bool = True
    run_stability_check: bool = True
    deterministic_seed: int = 42


@dataclass
class OptimizedSolveResult:
    """Structured output returned by OptimizationEngine.solve_system."""
    energies: np.ndarray
    wavefunctions: List[Wavefunction]
    method_used: OptimizedSolverMethod
    used_fallback: bool
    cache_hit: bool
    performance_report: Optional[PerformanceReport]
    stability_report: Optional[StabilityReport]


class OptimizationEngine:
    """
    High-performance Schrödinger solver orchestrator combining:
      - Single-pass sparse / tridiagonal / banded Hamiltonian construction
      - Parameter-fingerprinted LRU caching
      - Automatic solver selection with fallback recovery
      - Vectorized wavefunction reconstruction & continuous normalization
      - Integrated performance profiling & numerical stability diagnostics
    """

    def __init__(
        self,
        config: OptimizationConfig = OptimizationConfig(),
        cache: Optional[ComputationCache] = None
    ):
        self.config = config
        self.cache = cache if cache is not None else ComputationCache()

    def evaluate_potential_cached(
        self,
        grid: SpatialGrid,
        potential: Potential,
        interior_only: bool = True
    ) -> np.ndarray:
        """Evaluates V(x) on the grid with safe parameter-based caching."""
        def _compute() -> np.ndarray:
            x_pts = grid.interior_points if interior_only else grid.grid
            v_vals = potential(x_pts).astype(grid.grid.dtype)
            if np.iscomplexobj(v_vals) and np.any(np.imag(v_vals) != 0):
                raise ValueError("Potential evaluation must be strictly real.")
            return v_vals

        if not self.config.enable_cache:
            return _compute()

        key = ComputationCache.make_fingerprint({
            "grid": ComputationCache.grid_fingerprint(grid),
            "potential": ComputationCache.potential_fingerprint(potential),
            "interior_only": interior_only
        })
        return self.cache.get_or_compute("potential", key, _compute)

    def build_tridiagonal_components(
        self,
        grid: SpatialGrid,
        potential: Potential,
        mass: float,
        hbar: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Directly builds main diagonal `d` and off-diagonal `e` for a 2nd-order
        interior Dirichlet 1D Hamiltonian in O(N) memory and time.
        """
        v_int = self.evaluate_potential_cached(grid, potential, interior_only=True)
        n_int = len(v_int)
        dx2 = grid.dx ** 2
        off_val = -(hbar ** 2) / (2.0 * mass * dx2)
        main_kin = (hbar ** 2) / (mass * dx2)

        d = (v_int + main_kin).astype(grid.grid.dtype, copy=False)
        e = np.full(n_int - 1, off_val, dtype=grid.grid.dtype)
        return d, e

    def build_banded_components(
        self,
        grid: SpatialGrid,
        potential: Potential,
        mass: float,
        hbar: float
    ) -> np.ndarray:
        """
        Directly builds lower-banded storage array (3, N-2) for a 4th-order
        interior Dirichlet 1D Hamiltonian for scipy.linalg.eig_banded.
        """
        v_int = self.evaluate_potential_cached(grid, potential, interior_only=True)
        n_int = len(v_int)
        dx2 = grid.dx ** 2
        scale = (hbar ** 2) / (24.0 * mass * dx2)

        ab = np.zeros((3, n_int), dtype=grid.grid.dtype)
        ab[0, :] = v_int + 30.0 * scale
        ab[1, :-1] = -16.0 * scale
        ab[2, :-2] = 1.0 * scale
        return ab

    def build_sparse_hamiltonian(
        self,
        grid: SpatialGrid,
        potential: Potential,
        mass: float = 1.0,
        hbar: float = 1.0,
        unit_system: Literal["dimensionless", "si"] = "dimensionless",
        accuracy_order: Optional[int] = None,
        interior_only: bool = True
    ) -> Tuple[sparse.csr_matrix, bool]:
        """
        Constructs the sparse CSR Hamiltonian in a single pass for interior Dirichlet
        grids (avoiding intermediate LIL and separate T+V allocations), with caching.
        Returns (matrix, cache_hit).
        """
        OperatorValidator.validate_physical_constants(mass, hbar)
        OperatorValidator.validate_unit_consistency(unit_system, potential.unit_system)
        order = accuracy_order if accuracy_order is not None else self.config.accuracy_order

        cache_key = ComputationCache.hamiltonian_fingerprint(
            grid, potential, mass, hbar, unit_system, order, interior_only, "sparse"
        )
        if self.config.enable_cache:
            cached = self.cache.get("hamiltonian", cache_key)
            if cached is not None:
                return cached, True

        if interior_only and grid.boundary_condition == BoundaryCondition.DIRICHLET and order in (2, 4):
            v_int = self.evaluate_potential_cached(grid, potential, interior_only=True)
            n_int = len(v_int)
            dx2 = grid.dx ** 2
            dtype = grid.grid.dtype

            if order == 2:
                off = -(hbar ** 2) / (2.0 * mass * dx2)
                main = v_int + (hbar ** 2) / (mass * dx2)
                off_arr = np.full(n_int - 1, off, dtype=dtype)
                h_mat = sparse.diags(
                    [off_arr, main.astype(dtype), off_arr],
                    offsets=[-1, 0, 1],
                    shape=(n_int, n_int),
                    format="csr",
                    dtype=dtype
                )
            else:
                scale = (hbar ** 2) / (24.0 * mass * dx2)
                d0 = (v_int + 30.0 * scale).astype(dtype)
                d1 = np.full(n_int - 1, -16.0 * scale, dtype=dtype)
                d2 = np.full(n_int - 2, 1.0 * scale, dtype=dtype)
                h_mat = sparse.diags(
                    [d2, d1, d0, d1, d2],
                    offsets=[-2, -1, 0, 1, 2],
                    shape=(n_int, n_int),
                    format="csr",
                    dtype=dtype
                )
        else:
            # General fallback using validated Milestone 5 Hamiltonian builder
            fd_engine = FiniteDifferenceEngine(grid)
            kin_op = KineticEnergyOperator(fd_engine, mass=mass, hbar=hbar, unit_system=unit_system)
            pot_op = PotentialEnergyOperator(grid, potential)
            h_mat = Hamiltonian(kin_op, pot_op).build_sparse(accuracy_order=order, interior_only=interior_only)

        if self.config.enable_cache:
            self.cache.set("hamiltonian", cache_key, h_mat)
        return h_mat, False

    def _solve_with_method(
        self,
        method: OptimizedSolverMethod,
        h_sparse: sparse.csr_matrix,
        num_eigenvalues: int,
        tridiag_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        banded_data: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Executes the chosen numerical eigensolver and returns sorted (energies, vectors)."""
        n = h_sparse.shape[0]
        k = min(num_eigenvalues, n)

        if method == OptimizedSolverMethod.TRIDIAGONAL:
            if tridiag_data is not None:
                d, e = tridiag_data
            else:
                d = h_sparse.diagonal().astype(np.float64)
                e = h_sparse.diagonal(1).astype(np.float64)
            energies, vectors = linalg.eigh_tridiagonal(
                d, e, select="i", select_range=(0, k - 1)
            )
            return np.asarray(energies, dtype=np.float64), np.asarray(vectors)

        if method == OptimizedSolverMethod.BANDED:
            if banded_data is None:
                raise ValueError("Banded storage data required for BANDED solver.")
            energies, vectors = linalg.eig_banded(
                banded_data, lower=True, select="i", select_range=(0, k - 1)
            )
            return np.asarray(energies, dtype=np.float64), np.asarray(vectors)

        if method == OptimizedSolverMethod.DENSE:
            dense_mat = h_sparse.toarray()
            energies, vectors = linalg.eigh(dense_mat, subset_by_index=(0, k - 1))
            return np.asarray(energies, dtype=np.float64), np.asarray(vectors)

        # Deterministic starting vector for ARPACK reproducibility
        rng = np.random.default_rng(self.config.deterministic_seed)
        v0 = rng.standard_normal(n)

        if method == OptimizedSolverMethod.SPARSE_SHIFT_INVERT:
            if k >= n - 1:
                raise ValueError("SPARSE_SHIFT_INVERT requires k < N - 1.")
            # Sparse LU factorization is most stable in float64/complex128
            work_mat = h_sparse.astype(np.complex128 if np.iscomplexobj(h_sparse) else np.float64)
            lower_bound, _ = NumericalStabilityAnalyzer.gershgorin_bounds(work_mat)
            sigma = lower_bound - 1.0
            energies, vectors = eigsh(
                work_mat,
                k=k,
                sigma=sigma,
                which="LM",
                tol=self.config.tolerance,
                maxiter=self.config.max_iterations,
                v0=v0
            )
        else:
            # SPARSE_DIRECT
            if k >= n - 1:
                raise ValueError("SPARSE_DIRECT requires k < N - 1.")
            energies, vectors = eigsh(
                h_sparse,
                k=k,
                which="SA",
                tol=self.config.tolerance,
                maxiter=self.config.max_iterations,
                v0=v0.astype(h_sparse.dtype)
            )

        idx = np.argsort(energies)
        return np.asarray(energies[idx], dtype=np.float64), np.asarray(vectors[:, idx])

    @staticmethod
    def reconstruct_wavefunctions_vectorized(
        grid: SpatialGrid,
        energies: np.ndarray,
        raw_eigenvectors: np.ndarray
    ) -> List[Wavefunction]:
        """
        Vectorized boundary padding, continuous Simpson normalization, and
        global phase standardization across all eigenstates simultaneously.
        """
        num_states = len(energies)
        is_interior_only = (raw_eigenvectors.shape[0] == grid.points - 2)

        if is_interior_only and grid.boundary_condition == BoundaryCondition.DIRICHLET:
            full_psis = np.zeros((grid.points, num_states), dtype=np.complex128)
            full_psis[1:-1, :] = raw_eigenvectors
        else:
            full_psis = raw_eigenvectors.astype(np.complex128, copy=True)

        # Vectorized continuous Simpson integration across all columns
        densities = np.abs(full_psis) ** 2
        integrals = integrate.simpson(y=densities, x=grid.grid, axis=0)
        if np.any(integrals <= 0):
            raise ValueError("Non-positive probability integral encountered during reconstruction.")
        norms = np.sqrt(integrals)
        full_psis /= norms[np.newaxis, :]

        # Vectorized global phase standardization (peak amplitude real & positive)
        max_indices = np.argmax(np.abs(full_psis), axis=0)
        peak_vals = full_psis[max_indices, np.arange(num_states)]
        phases = np.angle(peak_vals)
        full_psis *= np.exp(-1j * phases)[np.newaxis, :]

        # Clean negligible numerical imaginary noise per state
        max_imag = np.max(np.abs(np.imag(full_psis)), axis=0)
        real_mask = max_imag < 1e-12
        if np.any(real_mask):
            full_psis[:, real_mask] = np.real(full_psis[:, real_mask]).astype(np.complex128)

        return [
            Wavefunction(x=grid.grid, psi=full_psis[:, i], energy=float(energies[i]))
            for i in range(num_states)
        ]

    def solve_system(
        self,
        grid: SpatialGrid,
        potential: Potential,
        num_eigenvalues: int = 5,
        mass: float = 1.0,
        hbar: float = 1.0,
        unit_system: Literal["dimensionless", "si"] = "dimensionless",
        interior_only: bool = True
    ) -> OptimizedSolveResult:
        """
        Executes the complete optimized pipeline: Hamiltonian assembly,
        automatic solver selection + execution, vectorized reconstruction,
        performance profiling, and stability verification.
        """
        monitor = PerformanceMonitor(track_memory=True) if self.config.monitor_performance else None
        if monitor is not None:
            monitor.start()

        order = self.config.accuracy_order
        is_interior_dirichlet = interior_only and (grid.boundary_condition == BoundaryCondition.DIRICHLET)

        # 1. Matrix Construction
        tridiag_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
        banded_data: Optional[np.ndarray] = None

        if monitor is not None:
            with monitor.stage("matrix_construction"):
                h_sparse, cache_hit = self.build_sparse_hamiltonian(
                    grid, potential, mass=mass, hbar=hbar, unit_system=unit_system,
                    accuracy_order=order, interior_only=interior_only
                )
                if is_interior_dirichlet and order == 2:
                    tridiag_data = self.build_tridiagonal_components(grid, potential, mass, hbar)
                elif is_interior_dirichlet and order == 4:
                    banded_data = self.build_banded_components(grid, potential, mass, hbar)
        else:
            h_sparse, cache_hit = self.build_sparse_hamiltonian(
                grid, potential, mass=mass, hbar=hbar, unit_system=unit_system,
                accuracy_order=order, interior_only=interior_only
            )
            if is_interior_dirichlet and order == 2:
                tridiag_data = self.build_tridiagonal_components(grid, potential, mass, hbar)
            elif is_interior_dirichlet and order == 4:
                banded_data = self.build_banded_components(grid, potential, mass, hbar)

        # 2. Automatic Solver Selection
        n_dim = h_sparse.shape[0]
        if self.config.forced_method is not None:
            selected_method = self.config.forced_method
            fallback_method = OptimizedSolverMethod.DENSE
        else:
            sparsity_ratio = 1.0 - (h_sparse.nnz / float(n_dim * n_dim))
            bandwidth = 1 if (is_interior_dirichlet and order == 2) else (2 if (is_interior_dirichlet and order == 4) else None)
            criteria = SolverSelectionCriteria(
                dimension=n_dim,
                num_eigenvalues=min(num_eigenvalues, n_dim),
                sparsity_ratio=sparsity_ratio,
                is_complex=bool(np.iscomplexobj(h_sparse)),
                is_symmetric_tridiagonal=(tridiag_data is not None),
                bandwidth=bandwidth,
                available_memory_mb=self.config.available_memory_mb,
                accuracy_tolerance=self.config.tolerance
            )
            rec = SolverSelector.select(criteria)
            selected_method = rec.method
            fallback_method = rec.fallback_method

        # 3. Eigenvalue Computation with Fallback Recovery
        used_fallback = False
        method_used = selected_method

        def _run_eigen() -> Tuple[np.ndarray, np.ndarray]:
            nonlocal used_fallback, method_used
            try:
                return self._solve_with_method(
                    selected_method, h_sparse, num_eigenvalues, tridiag_data, banded_data
                )
            except Exception:
                if fallback_method == selected_method:
                    raise
                used_fallback = True
                method_used = fallback_method
                return self._solve_with_method(
                    fallback_method, h_sparse, num_eigenvalues, tridiag_data, banded_data
                )

        if monitor is not None:
            with monitor.stage("eigenvalue_computation"):
                energies, vectors = _run_eigen()
        else:
            energies, vectors = _run_eigen()

        # 4. Vectorized Wavefunction Reconstruction
        if monitor is not None:
            with monitor.stage("wavefunction_reconstruction"):
                wavefunctions = self.reconstruct_wavefunctions_vectorized(grid, energies, vectors)
        else:
            wavefunctions = self.reconstruct_wavefunctions_vectorized(grid, energies, vectors)

        # 5. Stability Diagnostics
        stability_report: Optional[StabilityReport] = None
        if self.config.run_stability_check:
            stability_report = NumericalStabilityAnalyzer.analyze(
                h_sparse,
                energies=energies,
                eigenvectors=vectors,
                residual_tolerance=max(self.config.tolerance * 1e4, 1e-5)
            )

        perf_report: Optional[PerformanceReport] = None
        if monitor is not None:
            monitor.stop()
            monitor.add_metadata("grid_points", grid.points)
            monitor.add_metadata("num_eigenvalues", num_eigenvalues)
            monitor.add_metadata("method_used", method_used.value)
            monitor.add_metadata("cache_hit", cache_hit)
            perf_report = monitor.build_report()

        return OptimizedSolveResult(
            energies=energies,
            wavefunctions=wavefunctions,
            method_used=method_used,
            used_fallback=used_fallback,
            cache_hit=cache_hit,
            performance_report=perf_report,
            stability_report=stability_report
        )
