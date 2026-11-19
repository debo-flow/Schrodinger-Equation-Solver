from typing import Dict, Any, List, Literal
import numpy as np
from scipy.interpolate import interp1d
from schrodinger.potentials.base import Potential
from schrodinger.numerical.grid import SpatialGrid, BoundaryCondition
from schrodinger.optimization.optimization_engine import OptimizationEngine, OptimizationConfig
from .analytical_solver import AnalyticalSolution


class NumericalReferenceSolution(AnalyticalSolution):
    """
    High-resolution converged numerical reference solver for potentials lacking
    closed-form elementary solutions (e.g., Quartic Anharmonic, Periodic, Barrier,
    Regularized Coulomb-Like). Explicitly tagged with `is_exact = False`.
    """

    def __init__(
        self,
        potential: Potential,
        x_min: float,
        x_max: float,
        num_states: int = 5,
        reference_points: int = 2001,
        accuracy_order: int = 4,
        mass: float = 1.0,
        hbar: float = 1.0,
        unit_system: Literal["dimensionless", "si"] = "dimensionless"
    ):
        super().__init__(mass=mass, hbar=hbar, unit_system=unit_system)
        if not isinstance(potential, Potential):
            raise TypeError("potential must be a Potential instance.")
        if potential.unit_system != unit_system:
            raise ValueError(f"Unit mismatch: potential '{potential.unit_system}' != '{unit_system}'.")
        if num_states <= 0:
            raise ValueError("num_states must be strictly positive.")
        if reference_points < 101:
            raise ValueError("reference_points must be at least 101 for a converged reference.")

        self.potential = potential
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.num_states = int(num_states)
        self.reference_points = int(reference_points)
        self.accuracy_order = int(accuracy_order)

        ref_grid = SpatialGrid(
            x_min=self.x_min,
            x_max=self.x_max,
            points=self.reference_points,
            boundary_condition=BoundaryCondition.DIRICHLET
        )
        engine = OptimizationEngine(
            OptimizationConfig(
                accuracy_order=self.accuracy_order,
                enable_cache=False,
                monitor_performance=False,
                run_stability_check=True
            )
        )
        res = engine.solve_system(
            ref_grid,
            potential,
            num_eigenvalues=self.num_states,
            mass=self.mass,
            hbar=self.hbar,
            unit_system=self.unit_system
        )
        self._ref_x = ref_grid.grid
        self._ref_energies = res.energies
        self._ref_wavefunctions = res.wavefunctions

    @property
    def is_exact(self) -> bool:
        """Explicitly marks this reference as an approximate numerical benchmark."""
        return False

    @property
    def name(self) -> str:
        return f"NumericalReference({self.potential.__class__.__name__})"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "potential": repr(self.potential),
            "x_min": self.x_min,
            "x_max": self.x_max,
            "reference_points": self.reference_points,
            "accuracy_order": self.accuracy_order,
            "is_exact": False,
            "mass": self.mass,
            "hbar": self.hbar,
            "unit_system": self.unit_system
        }

    def _validate_quantum_number(self, n: int) -> None:
        if not isinstance(n, (int, np.integer)) or n < 0 or n >= self.num_states:
            raise ValueError(f"Quantum number n must be within [0, {self.num_states - 1}]. Received: {n}")

    def quantum_numbers(self, num_states: int) -> List[int]:
        if num_states <= 0 or num_states > self.num_states:
            raise ValueError(f"num_states must be within [1, {self.num_states}].")
        return list(range(0, num_states))

    def energy(self, n: int) -> float:
        self._validate_quantum_number(n)
        return float(self._ref_energies[n])

    def normalization_constant(self, n: int) -> float:
        self._validate_quantum_number(n)
        return 1.0

    def wavefunction(self, n: int, x: np.ndarray) -> np.ndarray:
        self._validate_quantum_number(n)
        x_arr = np.asarray(x, dtype=np.float64)
        ref_psi = np.real(self._ref_wavefunctions[n].psi)
        interp_fn = interp1d(
            self._ref_x,
            ref_psi,
            kind="cubic",
            bounds_error=False,
            fill_value=0.0
        )
        return np.asarray(interp_fn(x_arr), dtype=np.float64)

    def verify_boundary_conditions(self, n: int, x: np.ndarray, tolerance: float = 1e-6) -> bool:
        self._validate_quantum_number(n)
        vals = self.wavefunction(n, np.array([x[0], x[-1]], dtype=np.float64))
        return bool(np.all(np.abs(vals) <= tolerance))
