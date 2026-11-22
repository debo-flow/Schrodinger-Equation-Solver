from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional, Sequence, Union
import numpy as np
from schrodinger.numerical.grid import SpatialGrid, BoundaryCondition
from schrodinger.potentials.base import Potential
from schrodinger.wavefunctions.wavefunction import Wavefunction
from schrodinger.wavefunctions.normalization import NormalizationEngine, IntegrationMethod
from schrodinger.optimization.optimization_engine import OptimizationEngine, OptimizationConfig
from .gaussian_wavepacket import GaussianWavepacketGenerator
from .plane_wave import PlaneWaveGenerator
from .superposition import SuperpositionEngine
from .coherent_state import CoherentStateGenerator


class InitialStateType(Enum):
    """Supported initial quantum state families."""
    GAUSSIAN = "gaussian"
    PLANE_WAVE = "plane_wave"
    SUPERPOSITION = "superposition"
    COHERENT = "coherent"
    STATIONARY_EIGENSTATE = "stationary_eigenstate"
    CUSTOM = "custom"


@dataclass
class InitialStateReport:
    """Validation and boundary diagnostic summary for a generated initial state."""
    state_type: InitialStateType
    continuous_norm: float
    norm_error: float
    max_boundary_amplitude: float
    boundary_leakage_ok: bool
    energy_estimate: float
    warnings: List[str]


class InitialStateEngine:
    """
    Unified initial quantum state generation and validation engine supporting:
      - Gaussian wavepackets
      - Plane-wave states
      - Multi-eigenstate superpositions
      - Harmonic oscillator coherent states
      - Stationary eigenstates of arbitrary Potentials
      - Custom user-defined callable or array wavefunctions
    """

    def __init__(
        self,
        mass: float = 1.0,
        hbar: float = 1.0,
        integration_method: IntegrationMethod = IntegrationMethod.SIMPSON,
        boundary_leakage_tolerance: float = 1e-2
    ):
        if mass <= 0 or hbar <= 0:
            raise ValueError("mass and hbar must be strictly positive.")
        if boundary_leakage_tolerance <= 0:
            raise ValueError("boundary_leakage_tolerance must be strictly positive.")

        self.mass = float(mass)
        self.hbar = float(hbar)
        self.integration_method = integration_method
        self.boundary_leakage_tolerance = float(boundary_leakage_tolerance)

    def _build_report(
        self,
        state_type: InitialStateType,
        wf: Wavefunction,
        grid: SpatialGrid,
        check_boundary_decay: bool = True
    ) -> InitialStateReport:
        norm_val = NormalizationEngine.calculate_norm(wf.psi, grid.grid, method=self.integration_method)
        norm_err = abs(norm_val - 1.0)
        edge_amp = float(max(abs(wf.psi[0]), abs(wf.psi[-1])))
        warnings: List[str] = []

        leakage_ok = True
        if check_boundary_decay and edge_amp > self.boundary_leakage_tolerance:
            leakage_ok = False
            warnings.append(
                f"Significant wavefunction amplitude at domain boundary ({edge_amp:.2e} > "
                f"{self.boundary_leakage_tolerance:.2e}). Consider widening the spatial grid."
            )

        return InitialStateReport(
            state_type=state_type,
            continuous_norm=norm_val,
            norm_error=norm_err,
            max_boundary_amplitude=edge_amp,
            boundary_leakage_ok=leakage_ok,
            energy_estimate=wf.energy,
            warnings=warnings
        )

    def create_gaussian(
        self,
        grid: SpatialGrid,
        x0: float = 0.0,
        sigma: float = 0.5,
        k0: float = 0.0,
        enforce_dirichlet_boundaries: bool = False
    ) -> tuple:
        """Creates a normalized Gaussian wavepacket and returns (Wavefunction, InitialStateReport)."""
        wf = GaussianWavepacketGenerator.generate(
            grid=grid,
            x0=x0,
            sigma=sigma,
            k0=k0,
            mass=self.mass,
            hbar=self.hbar,
            enforce_dirichlet_boundaries=enforce_dirichlet_boundaries,
            integration_method=self.integration_method
        )
        return wf, self._build_report(InitialStateType.GAUSSIAN, wf, grid, check_boundary_decay=True)

    def create_plane_wave(
        self,
        grid: SpatialGrid,
        k: Optional[float] = None,
        mode_index: Optional[int] = None,
        taper_fraction: float = 0.0
    ) -> tuple:
        """Creates a box-normalized plane-wave state and returns (Wavefunction, InitialStateReport)."""
        wf = PlaneWaveGenerator.generate(
            grid=grid,
            k=k,
            mode_index=mode_index,
            mass=self.mass,
            hbar=self.hbar,
            taper_fraction=taper_fraction,
            integration_method=self.integration_method
        )
        return wf, self._build_report(InitialStateType.PLANE_WAVE, wf, grid, check_boundary_decay=False)

    def create_superposition(
        self,
        eigenstates: Sequence[Wavefunction],
        coefficients: Sequence[complex],
        grid: Optional[SpatialGrid] = None,
        orthogonality_tolerance: float = 1e-4,
        enforce_orthogonality: bool = False
    ) -> tuple:
        """Creates a normalized eigenstate superposition and returns (Wavefunction, InitialStateReport)."""
        wf = SuperpositionEngine.create(
            eigenstates=eigenstates,
            coefficients=coefficients,
            orthogonality_tolerance=orthogonality_tolerance,
            enforce_orthogonality=enforce_orthogonality,
            integration_method=self.integration_method
        )
        target_grid = grid if grid is not None else SpatialGrid(float(wf.x[0]), float(wf.x[-1]), len(wf.x))
        return wf, self._build_report(InitialStateType.SUPERPOSITION, wf, target_grid, check_boundary_decay=False)

    def create_coherent_state(
        self,
        grid: SpatialGrid,
        alpha: complex,
        omega: float = 1.0,
        center: float = 0.0,
        enforce_dirichlet_boundaries: bool = False
    ) -> tuple:
        """Creates a Harmonic Oscillator coherent state and returns (Wavefunction, InitialStateReport)."""
        wf = CoherentStateGenerator.generate(
            grid=grid,
            alpha=alpha,
            mass=self.mass,
            omega=omega,
            hbar=self.hbar,
            center=center,
            enforce_dirichlet_boundaries=enforce_dirichlet_boundaries,
            integration_method=self.integration_method
        )
        return wf, self._build_report(InitialStateType.COHERENT, wf, grid, check_boundary_decay=True)

    def create_stationary_eigenstate(
        self,
        grid: SpatialGrid,
        potential: Potential,
        state_index: int = 0,
        accuracy_order: int = 2
    ) -> tuple:
        """Solves the TISE for `potential` on `grid` and extracts eigenstate `state_index`."""
        if not isinstance(state_index, (int, np.integer)) or state_index < 0:
            raise ValueError(f"state_index must be a non-negative integer. Received: {state_index}")

        opt = OptimizationEngine(
            OptimizationConfig(
                accuracy_order=accuracy_order,
                enable_cache=True,
                monitor_performance=False,
                run_stability_check=True
            )
        )
        res = opt.solve_system(
            grid=grid,
            potential=potential,
            num_eigenvalues=int(state_index) + 1,
            mass=self.mass,
            hbar=self.hbar,
            unit_system=potential.unit_system
        )
        wf = res.wavefunctions[int(state_index)]
        return wf, self._build_report(InitialStateType.STATIONARY_EIGENSTATE, wf, grid, check_boundary_decay=False)

    def create_custom(
        self,
        grid: SpatialGrid,
        wavefunction_input: Union[Callable[[np.ndarray], np.ndarray], np.ndarray],
        energy: float = 0.0,
        enforce_dirichlet_boundaries: bool = False
    ) -> tuple:
        """
        Creates a normalized Wavefunction from a user-defined callable f(x) -> psi(x)
        or a raw 1D complex/real NumPy array.
        """
        if callable(wavefunction_input):
            psi_raw = np.asarray(wavefunction_input(grid.grid), dtype=np.complex128)
        else:
            psi_raw = np.asarray(wavefunction_input, dtype=np.complex128).copy()

        if psi_raw.shape != (grid.points,):
            raise ValueError(f"Custom wavefunction shape {psi_raw.shape} must match grid {(grid.points,)}.")
        if not np.all(np.isfinite(psi_raw)):
            raise ValueError("Custom wavefunction contains NaN or Inf values.")

        if enforce_dirichlet_boundaries and grid.boundary_condition == BoundaryCondition.DIRICHLET:
            psi_raw[0] = 0.0 + 0.0j
            psi_raw[-1] = 0.0 + 0.0j

        psi_norm, _ = NormalizationEngine.normalize(psi_raw, grid.grid, method=self.integration_method)
        wf = Wavefunction(x=grid.grid, psi=psi_norm, energy=float(energy))
        return wf, self._build_report(InitialStateType.CUSTOM, wf, grid, check_boundary_decay=False)

    def generate(self, state_type: Union[InitialStateType, str], **kwargs: Any) -> Wavefunction:
        """Unified dispatcher returning the normalized Wavefunction for any InitialStateType."""
        kind = state_type if isinstance(state_type, InitialStateType) else InitialStateType(state_type.lower())
        if kind == InitialStateType.GAUSSIAN:
            return self.create_gaussian(**kwargs)[0]
        if kind == InitialStateType.PLANE_WAVE:
            return self.create_plane_wave(**kwargs)[0]
        if kind == InitialStateType.SUPERPOSITION:
            return self.create_superposition(**kwargs)[0]
        if kind == InitialStateType.COHERENT:
            return self.create_coherent_state(**kwargs)[0]
        if kind == InitialStateType.STATIONARY_EIGENSTATE:
            return self.create_stationary_eigenstate(**kwargs)[0]
        if kind == InitialStateType.CUSTOM:
            return self.create_custom(**kwargs)[0]
        raise ValueError(f"Unsupported initial state type: {state_type}")
