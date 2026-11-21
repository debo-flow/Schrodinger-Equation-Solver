from dataclasses import dataclass
from typing import List, Optional, Sequence
import numpy as np
from schrodinger.potentials.base import Potential
from schrodinger.potentials.barrier import BarrierPotential
from .transmission import TransmissionCalculator
from .scattering_engine import ScatteringEngine, ScatteringConfig
from .scattering_states import ScatteringState


@dataclass
class TunnelingResult:
    """Single-point quantum tunneling analysis result."""
    energy: float
    barrier_height: Optional[float]
    barrier_width: Optional[float]
    transmission: float
    reflection: float
    probability_sum: float
    conservation_verified: bool
    is_classically_forbidden: bool
    state: ScatteringState


@dataclass
class TunnelingSweepResult:
    """Parameter sweep output for energy, barrier width, or barrier height dependence."""
    parameter_name: str
    parameter_values: np.ndarray
    transmissions: np.ndarray
    reflections: np.ndarray
    conservation_errors: np.ndarray
    all_conserved: bool


class TunnelingEngine:
    """
    Analyzes quantum tunneling through finite potential barriers across incident
    energy, barrier width, and barrier height while enforcing R + T = 1.
    """

    def __init__(
        self,
        config: ScatteringConfig = ScatteringConfig(),
        conservation_tolerance: float = 1e-5
    ):
        if conservation_tolerance <= 0:
            raise ValueError("conservation_tolerance must be strictly positive.")
        self.config = config
        self.conservation_tolerance = float(conservation_tolerance)
        self.scattering_engine = ScatteringEngine(config)

    def analyze_tunneling(
        self,
        potential: Potential,
        energy: float,
        x_left: Optional[float] = None,
        x_right: Optional[float] = None,
        v_left: Optional[float] = None,
        v_right: Optional[float] = None
    ) -> TunnelingResult:
        """
        Computes tunneling transmission and reflection at incident energy `energy`
        and verifies probability conservation.
        """
        state = self.scattering_engine.solve_scattering(
            potential=potential,
            energy=energy,
            x_left=x_left,
            x_right=x_right,
            v_left=v_left,
            v_right=v_right
        )
        prob_sum = state.reflection + state.transmission
        conserved = TransmissionCalculator.verify_probability_conservation(
            state.reflection, state.transmission, tolerance=self.conservation_tolerance
        )

        if isinstance(potential, BarrierPotential):
            b_height: Optional[float] = potential.height
            b_width: Optional[float] = potential.width
            forbidden = energy < potential.height
        else:
            v_max = float(np.max(potential(state.x)))
            b_height = v_max
            b_width = float(state.x_right - state.x_left)
            forbidden = energy < v_max

        return TunnelingResult(
            energy=float(energy),
            barrier_height=b_height,
            barrier_width=b_width,
            transmission=state.transmission,
            reflection=state.reflection,
            probability_sum=float(prob_sum),
            conservation_verified=conserved,
            is_classically_forbidden=forbidden,
            state=state
        )

    def energy_dependence(
        self,
        potential: Potential,
        energies: Sequence[float],
        x_left: Optional[float] = None,
        x_right: Optional[float] = None,
        v_left: Optional[float] = None,
        v_right: Optional[float] = None
    ) -> TunnelingSweepResult:
        """Evaluates T(E) and R(E) across an array of incident energies."""
        e_arr = np.asarray(energies, dtype=np.float64)
        if e_arr.ndim != 1 or len(e_arr) == 0:
            raise ValueError("energies must be a non-empty 1D sequence.")

        transmissions: List[float] = []
        reflections: List[float] = []
        errors: List[float] = []

        for e_val in e_arr:
            res = self.analyze_tunneling(
                potential, float(e_val), x_left=x_left, x_right=x_right, v_left=v_left, v_right=v_right
            )
            transmissions.append(res.transmission)
            reflections.append(res.reflection)
            errors.append(abs(res.probability_sum - 1.0))

        err_arr = np.asarray(errors, dtype=np.float64)
        return TunnelingSweepResult(
            parameter_name="energy",
            parameter_values=e_arr,
            transmissions=np.asarray(transmissions, dtype=np.float64),
            reflections=np.asarray(reflections, dtype=np.float64),
            conservation_errors=err_arr,
            all_conserved=bool(np.all(err_arr <= self.conservation_tolerance))
        )

    def width_dependence(
        self,
        widths: Sequence[float],
        energy: float,
        barrier_height: float,
        center: float = 0.0,
        baseline: float = 0.0,
        unit_system: str = "dimensionless"
    ) -> TunnelingSweepResult:
        """Evaluates T(L) and R(L) across barrier widths L at fixed energy and height."""
        w_arr = np.asarray(widths, dtype=np.float64)
        if w_arr.ndim != 1 or len(w_arr) == 0 or np.any(w_arr <= 0):
            raise ValueError("widths must be a non-empty 1D sequence of strictly positive values.")

        transmissions: List[float] = []
        reflections: List[float] = []
        errors: List[float] = []

        for w in w_arr:
            barrier = BarrierPotential(
                height=barrier_height,
                width=float(w),
                center=center,
                baseline=baseline,
                unit_system=unit_system  # type: ignore[arg-type]
            )
            res = self.analyze_tunneling(barrier, energy=energy)
            transmissions.append(res.transmission)
            reflections.append(res.reflection)
            errors.append(abs(res.probability_sum - 1.0))

        err_arr = np.asarray(errors, dtype=np.float64)
        return TunnelingSweepResult(
            parameter_name="barrier_width",
            parameter_values=w_arr,
            transmissions=np.asarray(transmissions, dtype=np.float64),
            reflections=np.asarray(reflections, dtype=np.float64),
            conservation_errors=err_arr,
            all_conserved=bool(np.all(err_arr <= self.conservation_tolerance))
        )

    def height_dependence(
        self,
        heights: Sequence[float],
        energy: float,
        barrier_width: float,
        center: float = 0.0,
        baseline: float = 0.0,
        unit_system: str = "dimensionless"
    ) -> TunnelingSweepResult:
        """Evaluates T(V_0) and R(V_0) across barrier heights V_0 at fixed energy and width."""
        h_arr = np.asarray(heights, dtype=np.float64)
        if h_arr.ndim != 1 or len(h_arr) == 0 or np.any(h_arr <= baseline):
            raise ValueError("heights must be a non-empty 1D sequence strictly exceeding baseline.")

        transmissions: List[float] = []
        reflections: List[float] = []
        errors: List[float] = []

        for h in h_arr:
            barrier = BarrierPotential(
                height=float(h),
                width=barrier_width,
                center=center,
                baseline=baseline,
                unit_system=unit_system  # type: ignore[arg-type]
            )
            res = self.analyze_tunneling(barrier, energy=energy)
            transmissions.append(res.transmission)
            reflections.append(res.reflection)
            errors.append(abs(res.probability_sum - 1.0))

        err_arr = np.asarray(errors, dtype=np.float64)
        return TunnelingSweepResult(
            parameter_name="barrier_height",
            parameter_values=h_arr,
            transmissions=np.asarray(transmissions, dtype=np.float64),
            reflections=np.asarray(reflections, dtype=np.float64),
            conservation_errors=err_arr,
            all_conserved=bool(np.all(err_arr <= self.conservation_tolerance))
        )
