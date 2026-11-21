import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple
import numpy as np
from scipy.integrate import solve_ivp

from schrodinger.potentials.base import Potential
from schrodinger.potentials.barrier import BarrierPotential
from schrodinger.potentials.step import StepPotential
from schrodinger.potentials.models import FiniteSquareWell, FreeParticle
from schrodinger.potentials.poschl_teller import PoschlTellerPotential
from .transmission import TransmissionCalculator
from .reflection import ReflectionCalculator
from .transfer_matrix import TransferMatrixEngine
from .scattering_states import ScatteringState, ScatteringStateBuilder


@dataclass
class ScatteringConfig:
    """Configuration for 1D stationary quantum scattering calculations."""
    mass: float = 1.0
    hbar: float = 1.0
    method: Literal["auto", "transfer_matrix", "ode"] = "auto"
    num_slabs: int = 250
    ode_rtol: float = 1e-9
    ode_atol: float = 1e-11
    grid_points: int = 500
    asymptotic_padding: float = 2.0


class ScatteringEngine:
    """
    One-dimensional stationary quantum scattering engine.
    Supports exact Transfer Matrix solving for piecewise-constant potentials
    and overflow-safe backward Runge-Kutta integration for smooth/arbitrary potentials.
    """

    def __init__(self, config: ScatteringConfig = ScatteringConfig()):
        if config.mass <= 0 or config.hbar <= 0:
            raise ValueError("mass and hbar must be strictly positive.")
        if config.num_slabs < 1:
            raise ValueError("num_slabs must be >= 1.")
        if config.grid_points < 10:
            raise ValueError("grid_points must be >= 10.")
        if config.asymptotic_padding < 0:
            raise ValueError("asymptotic_padding must be >= 0.")

        self.config = config
        self.tmm_engine = TransferMatrixEngine(mass=config.mass, hbar=config.hbar)

    @staticmethod
    def infer_scattering_domain(
        potential: Potential,
        x_left: Optional[float] = None,
        x_right: Optional[float] = None,
        v_left: Optional[float] = None,
        v_right: Optional[float] = None
    ) -> Tuple[float, float, float, float, bool]:
        """
        Determines scattering interval [x_left, x_right], asymptotic potentials
        (v_left, v_right), and whether the potential is natively piecewise-constant.
        """
        is_piecewise = False

        if isinstance(potential, BarrierPotential):
            xl = potential.center - 0.5 * potential.width if x_left is None else x_left
            xr = potential.center + 0.5 * potential.width if x_right is None else x_right
            vl = potential.baseline if v_left is None else v_left
            vr = potential.baseline if v_right is None else v_right
            is_piecewise = True
        elif isinstance(potential, FiniteSquareWell):
            xl = potential.center - 0.5 * potential.width if x_left is None else x_left
            xr = potential.center + 0.5 * potential.width if x_right is None else x_right
            vl = 0.0 if v_left is None else v_left
            vr = 0.0 if v_right is None else v_right
            is_piecewise = True
        elif isinstance(potential, StepPotential):
            xl = potential.step_position - 1.0 if x_left is None else x_left
            xr = potential.step_position + 1.0 if x_right is None else x_right
            vl = potential.left_height if v_left is None else v_left
            vr = potential.right_height if v_right is None else v_right
            is_piecewise = True
        elif isinstance(potential, FreeParticle):
            xl = -1.0 if x_left is None else x_left
            xr = 1.0 if x_right is None else x_right
            vl = 0.0 if v_left is None else v_left
            vr = 0.0 if v_right is None else v_right
            is_piecewise = True
        elif isinstance(potential, PoschlTellerPotential):
            span = 12.0 / potential.alpha
            xl = potential.center - span if x_left is None else x_left
            xr = potential.center + span if x_right is None else x_right
            vl = 0.0 if v_left is None else v_left
            vr = 0.0 if v_right is None else v_right
        else:
            if x_left is None or x_right is None:
                raise ValueError(
                    f"Explicit `x_left` and `x_right` bounds are required for {potential.__class__.__name__}."
                )
            xl, xr = float(x_left), float(x_right)
            endpoints = potential(np.array([xl, xr], dtype=np.float64))
            vl = float(endpoints[0]) if v_left is None else float(v_left)
            vr = float(endpoints[1]) if v_right is None else float(v_right)

        if xr <= xl:
            raise ValueError(f"x_right ({xr}) must be strictly greater than x_left ({xl}).")
        if not (math.isfinite(vl) and math.isfinite(vr)):
            raise ValueError("Asymptotic potentials v_left and v_right must be finite.")

        return float(xl), float(xr), float(vl), float(vr), is_piecewise

    def _solve_backward_ode(
        self,
        potential: Potential,
        energy: float,
        x_left: float,
        x_right: float,
        v_left: float,
        v_right: float,
        x_grid: np.ndarray
    ) -> ScatteringState:
        """
        Integrates the stationary Schrödinger ODE backward from x_right to x_left
        with outgoing/decaying boundary conditions at x_right, then normalizes
        the incident wave at x_left to unit amplitude.
        """
        mass = self.config.mass
        hbar = self.config.hbar
        prefactor = 2.0 * mass / (hbar ** 2)

        k_left = math.sqrt(2.0 * mass * (energy - v_left)) / hbar
        k_right = TransmissionCalculator.wavevector(energy, v_right, mass=mass, hbar=hbar)

        def rhs(x_val: float, y: np.ndarray) -> np.ndarray:
            v_val = float(potential(x_val))
            return np.array([y[1], prefactor * (v_val - energy) * y[0]], dtype=np.complex128)

        # Boundary conditions at x = x_right: psi(x_R) = 1, psi'(x_R) = i * k_R
        y_right = np.array([1.0 + 0.0j, 1j * k_right], dtype=np.complex128)

        interior_mask = (x_grid >= x_left) & (x_grid <= x_right)
        x_eval_desc = np.unique(np.concatenate(([x_right], x_grid[interior_mask], [x_left])))[::-1]

        sol = solve_ivp(
            rhs,
            t_span=(x_right, x_left),
            y0=y_right,
            t_eval=x_eval_desc,
            method="DOP853",
            rtol=self.config.ode_rtol,
            atol=self.config.ode_atol
        )
        if not sol.success:
            raise RuntimeError(f"Backward scattering ODE integration failed: {sol.message}")

        # Values at x = x_left (last point in descending evaluation)
        psi_at_left = complex(sol.y[0, -1])
        dpsi_at_left = complex(sol.y[1, -1])

        # Match psi(x_L) = A + B, psi'(x_L) = i * k_L * (A - B)
        a_inc = 0.5 * (psi_at_left + dpsi_at_left / (1j * k_left))
        b_ref = 0.5 * (psi_at_left - dpsi_at_left / (1j * k_left))
        if abs(a_inc) == 0.0:
            raise ArithmeticError("Incident wave amplitude vanished during ODE normalization.")

        r_amp = complex(b_ref / a_inc)
        t_amp = complex(1.0 / a_inc)
        reflection = ReflectionCalculator.from_amplitude(r_amp)
        transmission = TransmissionCalculator.from_amplitude(
            t_amp, energy=energy, v_left=v_left, v_right=v_right, mass=mass, hbar=hbar
        )

        psi_inc, psi_ref, psi_trans = ScatteringStateBuilder.build_asymptotic_waves(
            x_grid, x_left, x_right, k_left, k_right, r_amp, t_amp
        )

        psi = np.zeros_like(x_grid, dtype=np.complex128)
        dpsi = np.zeros_like(x_grid, dtype=np.complex128)

        left_mask = x_grid < x_left
        if np.any(left_mask):
            psi[left_mask] = psi_inc[left_mask] + psi_ref[left_mask]
            dpsi[left_mask] = 1j * k_left * (psi_inc[left_mask] - psi_ref[left_mask])

        if np.any(interior_mask):
            # Reverse descending solver points back to ascending x
            x_asc = x_eval_desc[::-1]
            psi_asc = (sol.y[0, ::-1]) / a_inc
            dpsi_asc = (sol.y[1, ::-1]) / a_inc
            psi[interior_mask] = (
                np.interp(x_grid[interior_mask], x_asc, np.real(psi_asc))
                + 1j * np.interp(x_grid[interior_mask], x_asc, np.imag(psi_asc))
            )
            dpsi[interior_mask] = (
                np.interp(x_grid[interior_mask], x_asc, np.real(dpsi_asc))
                + 1j * np.interp(x_grid[interior_mask], x_asc, np.imag(dpsi_asc))
            )

        right_mask = x_grid > x_right
        if np.any(right_mask):
            psi[right_mask] = psi_trans[right_mask]
            dpsi[right_mask] = 1j * k_right * psi_trans[right_mask]

        return ScatteringState(
            x=x_grid,
            psi=psi,
            dpsi_dx=dpsi,
            incident_wave=psi_inc,
            reflected_wave=psi_ref,
            transmitted_wave=psi_trans,
            energy=float(energy),
            r_amplitude=r_amp,
            t_amplitude=t_amp,
            reflection=reflection,
            transmission=transmission,
            k_left=k_left,
            k_right=k_right,
            x_left=x_left,
            x_right=x_right,
            mass=mass,
            hbar=hbar
        )

    def solve_scattering(
        self,
        potential: Potential,
        energy: float,
        x_left: Optional[float] = None,
        x_right: Optional[float] = None,
        v_left: Optional[float] = None,
        v_right: Optional[float] = None,
        x_grid: Optional[np.ndarray] = None
    ) -> ScatteringState:
        """
        Solves the 1D stationary scattering problem for `potential` at incident
        energy `energy`, returning a complete ScatteringState.
        """
        xl, xr, vl, vr, is_piecewise = self.infer_scattering_domain(
            potential, x_left=x_left, x_right=x_right, v_left=v_left, v_right=v_right
        )
        if energy <= vl:
            raise ValueError(
                f"Incident energy ({energy}) must exceed left asymptotic potential ({vl})."
            )

        if x_grid is None:
            pad = self.config.asymptotic_padding
            x_arr = np.linspace(xl - pad, xr + pad, self.config.grid_points, dtype=np.float64)
        else:
            x_arr = np.asarray(x_grid, dtype=np.float64)
            if x_arr.ndim != 1 or len(x_arr) < 3:
                raise ValueError("x_grid must be a 1D array with at least 3 points.")

        method = self.config.method
        if method == "auto":
            method = "transfer_matrix" if is_piecewise else "ode"

        if method == "transfer_matrix":
            try:
                slabs, v_l_disc, v_r_disc = TransferMatrixEngine.discretize_potential(
                    potential, xl, xr, num_slabs=self.config.num_slabs
                )
                v_l_eff = vl if v_left is not None else v_l_disc
                v_r_eff = vr if v_right is not None else v_r_disc
                tmm_res = self.tmm_engine.solve(energy, slabs, v_left=v_l_eff, v_right=v_r_eff)
                if self.config.method == "auto" and not tmm_res.stability.is_stable:
                    return self._solve_backward_ode(potential, energy, xl, xr, vl, vr, x_arr)
                return ScatteringStateBuilder.from_slabs(
                    x=x_arr,
                    energy=energy,
                    slabs=slabs,
                    slab_amplitudes=tmm_res.slab_amplitudes,
                    r_amplitude=tmm_res.r_amplitude,
                    t_amplitude=tmm_res.t_amplitude,
                    reflection=tmm_res.reflection,
                    transmission=tmm_res.transmission,
                    v_left=v_l_eff,
                    v_right=v_r_eff,
                    mass=self.config.mass,
                    hbar=self.config.hbar
                )
            except OverflowError:
                if self.config.method == "transfer_matrix":
                    raise
                return self._solve_backward_ode(potential, energy, xl, xr, vl, vr, x_arr)

        if method == "ode":
            return self._solve_backward_ode(potential, energy, xl, xr, vl, vr, x_arr)

        raise ValueError(f"Unsupported scattering method: {method}")
