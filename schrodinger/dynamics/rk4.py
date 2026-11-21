import math
from typing import List, Optional, Union
import numpy as np
from schrodinger.numerical.grid import SpatialGrid, BoundaryCondition
from schrodinger.numerical.finite_difference import FiniteDifferenceEngine
from schrodinger.potentials.base import Potential
from schrodinger.potentials.time_dependent import TimeDependentPotential
from schrodinger.operators.kinetic_operator import KineticEnergyOperator
from .time_integrators import TimeIntegrator, IntegratorMetadata


class RK4Integrator(TimeIntegrator):
    """
    Explicit Fourth-Order Runge-Kutta (RK4) integrator for the 1D TDSE:
        d(psi)/dt = -(i / hbar) * H(t) * psi(t)

    Assumptions & Constraints:
        - Explicit 4th-order temporal accuracy O(dt^4).
        - Conditionally stable: requires dt * ||H||_2 / hbar <= 2 * sqrt(2) (~2.828).
        - Nearly norm-preserving (O(dt^5) local norm error), not strictly unitary.
        - Supports Dirichlet and Neumann grid boundary conditions.
    """

    def __init__(
        self,
        grid: SpatialGrid,
        potential: Union[Potential, TimeDependentPotential],
        mass: float = 1.0,
        hbar: float = 1.0,
        accuracy_order: int = 2
    ):
        super().__init__(grid=grid, potential=potential, mass=mass, hbar=hbar)
        self.accuracy_order = int(accuracy_order)
        self._is_dirichlet = (grid.boundary_condition == BoundaryCondition.DIRICHLET)

        fd_engine = FiniteDifferenceEngine(grid)
        kin_op = KineticEnergyOperator(fd_engine, mass=self.mass, hbar=self.hbar)
        self._t_mat = kin_op.build_matrix(
            accuracy_order=self.accuracy_order,
            interior_only=self._is_dirichlet
        )

        self._static_v: Optional[np.ndarray] = None
        if not self.is_time_dependent:
            self._static_v = self._eval_potential_slice(0.0)

    @property
    def metadata(self) -> IntegratorMetadata:
        return IntegratorMetadata(
            name="RK4",
            temporal_order=4,
            is_unconditionally_unitary=False,
            supports_time_dependent_potential=True,
            boundary_assumption=f"Finite difference ({self.grid.boundary_condition.name})",
            constraints=[
                "Conditionally stable: dt * lambda_max(H) / hbar <= 2 * sqrt(2) (~2.828).",
                "Explicit time stepping requires dt scaling as O(dx^2) on fine spatial grids.",
                "Requires finite potential values across active grid nodes."
            ]
        )

    def _eval_potential_slice(self, t: float) -> np.ndarray:
        x_pts = self.grid.interior_points if self._is_dirichlet else self.grid.grid
        v = np.asarray(self._td_potential(x_pts, t), dtype=np.float64)
        if not np.all(np.isfinite(v)):
            raise ValueError("RK4Integrator requires finite potential values on active grid points.")
        return v

    def validate_stability(self, dt: float) -> List[str]:
        warnings = super().validate_stability(dt)
        v0 = self._static_v if self._static_v is not None else self._eval_potential_slice(0.0)
        # Gershgorin upper bound on spectral radius of H
        kinetic_bound = (2.0 * (self.hbar ** 2)) / (self.mass * (self.grid.dx ** 2))
        if self.accuracy_order == 4:
            kinetic_bound *= (16.0 / 12.0)
        h_max = kinetic_bound + float(np.max(np.abs(v0)))
        cfl_number = (h_max * dt) / self.hbar
        if cfl_number > 2.0 * math.sqrt(2.0):
            warnings.append(
                f"RK4 stability limit exceeded: (||H|| * dt / hbar) = {cfl_number:.2f} > 2.83. "
                "Reduce dt or use Crank-Nicolson / Split-Operator."
            )
        return warnings

    def _rhs(self, psi_active: np.ndarray, v_active: np.ndarray) -> np.ndarray:
        h_psi = self._t_mat.dot(psi_active) + v_active * psi_active
        return (-1j / self.hbar) * h_psi

    def step(self, psi: np.ndarray, t: float, dt: float) -> np.ndarray:
        if dt <= 0:
            raise ValueError("dt must be strictly positive.")

        if self._is_dirichlet:
            psi_act = psi[1:-1]
        else:
            psi_act = psi

        if self._static_v is not None:
            v0 = v_mid = v1 = self._static_v
        else:
            v0 = self._eval_potential_slice(t)
            v_mid = self._eval_potential_slice(t + 0.5 * dt)
            v1 = self._eval_potential_slice(t + dt)

        k1 = self._rhs(psi_act, v0)
        k2 = self._rhs(psi_act + 0.5 * dt * k1, v_mid)
        k3 = self._rhs(psi_act + 0.5 * dt * k2, v_mid)
        k4 = self._rhs(psi_act + dt * k3, v1)

        psi_next_act = psi_act + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if self._is_dirichlet:
            out = np.zeros(self.grid.points, dtype=np.complex128)
            out[1:-1] = psi_next_act
            return out
        return psi_next_act.astype(np.complex128, copy=False)
