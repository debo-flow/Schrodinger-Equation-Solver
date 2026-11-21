from typing import Optional, Union
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu, spsolve
from schrodinger.numerical.grid import SpatialGrid, BoundaryCondition
from schrodinger.numerical.finite_difference import FiniteDifferenceEngine
from schrodinger.potentials.base import Potential
from schrodinger.potentials.time_dependent import TimeDependentPotential
from schrodinger.operators.kinetic_operator import KineticEnergyOperator
from .time_integrators import TimeIntegrator, IntegratorMetadata


class CrankNicolsonIntegrator(TimeIntegrator):
    """
    Implicit Second-Order Unitary Crank-Nicolson integrator:
        (I + i * dt / (2 * hbar) * H(t + dt/2)) * psi(t + dt)
            = (I - i * dt / (2 * hbar) * H(t + dt/2)) * psi(t)

    Assumptions & Constraints:
        - Second-order accurate in time O(dt^2); spatial order matches FiniteDifferenceEngine.
        - Unconditionally stable and strictly unitary (norm-conserving) for Hermitian H.
        - Caches sparse LU factorization when potential is time-independent and dt is constant.
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
        ).astype(np.complex128)

        n_act = self._t_mat.shape[0]
        self._identity = sparse.eye(n_act, format="csc", dtype=np.complex128)

        self._static_v: Optional[np.ndarray] = None
        if not self.is_time_dependent:
            self._static_v = self._eval_potential_slice(0.0)

        self._cached_dt: Optional[float] = None
        self._cached_lu = None
        self._cached_rhs_mat: Optional[sparse.csr_matrix] = None

    @property
    def metadata(self) -> IntegratorMetadata:
        return IntegratorMetadata(
            name="Crank-Nicolson",
            temporal_order=2,
            is_unconditionally_unitary=True,
            supports_time_dependent_potential=True,
            boundary_assumption=f"Finite difference ({self.grid.boundary_condition.name})",
            constraints=[
                "Unconditionally stable for all dt > 0; accuracy scales as O(dt^2 + dx^p).",
                "Requires finite potential values across active grid nodes."
            ]
        )

    def _eval_potential_slice(self, t: float) -> np.ndarray:
        x_pts = self.grid.interior_points if self._is_dirichlet else self.grid.grid
        v = np.asarray(self._td_potential(x_pts, t), dtype=np.float64)
        if not np.all(np.isfinite(v)):
            raise ValueError("CrankNicolsonIntegrator requires finite potential values on active grid points.")
        return v

    def _build_cn_operators(self, v_active: np.ndarray, dt: float):
        v_mat = sparse.diags(v_active.astype(np.complex128), offsets=0, format="csc")
        h_mat = self._t_mat.tocsc() + v_mat
        alpha = 1j * dt / (2.0 * self.hbar)
        lhs = (self._identity + alpha * h_mat).tocsc()
        rhs = (self._identity - alpha * h_mat).tocsr()
        return lhs, rhs

    def step(self, psi: np.ndarray, t: float, dt: float) -> np.ndarray:
        if dt <= 0:
            raise ValueError("dt must be strictly positive.")

        psi_act = psi[1:-1] if self._is_dirichlet else psi

        if self._static_v is not None:
            if self._cached_lu is None or self._cached_dt != dt:
                lhs, rhs = self._build_cn_operators(self._static_v, dt)
                self._cached_lu = splu(lhs)
                self._cached_rhs_mat = rhs
                self._cached_dt = dt
            b_vec = self._cached_rhs_mat.dot(psi_act)  # type: ignore[union-attr]
            psi_next_act = self._cached_lu.solve(b_vec)
        else:
            v_mid = self._eval_potential_slice(t + 0.5 * dt)
            lhs, rhs = self._build_cn_operators(v_mid, dt)
            b_vec = rhs.dot(psi_act)
            psi_next_act = spsolve(lhs, b_vec)

        if self._is_dirichlet:
            out = np.zeros(self.grid.points, dtype=np.complex128)
            out[1:-1] = psi_next_act
            return out
        return np.asarray(psi_next_act, dtype=np.complex128)
