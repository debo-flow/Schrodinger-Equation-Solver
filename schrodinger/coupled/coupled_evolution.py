import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Literal, Optional, Union
import numpy as np
from scipy import integrate, sparse
from scipy.sparse.linalg import splu, spsolve

from schrodinger.numerical.grid import SpatialGrid, BoundaryCondition
from schrodinger.numerical.finite_difference import FiniteDifferenceEngine
from schrodinger.operators.kinetic_operator import KineticEnergyOperator
from schrodinger.operators.operator_validation import OperatorValidator
from .coupled_potential import CoupledPotential, TimeDependentCoupledPotential
from .multi_wavefunction import MultiComponentWavefunction
from .channel_analysis import ChannelTrajectoryReport, ChannelPopulationAnalyzer


@dataclass
class CoupledEvolutionConfig:
    """Configuration for multi-component time-dependent Schrödinger evolution."""
    t_start: float = 0.0
    t_end: float = 1.0
    dt: float = 1e-3
    method: Literal["crank_nicolson", "rk4"] = "crank_nicolson"
    accuracy_order: int = 2
    save_every: int = 1
    store_intermediate_states: bool = True
    norm_tolerance: float = 1e-3
    abort_on_instability: bool = True

    def __post_init__(self) -> None:
        if not (math.isfinite(self.t_start) and math.isfinite(self.t_end)):
            raise ValueError("t_start and t_end must be finite.")
        if self.t_end <= self.t_start:
            raise ValueError(f"t_end ({self.t_end}) must exceed t_start ({self.t_start}).")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError(f"Time step dt must be positive. Received: {self.dt}")
        if self.method not in ("crank_nicolson", "rk4"):
            raise ValueError(f"Unsupported coupled integrator method: {self.method}")
        if self.save_every < 1:
            raise ValueError("save_every must be >= 1.")
        if self.norm_tolerance <= 0.0:
            raise ValueError("norm_tolerance must be strictly positive.")


@dataclass
class CoupledEvolutionResult:
    """Output container for coupled-channel time propagation."""
    times: np.ndarray
    initial_state: MultiComponentWavefunction
    final_state: MultiComponentWavefunction
    states: List[MultiComponentWavefunction]
    norms: np.ndarray
    max_norm_deviation: float
    channel_trajectory: ChannelTrajectoryReport
    integrator_name: str
    num_steps: int
    stability_warnings: List[str]


class CoupledTimeIntegrator(ABC):
    """Abstract single-step propagator for N-channel coupled wavefunctions."""

    def __init__(
        self,
        grid: SpatialGrid,
        coupled_potential: Union[CoupledPotential, TimeDependentCoupledPotential],
        mass: float = 1.0,
        hbar: float = 1.0,
        accuracy_order: int = 2
    ):
        OperatorValidator.validate_physical_constants(mass, hbar)
        if isinstance(coupled_potential, TimeDependentCoupledPotential):
            self._td_potential = coupled_potential
            self._is_time_dependent = True
        elif isinstance(coupled_potential, CoupledPotential):
            self._td_potential = TimeDependentCoupledPotential.from_static(coupled_potential)
            self._is_time_dependent = False
        else:
            raise TypeError(
                "coupled_potential must be CoupledPotential or TimeDependentCoupledPotential."
            )

        self.grid = grid
        self.num_channels = self._td_potential.num_channels
        self.mass = float(mass)
        self.hbar = float(hbar)
        self.accuracy_order = int(accuracy_order)
        self._is_dirichlet = (grid.boundary_condition == BoundaryCondition.DIRICHLET)

        fd_engine = FiniteDifferenceEngine(grid)
        kin_single = KineticEnergyOperator(
            fd_engine, mass=self.mass, hbar=self.hbar
        ).build_matrix(accuracy_order=self.accuracy_order, interior_only=self._is_dirichlet)

        self._t_block = sparse.kron(
            sparse.eye(self.num_channels, format="csr", dtype=np.complex128),
            kin_single.astype(np.complex128),
            format="csr"
        )

    @property
    def is_time_dependent(self) -> bool:
        return self._is_time_dependent

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def _build_potential_block_at(self, t: float) -> sparse.csr_matrix:
        x_pts = self.grid.interior_points if self._is_dirichlet else self.grid.grid
        v_eval = self._td_potential.evaluate(x_pts, t)  # (N, N, M_act)
        if not np.all(np.isfinite(v_eval)):
            raise ValueError("Coupled potential contains non-finite entries on active nodes.")

        n = self.num_channels
        m_act = len(x_pts)
        blocks = [[None for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                diag_vals = v_eval[i, j, :].astype(np.complex128)
                if np.any(diag_vals != 0):
                    blocks[i][j] = sparse.diags(diag_vals, offsets=0, shape=(m_act, m_act), format="csr")
                else:
                    blocks[i][j] = sparse.csr_matrix((m_act, m_act), dtype=np.complex128)
        return sparse.bmat(blocks, format="csr")

    @abstractmethod
    def step(self, components: np.ndarray, t: float, dt: float) -> np.ndarray:
        """Advances 2D components array (N, M) from time t to t + dt."""
        pass


class CoupledCrankNicolsonIntegrator(CoupledTimeIntegrator):
    """
    Implicit unitary Crank-Nicolson propagator for coupled-channel systems:
        (I + i * dt / (2 * hbar) * H(t + dt/2)) Psi(t + dt)
            = (I - i * dt / (2 * hbar) * H(t + dt/2)) Psi(t)
    Caches sparse LU factorization for static coupled potentials.
    """

    def __init__(
        self,
        grid: SpatialGrid,
        coupled_potential: Union[CoupledPotential, TimeDependentCoupledPotential],
        mass: float = 1.0,
        hbar: float = 1.0,
        accuracy_order: int = 2
    ):
        super().__init__(grid, coupled_potential, mass=mass, hbar=hbar, accuracy_order=accuracy_order)
        n_tot = self._t_block.shape[0]
        self._identity = sparse.eye(n_tot, format="csc", dtype=np.complex128)
        self._static_v_block: Optional[sparse.csr_matrix] = (
            self._build_potential_block_at(0.0) if not self.is_time_dependent else None
        )
        self._cached_dt: Optional[float] = None
        self._cached_lu = None
        self._cached_rhs: Optional[sparse.csr_matrix] = None

    @property
    def name(self) -> str:
        return "CoupledCrankNicolson"

    def _build_operators(self, v_block: sparse.csr_matrix, dt: float):
        h_mat = (self._t_block + v_block).tocsc()
        alpha = 1j * dt / (2.0 * self.hbar)
        lhs = (self._identity + alpha * h_mat).tocsc()
        rhs = (self._identity - alpha * h_mat).tocsr()
        return lhs, rhs

    def step(self, components: np.ndarray, t: float, dt: float) -> np.ndarray:
        act = components[:, 1:-1] if self._is_dirichlet else components
        vec = act.reshape(-1)

        if self._static_v_block is not None:
            if self._cached_lu is None or self._cached_dt != dt:
                lhs, rhs = self._build_operators(self._static_v_block, dt)
                self._cached_lu = splu(lhs)
                self._cached_rhs = rhs
                self._cached_dt = dt
            b_vec = self._cached_rhs.dot(vec)  # type: ignore[union-attr]
            vec_next = self._cached_lu.solve(b_vec)
        else:
            v_mid = self._build_potential_block_at(t + 0.5 * dt)
            lhs, rhs = self._build_operators(v_mid, dt)
            b_vec = rhs.dot(vec)
            vec_next = spsolve(lhs, b_vec)

        next_act = vec_next.reshape(self.num_channels, -1)
        if self._is_dirichlet:
            out = np.zeros((self.num_channels, self.grid.points), dtype=np.complex128)
            out[:, 1:-1] = next_act
            return out
        return next_act.astype(np.complex128, copy=False)


class CoupledRK4Integrator(CoupledTimeIntegrator):
    """Explicit 4th-order Runge-Kutta propagator for coupled-channel systems."""

    def __init__(
        self,
        grid: SpatialGrid,
        coupled_potential: Union[CoupledPotential, TimeDependentCoupledPotential],
        mass: float = 1.0,
        hbar: float = 1.0,
        accuracy_order: int = 2
    ):
        super().__init__(grid, coupled_potential, mass=mass, hbar=hbar, accuracy_order=accuracy_order)
        self._static_v_block: Optional[sparse.csr_matrix] = (
            self._build_potential_block_at(0.0) if not self.is_time_dependent else None
        )

    @property
    def name(self) -> str:
        return "CoupledRK4"

    def _rhs(self, vec: np.ndarray, v_block: sparse.csr_matrix) -> np.ndarray:
        h_vec = self._t_block.dot(vec) + v_block.dot(vec)
        return (-1j / self.hbar) * h_vec

    def step(self, components: np.ndarray, t: float, dt: float) -> np.ndarray:
        act = components[:, 1:-1] if self._is_dirichlet else components
        vec = act.reshape(-1)

        if self._static_v_block is not None:
            v0 = v_mid = v1 = self._static_v_block
        else:
            v0 = self._build_potential_block_at(t)
            v_mid = self._build_potential_block_at(t + 0.5 * dt)
            v1 = self._build_potential_block_at(t + dt)

        k1 = self._rhs(vec, v0)
        k2 = self._rhs(vec + 0.5 * dt * k1, v_mid)
        k3 = self._rhs(vec + 0.5 * dt * k2, v_mid)
        k4 = self._rhs(vec + dt * k3, v1)

        vec_next = vec + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        next_act = vec_next.reshape(self.num_channels, -1)

        if self._is_dirichlet:
            out = np.zeros((self.num_channels, self.grid.points), dtype=np.complex128)
            out[:, 1:-1] = next_act
            return out
        return next_act.astype(np.complex128, copy=False)


class CoupledTimeEvolution:
    """
    Propagates an N-channel MultiComponentWavefunction through time while
    monitoring complete-state probability conservation and channel populations.
    """

    def __init__(
        self,
        grid: SpatialGrid,
        coupled_potential: Union[CoupledPotential, TimeDependentCoupledPotential],
        mass: float = 1.0,
        hbar: float = 1.0,
        config: CoupledEvolutionConfig = CoupledEvolutionConfig()
    ):
        self.grid = grid
        self.coupled_potential = coupled_potential
        self.mass = float(mass)
        self.hbar = float(hbar)
        self.config = config

        if config.method == "crank_nicolson":
            self.integrator: CoupledTimeIntegrator = CoupledCrankNicolsonIntegrator(
                grid, coupled_potential, mass=self.mass, hbar=self.hbar, accuracy_order=config.accuracy_order
            )
        else:
            self.integrator = CoupledRK4Integrator(
                grid, coupled_potential, mass=self.mass, hbar=self.hbar, accuracy_order=config.accuracy_order
            )

    def propagate(self, initial_state: MultiComponentWavefunction) -> CoupledEvolutionResult:
        """Executes multi-component time evolution from t_start to t_end."""
        if initial_state.num_channels != self.integrator.num_channels:
            raise ValueError(
                f"Initial state channels ({initial_state.num_channels}) != "
                f"potential channels ({self.integrator.num_channels})."
            )
        if initial_state.x.shape != self.grid.grid.shape or not np.allclose(initial_state.x, self.grid.grid):
            raise ValueError("Initial state spatial grid must match CoupledTimeEvolution grid.")

        x = self.grid.grid
        total_duration = self.config.t_end - self.config.t_start
        num_steps = max(1, int(math.ceil(total_duration / self.config.dt - 1e-12)))
        dt_actual = total_duration / num_steps

        curr_comp = initial_state.components.astype(np.complex128, copy=True)
        init_integral = float(integrate.simpson(y=np.sum(np.abs(curr_comp) ** 2, axis=0), x=x))

        saved_times: List[float] = [float(self.config.t_start)]
        saved_states: List[MultiComponentWavefunction] = [initial_state]
        saved_norms: List[float] = [init_integral]
        warnings: List[str] = []

        t_curr = float(self.config.t_start)
        for step_idx in range(1, num_steps + 1):
            curr_comp = self.integrator.step(curr_comp, t_curr, dt_actual)
            t_curr = self.config.t_start + step_idx * dt_actual

            if not np.all(np.isfinite(curr_comp)):
                raise RuntimeError(
                    f"Numerical instability: NaN/Inf encountered at step {step_idx} (t={t_curr:.6f})."
                )

            is_final = (step_idx == num_steps)
            if (step_idx % self.config.save_every == 0) or is_final:
                norm_val = float(integrate.simpson(y=np.sum(np.abs(curr_comp) ** 2, axis=0), x=x))
                if not math.isfinite(norm_val) or abs(norm_val - 1.0) > self.config.norm_tolerance:
                    msg = f"Coupled probability conservation deviation at t={t_curr:.6f}: integral={norm_val:.6f}."
                    if self.config.abort_on_instability:
                        raise RuntimeError(msg)
                    warnings.append(msg)

                saved_times.append(float(t_curr))
                saved_norms.append(norm_val)

                if self.config.store_intermediate_states or is_final:
                    snap_state = MultiComponentWavefunction(
                        x=x,
                        components=curr_comp,
                        energy=initial_state.energy,
                        channel_labels=initial_state.channel_labels,
                        normalize=True
                    )
                    saved_states.append(snap_state)

        final_state = saved_states[-1]
        if not self.config.store_intermediate_states:
            saved_states = [initial_state, final_state]

        trajectory = ChannelPopulationAnalyzer.analyze_trajectory(
            saved_times if self.config.store_intermediate_states else [saved_times[0], saved_times[-1]],
            saved_states,
            tolerance=self.config.norm_tolerance
        )
        max_norm_dev = float(np.max(np.abs(np.asarray(saved_norms) - 1.0)))

        return CoupledEvolutionResult(
            times=np.asarray(saved_times, dtype=np.float64),
            initial_state=initial_state,
            final_state=final_state,
            states=saved_states,
            norms=np.asarray(saved_norms, dtype=np.float64),
            max_norm_deviation=max_norm_dev,
            channel_trajectory=trajectory,
            integrator_name=self.integrator.name,
            num_steps=num_steps,
            stability_warnings=warnings
        )
