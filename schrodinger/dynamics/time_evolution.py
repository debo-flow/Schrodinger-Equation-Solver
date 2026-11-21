import math
from dataclasses import dataclass
from typing import List
import numpy as np
from scipy import integrate
from schrodinger.numerical.grid import SpatialGrid
from schrodinger.wavefunctions.wavefunction import Wavefunction
from schrodinger.wavefunctions.normalization import NormalizationEngine, IntegrationMethod
from .time_integrators import TimeIntegrator


@dataclass
class TimeEvolutionConfig:
    """Execution parameters for numerical time propagation."""
    t_start: float = 0.0
    t_end: float = 1.0
    dt: float = 1e-3
    save_every: int = 1
    store_intermediate_states: bool = True
    norm_tolerance: float = 1e-2
    abort_on_instability: bool = True

    def __post_init__(self) -> None:
        if not (math.isfinite(self.t_start) and math.isfinite(self.t_end)):
            raise ValueError("t_start and t_end must be finite.")
        if self.t_end <= self.t_start:
            raise ValueError(f"t_end ({self.t_end}) must be strictly greater than t_start ({self.t_start}).")
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError(f"Time step dt must be finite and strictly positive. Received: {self.dt}")
        if self.save_every < 1:
            raise ValueError("save_every must be >= 1.")
        if self.norm_tolerance <= 0:
            raise ValueError("norm_tolerance must be strictly positive.")


@dataclass
class TimeEvolutionResult:
    """Container holding initial, intermediate, and final propagated states."""
    grid: SpatialGrid
    times: np.ndarray
    initial_state: Wavefunction
    final_state: Wavefunction
    states: List[np.ndarray]
    norms: np.ndarray
    integrator_name: str
    num_steps: int
    stability_warnings: List[str]


class TimeEvolutionEngine:
    """
    Propagates a quantum state psi(x, t_start) -> psi(x, t_end) using a configured
    TimeIntegrator while monitoring probability norm and numerical stability.
    """

    def __init__(self, integrator: TimeIntegrator, config: TimeEvolutionConfig = TimeEvolutionConfig()):
        self.integrator = integrator
        self.config = config

    def propagate(self, initial_state: Wavefunction) -> TimeEvolutionResult:
        """Executes time propagation and returns stored states and norm diagnostics."""
        grid = self.integrator.grid
        if initial_state.x.shape != grid.grid.shape or not np.allclose(initial_state.x, grid.grid):
            raise ValueError("initial_state spatial grid must match the integrator's SpatialGrid.")

        warnings = self.integrator.validate_stability(self.config.dt)
        if warnings and self.config.abort_on_instability and self.integrator.metadata.name == "RK4":
            raise RuntimeError(f"Aborted unstable integration: {warnings[0]}")

        total_duration = self.config.t_end - self.config.t_start
        num_steps = max(1, int(math.ceil(total_duration / self.config.dt - 1e-12)))
        dt_actual = total_duration / num_steps

        x = grid.grid
        psi_curr = initial_state.psi.astype(np.complex128, copy=True)
        norm_init = float(integrate.simpson(y=np.abs(psi_curr) ** 2, x=x))

        saved_times: List[float] = [float(self.config.t_start)]
        saved_states: List[np.ndarray] = [psi_curr.copy()] if self.config.store_intermediate_states else []
        saved_norms: List[float] = [norm_init]

        t_curr = float(self.config.t_start)
        for step_idx in range(1, num_steps + 1):
            psi_curr = self.integrator.step(psi_curr, t_curr, dt_actual)
            t_curr = self.config.t_start + step_idx * dt_actual

            if not np.all(np.isfinite(psi_curr)):
                raise RuntimeError(f"Numerical instability: NaN/Inf encountered at step {step_idx} (t={t_curr:.6f}).")

            is_final = (step_idx == num_steps)
            if (step_idx % self.config.save_every == 0) or is_final:
                norm_val = float(integrate.simpson(y=np.abs(psi_curr) ** 2, x=x))
                if not math.isfinite(norm_val) or abs(norm_val - 1.0) > self.config.norm_tolerance:
                    msg = f"Norm conservation failure at t={t_curr:.6f}: norm={norm_val:.6f}."
                    if self.config.abort_on_instability:
                        raise RuntimeError(msg)
                    warnings.append(msg)

                saved_times.append(float(t_curr))
                saved_norms.append(norm_val)
                if self.config.store_intermediate_states or is_final:
                    saved_states.append(psi_curr.copy())

        if not self.config.store_intermediate_states:
            # Ensure initial and final states are still present in `states`
            saved_states = [initial_state.psi.copy(), psi_curr.copy()]

        # Normalize a clean copy for the Wavefunction object while keeping raw trajectory un-renormalized
        psi_final_normed, _ = NormalizationEngine.normalize(psi_curr, x, method=IntegrationMethod.SIMPSON)
        final_wf = Wavefunction(x=x, psi=psi_final_normed, energy=initial_state.energy)

        return TimeEvolutionResult(
            grid=grid,
            times=np.asarray(saved_times, dtype=np.float64),
            initial_state=initial_state,
            final_state=final_wf,
            states=saved_states,
            norms=np.asarray(saved_norms, dtype=np.float64),
            integrator_name=self.integrator.metadata.name,
            num_steps=num_steps,
            stability_warnings=warnings
        )
