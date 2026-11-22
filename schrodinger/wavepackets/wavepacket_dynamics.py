from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
from schrodinger.numerical.grid import SpatialGrid
from schrodinger.potentials.base import Potential
from schrodinger.potentials.time_dependent import TimeDependentPotential
from schrodinger.wavefunctions.wavefunction import Wavefunction
from schrodinger.dynamics.time_integrators import IntegratorType
from schrodinger.dynamics.time_dependent_engine import (
    TimeDependentSchrodingerEngine,
    TimeDependentSolveResult
)
from .momentum_space import MomentumSpaceEngine
from .gaussian_wavepacket import GaussianWavepacketGenerator


@dataclass
class WavepacketDynamicsReport:
    """Comprehensive space-time and momentum-space wavepacket evolution diagnostics."""
    times: np.ndarray
    expectation_x: np.ndarray
    peak_positions: np.ndarray
    spatial_widths: np.ndarray
    spreading_ratios: np.ndarray
    analytical_free_widths: Optional[np.ndarray]
    expectation_p: np.ndarray
    fourier_expectation_p: np.ndarray
    momentum_widths: np.ndarray
    uncertainty_products: np.ndarray
    kinetic_energy: np.ndarray
    potential_energy: np.ndarray
    total_energy: np.ndarray
    probability_densities: np.ndarray
    momentum_densities: np.ndarray
    momentum_grid: np.ndarray
    norms: np.ndarray
    max_boundary_amplitude: float
    boundary_reflection_detected: bool
    solver_result: TimeDependentSolveResult


class WavepacketDynamicsAnalyzer:
    """
    Propagates a wavepacket through Milestone 14's TimeDependentSchrodingerEngine
    and extracts joint position-space and momentum-space spreading, trajectory,
    energy, and boundary reflection metrics.
    """

    def __init__(
        self,
        mass: float = 1.0,
        hbar: float = 1.0,
        accuracy_order: int = 2,
        boundary_warning_threshold: float = 1e-2
    ):
        if mass <= 0 or hbar <= 0:
            raise ValueError("mass and hbar must be strictly positive.")
        self.mass = float(mass)
        self.hbar = float(hbar)
        self.accuracy_order = int(accuracy_order)
        self.boundary_warning_threshold = float(boundary_warning_threshold)
        self.tdse_engine = TimeDependentSchrodingerEngine(
            mass=self.mass, hbar=self.hbar, accuracy_order=self.accuracy_order
        )

    def simulate(
        self,
        grid: SpatialGrid,
        potential: Union[Potential, TimeDependentPotential],
        initial_state: Wavefunction,
        t_end: float = 1.0,
        dt: float = 1e-3,
        t_start: float = 0.0,
        method: Union[IntegratorType, str] = IntegratorType.SPLIT_OPERATOR,
        save_every: int = 5,
        initial_gaussian_sigma: Optional[float] = None,
        norm_tolerance: float = 1e-3
    ) -> WavepacketDynamicsReport:
        """
        Runs time evolution and computes position, momentum, spreading, and energy trajectories.
        """
        td_res = self.tdse_engine.propagate(
            grid=grid,
            potential=potential,
            initial_state=initial_state,
            t_end=t_end,
            dt=dt,
            t_start=t_start,
            method=method,
            save_every=save_every,
            store_intermediate_states=True,
            compute_observables=True,
            norm_tolerance=norm_tolerance
        )

        if td_res.observables is None:
            raise RuntimeError("Observables trajectory was not populated during wavepacket propagation.")

        obs = td_res.observables
        evo = td_res.evolution
        x = grid.grid

        num_snapshots = len(evo.states)
        peak_positions = np.zeros(num_snapshots, dtype=np.float64)
        fourier_exp_p = np.zeros(num_snapshots, dtype=np.float64)
        momentum_widths = np.zeros(num_snapshots, dtype=np.float64)
        momentum_densities = np.zeros((num_snapshots, grid.points), dtype=np.float64)
        p_grid = np.zeros(grid.points, dtype=np.float64)
        max_edge_amp = 0.0

        for idx, psi_snap in enumerate(evo.states):
            dens = np.abs(psi_snap) ** 2
            peak_positions[idx] = float(x[int(np.argmax(dens))])

            # Monitor interior-adjacent edge amplitude for boundary collision/wrapping
            edge_amp = float(max(abs(psi_snap[1]), abs(psi_snap[-2]))) if grid.points > 4 else 0.0
            max_edge_amp = max(max_edge_amp, edge_amp)

            mom_rep = MomentumSpaceEngine.to_momentum_space(psi_snap, grid, hbar=self.hbar)
            if idx == 0:
                p_grid = mom_rep.p
            fourier_exp_p[idx] = mom_rep.expectation_p
            momentum_widths[idx] = mom_rep.uncertainty_p
            momentum_densities[idx, :] = mom_rep.p_probability_density

        widths = obs.uncertainty_x
        initial_width = float(widths[0]) if widths[0] > 0.0 else 1e-15
        spreading_ratios = widths / initial_width
        uncertainty_products = widths * momentum_widths

        ana_widths: Optional[np.ndarray] = None
        if initial_gaussian_sigma is not None:
            ana_widths = np.array(
                [
                    GaussianWavepacketGenerator.analytical_spreading_width(
                        initial_gaussian_sigma, float(t - t_start), mass=self.mass, hbar=self.hbar
                    )
                    for t in evo.times
                ],
                dtype=np.float64
            )

        return WavepacketDynamicsReport(
            times=evo.times,
            expectation_x=obs.expectation_x,
            peak_positions=peak_positions,
            spatial_widths=widths,
            spreading_ratios=spreading_ratios,
            analytical_free_widths=ana_widths,
            expectation_p=obs.expectation_p,
            fourier_expectation_p=fourier_exp_p,
            momentum_widths=momentum_widths,
            uncertainty_products=uncertainty_products,
            kinetic_energy=obs.kinetic_energy,
            potential_energy=obs.potential_energy,
            total_energy=obs.total_energy,
            probability_densities=obs.probability_densities,
            momentum_densities=momentum_densities,
            momentum_grid=p_grid,
            norms=evo.norms,
            max_boundary_amplitude=max_edge_amp,
            boundary_reflection_detected=(max_edge_amp > self.boundary_warning_threshold),
            solver_result=td_res
        )
