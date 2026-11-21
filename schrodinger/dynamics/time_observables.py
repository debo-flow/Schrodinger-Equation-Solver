from dataclasses import dataclass
from typing import List, Sequence, Union
import numpy as np
from scipy import integrate
from schrodinger.numerical.grid import SpatialGrid
from schrodinger.numerical.finite_difference import FiniteDifferenceEngine
from schrodinger.potentials.base import Potential
from schrodinger.potentials.time_dependent import TimeDependentPotential, StaticPotentialWrapper
from schrodinger.operators.kinetic_operator import KineticEnergyOperator
from schrodinger.observables.momentum_operator import MomentumOperator


@dataclass
class DynamicsSnapshot:
    """Instantaneous physical observables of psi(x, t) at time t."""
    time: float
    norm: float
    expectation_x: float
    uncertainty_x: float
    expectation_p: float
    uncertainty_p: float
    kinetic_energy: float
    potential_energy: float
    total_energy: float
    probability_density: np.ndarray


@dataclass
class TimeObservablesSeries:
    """Time-series trajectories of quantum observables across a simulation."""
    times: np.ndarray
    norms: np.ndarray
    norm_errors: np.ndarray
    expectation_x: np.ndarray
    uncertainty_x: np.ndarray
    expectation_p: np.ndarray
    uncertainty_p: np.ndarray
    kinetic_energy: np.ndarray
    potential_energy: np.ndarray
    total_energy: np.ndarray
    probability_densities: np.ndarray


class TimeObservablesAnalyzer:
    """
    Computes time-dependent probability densities and expectation values
    <x(t)>, <p(t)>, <T(t)>, <V(t)>, and <E(t)> from propagated wavefunctions.
    """

    def __init__(
        self,
        grid: SpatialGrid,
        potential: Union[Potential, TimeDependentPotential],
        mass: float = 1.0,
        hbar: float = 1.0,
        accuracy_order: int = 2
    ):
        if mass <= 0 or hbar <= 0:
            raise ValueError("mass and hbar must be strictly positive.")
        if isinstance(potential, TimeDependentPotential):
            self._td_potential = potential
        elif isinstance(potential, Potential):
            self._td_potential = StaticPotentialWrapper(potential)
        else:
            raise TypeError("potential must be Potential or TimeDependentPotential.")

        self.grid = grid
        self.mass = float(mass)
        self.hbar = float(hbar)

        fd_engine = FiniteDifferenceEngine(grid)
        self._p_op = MomentumOperator.build_p(fd_engine, hbar=self.hbar, accuracy_order=accuracy_order)
        self._t_op = KineticEnergyOperator(fd_engine, mass=self.mass, hbar=self.hbar).build_matrix(
            accuracy_order=accuracy_order, interior_only=False
        )

    def analyze_snapshot(self, psi: np.ndarray, t: float) -> DynamicsSnapshot:
        """Evaluates all physical observables for state `psi` at instant `t`."""
        x = self.grid.grid
        psi_c = np.asarray(psi, dtype=np.complex128)
        if psi_c.shape != x.shape:
            raise ValueError(f"Wavefunction shape {psi_c.shape} does not match grid {x.shape}.")

        density = np.abs(psi_c) ** 2
        norm = float(integrate.simpson(y=density, x=x))
        if norm <= 0.0 or not np.isfinite(norm):
            raise ValueError(f"Invalid wavefunction norm ({norm}) at t={t}.")

        # Position expectation and uncertainty
        exp_x = float(integrate.simpson(y=x * density, x=x) / norm)
        exp_x2 = float(integrate.simpson(y=(x ** 2) * density, x=x) / norm)
        var_x = max(0.0, exp_x2 - exp_x ** 2)
        delta_x = float(np.sqrt(var_x))

        # Momentum expectation and uncertainty
        p_psi = self._p_op.dot(psi_c)
        p2_psi = self._p_op.dot(p_psi)
        exp_p = float(np.real(integrate.simpson(y=np.conj(psi_c) * p_psi, x=x)) / norm)
        exp_p2 = float(np.real(integrate.simpson(y=np.conj(psi_c) * p2_psi, x=x)) / norm)
        var_p = max(0.0, exp_p2 - exp_p ** 2)
        delta_p = float(np.sqrt(var_p))

        # Kinetic, potential, and total energy expectations
        t_psi = self._t_op.dot(psi_c)
        kin_e = float(np.real(integrate.simpson(y=np.conj(psi_c) * t_psi, x=x)) / norm)

        v_vals = np.asarray(self._td_potential(x, t), dtype=np.float64)
        pot_e = float(integrate.simpson(y=v_vals * density, x=x) / norm)
        tot_e = kin_e + pot_e

        return DynamicsSnapshot(
            time=float(t),
            norm=norm,
            expectation_x=exp_x,
            uncertainty_x=delta_x,
            expectation_p=exp_p,
            uncertainty_p=delta_p,
            kinetic_energy=kin_e,
            potential_energy=pot_e,
            total_energy=tot_e,
            probability_density=density
        )

    def analyze_trajectory(
        self,
        times: Sequence[float],
        states: Sequence[np.ndarray]
    ) -> TimeObservablesSeries:
        """Computes observable trajectories across a sequence of time snapshots."""
        if len(times) != len(states):
            raise ValueError("times and states must have identical lengths.")
        if len(times) == 0:
            raise ValueError("Cannot analyze an empty trajectory.")

        snapshots: List[DynamicsSnapshot] = [
            self.analyze_snapshot(psi, float(t)) for t, psi in zip(times, states)
        ]

        t_arr = np.array([s.time for s in snapshots], dtype=np.float64)
        norms = np.array([s.norm for s in snapshots], dtype=np.float64)
        return TimeObservablesSeries(
            times=t_arr,
            norms=norms,
            norm_errors=np.abs(norms - 1.0),
            expectation_x=np.array([s.expectation_x for s in snapshots], dtype=np.float64),
            uncertainty_x=np.array([s.uncertainty_x for s in snapshots], dtype=np.float64),
            expectation_p=np.array([s.expectation_p for s in snapshots], dtype=np.float64),
            uncertainty_p=np.array([s.uncertainty_p for s in snapshots], dtype=np.float64),
            kinetic_energy=np.array([s.kinetic_energy for s in snapshots], dtype=np.float64),
            potential_energy=np.array([s.potential_energy for s in snapshots], dtype=np.float64),
            total_energy=np.array([s.total_energy for s in snapshots], dtype=np.float64),
            probability_densities=np.vstack([s.probability_density for s in snapshots])
        )
