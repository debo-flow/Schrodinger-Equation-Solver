from dataclasses import dataclass
from typing import Literal, Optional, Union
from schrodinger.numerical.grid import SpatialGrid
from schrodinger.potentials.base import Potential
from schrodinger.potentials.time_dependent import TimeDependentPotential
from schrodinger.wavefunctions.wavefunction import Wavefunction
from .time_integrators import IntegratorType, TimeIntegrator
from .rk4 import RK4Integrator
from .crank_nicolson import CrankNicolsonIntegrator
from .split_operator import SplitOperatorIntegrator
from .time_evolution import TimeEvolutionConfig, TimeEvolutionResult, TimeEvolutionEngine
from .time_observables import TimeObservablesSeries, TimeObservablesAnalyzer
from .evolution_validation import EvolutionValidationReport, EvolutionValidator


@dataclass
class TimeDependentSolveResult:
    """Complete output package from TimeDependentSchrodingerEngine."""
    evolution: TimeEvolutionResult
    observables: Optional[TimeObservablesSeries]
    validation: EvolutionValidationReport


class TimeDependentSchrodingerEngine:
    """
    High-level orchestrator for the 1D Time-Dependent Schrödinger Equation.
    Binds SpatialGrid, static or time-dependent Potentials, InitialState,
    TimeIntegrator selection, dynamics observables, and validation reporting.
    """

    def __init__(
        self,
        mass: float = 1.0,
        hbar: float = 1.0,
        accuracy_order: int = 2
    ):
        if mass <= 0 or hbar <= 0:
            raise ValueError("mass and hbar must be strictly positive.")
        self.mass = float(mass)
        self.hbar = float(hbar)
        self.accuracy_order = int(accuracy_order)

    def create_integrator(
        self,
        method: Union[IntegratorType, str],
        grid: SpatialGrid,
        potential: Union[Potential, TimeDependentPotential],
        accuracy_order: Optional[int] = None
    ) -> TimeIntegrator:
        """Factory instantiating the requested numerical time integrator."""
        method_enum = method if isinstance(method, IntegratorType) else IntegratorType(method.lower())
        order = accuracy_order if accuracy_order is not None else self.accuracy_order

        if method_enum == IntegratorType.RK4:
            return RK4Integrator(grid, potential, mass=self.mass, hbar=self.hbar, accuracy_order=order)
        if method_enum == IntegratorType.CRANK_NICOLSON:
            return CrankNicolsonIntegrator(grid, potential, mass=self.mass, hbar=self.hbar, accuracy_order=order)
        if method_enum == IntegratorType.SPLIT_OPERATOR:
            return SplitOperatorIntegrator(grid, potential, mass=self.mass, hbar=self.hbar)
        raise ValueError(f"Unsupported integrator method: {method}")

    def propagate(
        self,
        grid: SpatialGrid,
        potential: Union[Potential, TimeDependentPotential],
        initial_state: Wavefunction,
        t_end: float = 1.0,
        dt: float = 1e-3,
        t_start: float = 0.0,
        method: Union[IntegratorType, str] = IntegratorType.CRANK_NICOLSON,
        save_every: int = 1,
        store_intermediate_states: bool = True,
        compute_observables: bool = True,
        norm_tolerance: float = 1e-3,
        abort_on_instability: bool = True
    ) -> TimeDependentSolveResult:
        """
        Propagates `initial_state` from `t_start` to `t_end` and returns the
        propagated trajectory, time-dependent observables, and validation report.
        """
        integrator = self.create_integrator(method, grid, potential)
        evo_config = TimeEvolutionConfig(
            t_start=t_start,
            t_end=t_end,
            dt=dt,
            save_every=save_every,
            store_intermediate_states=store_intermediate_states,
            norm_tolerance=norm_tolerance,
            abort_on_instability=abort_on_instability
        )
        evo_engine = TimeEvolutionEngine(integrator, evo_config)
        evo_result = evo_engine.propagate(initial_state)

        obs_series: Optional[TimeObservablesSeries] = None
        if compute_observables and store_intermediate_states:
            obs_analyzer = TimeObservablesAnalyzer(
                grid, potential, mass=self.mass, hbar=self.hbar, accuracy_order=self.accuracy_order
            )
            obs_series = obs_analyzer.analyze_trajectory(evo_result.times, evo_result.states)

        validation_report = EvolutionValidator.validate_run(
            integrator_name=evo_result.integrator_name,
            num_steps=evo_result.num_steps,
            norms=evo_result.norms,
            final_psi=evo_result.final_state.psi,
            is_time_independent=(not integrator.is_time_dependent),
            energies=(obs_series.total_energy if obs_series is not None else None),
            norm_tolerance=norm_tolerance,
            existing_warnings=evo_result.stability_warnings
        )

        return TimeDependentSolveResult(
            evolution=evo_result,
            observables=obs_series,
            validation=validation_report
        )
