from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union
from schrodinger.numerical.grid import SpatialGrid
from schrodinger.solvers.solver_configuration import SolverConfig
from .coupled_potential import (
    CouplingTerm,
    CoupledPotential,
    TimeDependentCoupledPotential
)
from .multi_wavefunction import MultiComponentWavefunction
from .coupled_hamiltonian import CoupledHamiltonian
from .coupled_solver import CoupledSolverResult, CoupledEigenvalueSolver
from .channel_analysis import ChannelPopulationReport, ChannelPopulationAnalyzer
from .coupled_evolution import (
    CoupledEvolutionConfig,
    CoupledEvolutionResult,
    CoupledTimeEvolution
)
from .coupled_validation import CoupledValidationReport, CoupledValidationEngine


@dataclass
class CoupledSystemWorkflowResult:
    """Unified output container for CoupledQuantumSystem workflows."""
    solver_result: CoupledSolverResult
    population_reports: List[ChannelPopulationReport]
    evolution_result: Optional[CoupledEvolutionResult]
    validation_report: CoupledValidationReport


class CoupledQuantumSystem:
    """
    Unified high-level orchestrator for N-channel coupled quantum systems.
    Integrates CoupledPotential, CoupledHamiltonian, CoupledEigenvalueSolver,
    ChannelPopulationAnalyzer, CoupledTimeEvolution, and CoupledValidationEngine.
    """

    def __init__(
        self,
        grid: SpatialGrid,
        coupled_potential: CoupledPotential,
        mass: float = 1.0,
        hbar: float = 1.0,
        unit_system: Literal["dimensionless", "si"] = "dimensionless",
        channel_labels: Optional[Sequence[str]] = None
    ):
        self.grid = grid
        self.coupled_potential = coupled_potential
        self.num_channels = coupled_potential.num_channels
        self.mass = float(mass)
        self.hbar = float(hbar)
        self.unit_system = unit_system
        self.channel_labels = (
            [str(lbl) for lbl in channel_labels]
            if channel_labels is not None
            else [f"channel_{i}" for i in range(self.num_channels)]
        )

        self.hamiltonian = CoupledHamiltonian(
            grid=grid,
            coupled_potential=coupled_potential,
            mass=self.mass,
            hbar=self.hbar,
            unit_system=self.unit_system
        )

    @classmethod
    def from_channels(
        cls,
        grid: SpatialGrid,
        channel_potentials: Sequence[CouplingTerm],
        couplings: Optional[Dict[Tuple[int, int], CouplingTerm]] = None,
        mass: float = 1.0,
        hbar: float = 1.0,
        unit_system: Literal["dimensionless", "si"] = "dimensionless",
        channel_labels: Optional[Sequence[str]] = None
    ) -> "CoupledQuantumSystem":
        """Convenience constructor directly from diagonal channel potentials and coupling dictionary."""
        coupled_pot = CoupledPotential.from_channels(
            channel_potentials=channel_potentials,
            couplings=couplings,
            unit_system=unit_system
        )
        return cls(
            grid=grid,
            coupled_potential=coupled_pot,
            mass=mass,
            hbar=hbar,
            unit_system=unit_system,
            channel_labels=channel_labels
        )

    def solve_eigenstates(
        self,
        num_eigenvalues: int = 4,
        solver_config: Optional[SolverConfig] = None,
        accuracy_order: int = 2,
        interior_only: bool = True,
        include_uncoupled_comparison: bool = True
    ) -> CoupledSolverResult:
        """Solves the stationary coupled-channel eigenvalue problem."""
        cfg = solver_config if solver_config is not None else SolverConfig(num_eigenvalues=num_eigenvalues)
        solver = CoupledEigenvalueSolver(cfg)
        return solver.solve(
            coupled_ham=self.hamiltonian,
            accuracy_order=accuracy_order,
            interior_only=interior_only,
            include_uncoupled_comparison=include_uncoupled_comparison,
            channel_labels=self.channel_labels
        ) if False else solver.solve(
            coupled_hamiltonian=self.hamiltonian,
            accuracy_order=accuracy_order,
            interior_only=interior_only,
            include_uncoupled_comparison=include_uncoupled_comparison,
            channel_labels=self.channel_labels
        )

    def evolve(
        self,
        initial_state: MultiComponentWavefunction,
        t_end: float = 1.0,
        dt: float = 1e-3,
        t_start: float = 0.0,
        method: Literal["crank_nicolson", "rk4"] = "crank_nicolson",
        time_dependent_potential: Optional[TimeDependentCoupledPotential] = None,
        save_every: int = 1,
        norm_tolerance: float = 1e-3
    ) -> CoupledEvolutionResult:
        """Propagates a MultiComponentWavefunction through time."""
        pot: Union[CoupledPotential, TimeDependentCoupledPotential] = (
            time_dependent_potential if time_dependent_potential is not None else self.coupled_potential
        )
        evo_cfg = CoupledEvolutionConfig(
            t_start=t_start,
            t_end=t_end,
            dt=dt,
            method=method,
            save_every=save_every,
            norm_tolerance=norm_tolerance
        )
        evo_engine = CoupledTimeEvolution(
            grid=self.grid,
            coupled_potential=pot,
            mass=self.mass,
            hbar=self.hbar,
            config=evo_cfg
        )
        return evo_engine.propagate(initial_state)

    def run_workflow(
        self,
        num_eigenvalues: int = 4,
        solver_config: Optional[SolverConfig] = None,
        initial_state: Optional[MultiComponentWavefunction] = None,
        t_end: float = 1.0,
        dt: float = 1e-3,
        evolution_method: Literal["crank_nicolson", "rk4"] = "crank_nicolson",
        time_dependent_potential: Optional[TimeDependentCoupledPotential] = None,
        accuracy_order: int = 2,
        interior_only: bool = True
    ) -> CoupledSystemWorkflowResult:
        """
        Executes the end-to-end coupled quantum system workflow:
          1. Solve coupled eigenvalue problem & reconstruct multi-component eigenstates
          2. Analyze channel occupation probabilities
          3. Optionally propagate initial_state in time
          4. Generate JSON-exportable CoupledValidationReport
        """
        cfg = solver_config if solver_config is not None else SolverConfig(num_eigenvalues=num_eigenvalues)
        solver_res = self.solve_eigenstates(
            num_eigenvalues=cfg.num_eigenvalues,
            solver_config=cfg,
            accuracy_order=accuracy_order,
            interior_only=interior_only,
            include_uncoupled_comparison=True
        )

        pop_reports = [
            ChannelPopulationAnalyzer.analyze_state(st) for st in solver_res.eigenstates
        ]

        evo_res: Optional[CoupledEvolutionResult] = None
        if initial_state is not None:
            evo_res = self.evolve(
                initial_state=initial_state,
                t_end=t_end,
                dt=dt,
                method=evolution_method,
                time_dependent_potential=time_dependent_potential
            )

        validator = CoupledValidationEngine()
        val_report = validator.generate_report(
            coupled_hamiltonian=self.hamiltonian,
            solver_result=solver_res,
            solver_config=cfg,
            evolution_result=evo_res,
            accuracy_order=accuracy_order,
            interior_only=interior_only
        )

        return CoupledSystemWorkflowResult(
            solver_result=solver_res,
            population_reports=pop_reports,
            evolution_result=evo_res,
            validation_report=val_report
        )
