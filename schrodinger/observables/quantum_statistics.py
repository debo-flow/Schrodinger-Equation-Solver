from dataclasses import dataclass
from schrodinger.numerical.grid import SpatialGrid
from schrodinger.numerical.finite_difference import FiniteDifferenceEngine
from schrodinger.wavefunctions.wavefunction import Wavefunction
from .position_operator import PositionOperator
from .momentum_operator import MomentumOperator
from .observable_engine import ObservableEngine
from .uncertainty import UncertaintyAnalyzer

@dataclass
class StateObservablesReport:
    energy: float
    expectation_x: float
    uncertainty_x: float
    expectation_p: float
    uncertainty_p: float
    uncertainty_product: float
    heisenberg_satisfied: bool

class QuantumStatistics:
    """High-level aggregator for full state-dependent observable reports."""

    def __init__(self, grid: SpatialGrid, hbar: float = 1.0):
        self.grid = grid
        self.hbar = hbar
        self.fd_engine = FiniteDifferenceEngine(grid)
        
        # Build operator matrices once
        self.x_op = PositionOperator.build_x(grid)
        self.x2_op = PositionOperator.build_x_squared(grid)
        self.p_op = MomentumOperator.build_p(self.fd_engine, self.hbar)
        self.p2_op = MomentumOperator.build_p_squared(self.fd_engine, self.hbar)

    def analyze_state(self, wf: Wavefunction) -> StateObservablesReport:
        """Generates a complete statistics report for a given wavefunction."""
        exp_x = ObservableEngine.expectation(self.x_op, wf, is_hermitian=True)
        delta_x = UncertaintyAnalyzer.calculate_uncertainty(self.x_op, self.x2_op, wf)
        
        exp_p = ObservableEngine.expectation(self.p_op, wf, is_hermitian=True)
        delta_p = UncertaintyAnalyzer.calculate_uncertainty(self.p_op, self.p2_op, wf)
        
        product = delta_x * delta_p
        heisenberg_ok = UncertaintyAnalyzer.verify_heisenberg(delta_x, delta_p, self.hbar)
        
        return StateObservablesReport(
            energy=wf.energy,
            expectation_x=exp_x,
            uncertainty_x=delta_x,
            expectation_p=exp_p,
            uncertainty_p=delta_p,
            uncertainty_product=product,
            heisenberg_satisfied=heisenberg_ok
        )
