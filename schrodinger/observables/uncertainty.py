import numpy as np
from scipy import sparse
from schrodinger.wavefunctions.wavefunction import Wavefunction
from .observable_engine import ObservableEngine

class UncertaintyAnalyzer:
    """Evaluates quantum uncertainties and the Heisenberg relations."""

    @staticmethod
    def calculate_uncertainty(op: sparse.spmatrix, op_sq: sparse.spmatrix, wf: Wavefunction) -> float:
        """
        Calculates standard deviation: Delta A = sqrt(<A^2> - <A>^2).
        """
        exp_A = ObservableEngine.expectation(op, wf, is_hermitian=True)
        exp_A2 = ObservableEngine.expectation(op_sq, wf, is_hermitian=True)
        
        variance = exp_A2 - (exp_A**2)
        # Prevent math domain error from numerical zero-crossings
        if variance < 0 and abs(variance) < 1e-10:
            variance = 0.0
            
        return float(np.sqrt(variance))

    @staticmethod
    def verify_heisenberg(delta_x: float, delta_p: float, hbar: float = 1.0, tolerance: float = 1e-5) -> bool:
        """
        Verifies if Delta x * Delta p >= hbar / 2.
        Includes a numerical tolerance to account for boundary grid effects.
        """
        product = delta_x * delta_p
        limit = hbar / 2.0
        
        # Check if product is greater than or approximately equal to the limit
        return product >= limit or abs(product - limit) < tolerance
