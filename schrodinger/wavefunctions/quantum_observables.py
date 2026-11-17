import numpy as np
from scipy import integrate

class QuantumObservables:
    """Calculates physical observables from normalized spatial wavefunctions."""

    @staticmethod
    def expectation_x(psi: np.ndarray, x: np.ndarray) -> float:
        """Calculates <x> = integral(x * |psi|^2 dx)"""
        density = np.abs(psi)**2
        integrand = x * density
        return float(integrate.simpson(y=integrand, x=x))

    @staticmethod
    def variance_x(psi: np.ndarray, x: np.ndarray) -> float:
        """Calculates Var(x) = <x^2> - <x>^2"""
        density = np.abs(psi)**2
        
        exp_x = QuantumObservables.expectation_x(psi, x)
        exp_x2 = float(integrate.simpson(y=(x**2 * density), x=x))
        
        variance = exp_x2 - exp_x**2
        return max(0.0, variance)  # Prevent tiny negative floats from numerical error

    @staticmethod
    def count_nodes(psi: np.ndarray) -> int:
        """
        Counts the number of nodes (zero-crossings) in the real part of the wavefunction.
        Nodes define the excitation level of bound states in 1D.
        """
        real_part = np.real(psi)
        # Find where the sign flips between adjacent elements
        sign_flips = np.where(np.diff(np.signbit(real_part)))[0]
        
        # Filter out tiny numerical noise near exactly zero
        threshold = 1e-8 * np.max(np.abs(real_part))
        valid_flips = [
            i for i in sign_flips 
            if abs(real_part[i]) > threshold or abs(real_part[i+1]) > threshold
        ]
        
        return len(valid_flips)
