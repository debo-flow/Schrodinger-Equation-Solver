import numpy as np
from scipy import integrate
from schrodinger.wavefunctions.wavefunction import Wavefunction

class ProbabilityEngine:
    """Evaluates probability intervals and distributions for quantum states."""

    @staticmethod
    def interval_probability(wf: Wavefunction, a: float, b: float) -> float:
        """
        Calculates P(a <= x <= b) = int_a^b |psi(x)|^2 dx.
        """
        if a > b:
            raise ValueError("Lower bound 'a' cannot be greater than upper bound 'b'.")
            
        x = wf.x
        density = wf.probability_density
        
        # Create a boolean mask for the interval
        mask = (x >= a) & (x <= b)
        x_interval = x[mask]
        density_interval = density[mask]
        
        # If the interval is smaller than grid spacing, probability is ~0
        if len(x_interval) < 2:
            return 0.0
            
        prob = integrate.simpson(y=density_interval, x=x_interval)
        return float(np.clip(prob, 0.0, 1.0))

    @staticmethod
    def cumulative_probability(wf: Wavefunction) -> np.ndarray:
        """
        Calculates the Cumulative Distribution Function (CDF) of the position probability.
        CDF(x) = int_{x_min}^{x} |psi(x')|^2 dx'
        """
        density = wf.probability_density
        # cumulative trapezoidal integration
        cdf = integrate.cumulative_trapezoid(density, wf.x, initial=0.0)
        
        # Normalize strictly to 1.0 at the end to correct minor integration drift
        if cdf[-1] > 0:
            cdf /= cdf[-1]
            
        return cdf
