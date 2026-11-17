import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Literal

class Potential(ABC):
    """
    Abstract base class for all potential energy functions.
    Ensures safe evaluation, type promotion, and unit system tracking.
    """

    def __init__(self, unit_system: Literal["dimensionless", "si"] = "dimensionless"):
        """
        Args:
            unit_system: "dimensionless" (default, typically atomic units) or "si".
        """
        if unit_system not in ("dimensionless", "si"):
            raise ValueError("unit_system must be 'dimensionless' or 'si'.")
        self.unit_system = unit_system

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluates the potential safely for both scalar and array inputs.
        
        Args:
            x (float | np.ndarray): Position coordinate(s).
            
        Returns:
            float | np.ndarray: Evaluated potential energy.
        """
        x_arr = np.atleast_1d(x).astype(np.float64)
        
        # Guard against NaN or Inf inputs
        if not np.all(np.isfinite(x_arr)):
            raise ValueError("Input positions 'x' must be finite numbers.")
            
        # Delegate mathematical evaluation to child classes
        v_arr = self._evaluate(x_arr)
        
        # Protect against silent numerical overflow/underflow (except intended infs)
        if np.any(np.isnan(v_arr)):
            raise ArithmeticError("Potential evaluation resulted in NaN values.")

        # Return scalar if input was scalar
        if np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0):
            return float(v_arr[0])
        return v_arr

    @abstractmethod
    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Core mathematical implementation to be overridden by subclasses.
        
        Args:
            x (np.ndarray): 1D array of position coordinates.
            
        Returns:
            np.ndarray: Evaluated potential energies.
        """
        pass
