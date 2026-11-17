import numpy as np
from typing import Literal
from .base import Potential


class FreeParticle(Potential):
    """A region of space with strictly zero potential energy."""
    
    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)


class InfiniteSquareWell(Potential):
    """An infinitely deep well. V=0 inside, V=infinity outside."""
    
    def __init__(self, width: float = 1.0, center: float = 0.0, 
                 unit_system: Literal["dimensionless", "si"] = "dimensionless"):
        super().__init__(unit_system)
        if width <= 0:
            raise ValueError("Width of the well must be strictly positive.")
        self.width = width
        self.center = center

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        v = np.full_like(x, np.inf)
        left_bound = self.center - self.width / 2.0
        right_bound = self.center + self.width / 2.0
        
        inside = (x > left_bound) & (x < right_bound)
        v[inside] = 0.0
        return v


class FiniteSquareWell(Potential):
    """A well of finite depth -V0. V=-V0 inside, V=0 outside."""
    
    def __init__(self, depth: float, width: float, center: float = 0.0, 
                 unit_system: Literal["dimensionless", "si"] = "dimensionless"):
        super().__init__(unit_system)
        if depth <= 0:
            raise ValueError("Depth must be strictly positive.")
        if width <= 0:
            raise ValueError("Width must be strictly positive.")
        self.depth = depth
        self.width = width
        self.center = center

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        v = np.zeros_like(x)
        left_bound = self.center - self.width / 2.0
        right_bound = self.center + self.width / 2.0
        
        inside = (x >= left_bound) & (x <= right_bound)
        v[inside] = -self.depth
        return v


class HarmonicOscillator(Potential):
    """Quadratic potential V(x) = 0.5 * m * omega^2 * (x - center)^2."""
    
    def __init__(self, mass: float = 1.0, omega: float = 1.0, center: float = 0.0, 
                 unit_system: Literal["dimensionless", "si"] = "dimensionless"):
        super().__init__(unit_system)
        if mass <= 0 or omega <= 0:
            raise ValueError("Mass and angular frequency (omega) must be positive.")
        self.mass = mass
        self.omega = omega
        self.center = center
        self.k = mass * omega**2

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * self.k * (x - self.center)**2


class DoubleWell(Potential):
    """Symmetric double well V(x) = V_b * ((x - x_c)^2 / a^2 - 1)^2."""
    
    def __init__(self, barrier_height: float, minima_offset: float, center: float = 0.0, 
                 unit_system: Literal["dimensionless", "si"] = "dimensionless"):
        super().__init__(unit_system)
        if barrier_height <= 0:
            raise ValueError("Barrier height must be strictly positive.")
        if minima_offset <= 0:
            raise ValueError("Minima offset must be strictly positive.")
        self.barrier_height = barrier_height
        self.minima_offset = minima_offset
        self.center = center

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        scaled_dist_sq = ((x - self.center) / self.minima_offset)**2
        return self.barrier_height * (scaled_dist_sq - 1.0)**2
