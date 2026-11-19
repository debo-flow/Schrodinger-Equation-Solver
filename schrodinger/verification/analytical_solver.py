from abc import ABC, abstractmethod
from typing import Dict, Any, List, Literal, Type
import numpy as np
from schrodinger.wavefunctions.wavefunction import Wavefunction


class AnalyticalSolution(ABC):
    """
    Abstract base class for analytical and reference quantum mechanical solutions.
    Provides a standardized reference interface for numerical verification.
    """

    def __init__(
        self,
        mass: float = 1.0,
        hbar: float = 1.0,
        unit_system: Literal["dimensionless", "si"] = "dimensionless"
    ):
        if mass <= 0:
            raise ValueError(f"Particle mass must be strictly positive. Received: {mass}")
        if hbar <= 0:
            raise ValueError(f"Reduced Planck constant (hbar) must be strictly positive. Received: {hbar}")
        if unit_system not in ("dimensionless", "si"):
            raise ValueError("unit_system must be 'dimensionless' or 'si'.")

        self.mass = float(mass)
        self.hbar = float(hbar)
        self.unit_system = unit_system

    @property
    def is_exact(self) -> bool:
        """Indicates whether this reference provides exact closed-form solutions."""
        return True

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier of the analytical model."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """Dictionary of physical parameters defining the system."""
        pass

    @abstractmethod
    def quantum_numbers(self, num_states: int) -> List[int]:
        """
        Returns the ordered quantum numbers corresponding to the lowest `num_states` levels.
        """
        pass

    @abstractmethod
    def energy(self, n: int) -> float:
        """Calculates the reference energy eigenvalue for quantum number n."""
        pass

    @abstractmethod
    def normalization_constant(self, n: int) -> float:
        """Returns the normalization constant for state n."""
        pass

    @abstractmethod
    def wavefunction(self, n: int, x: np.ndarray) -> np.ndarray:
        """Evaluates the normalized eigenfunction psi_n(x) on spatial grid x."""
        pass

    @abstractmethod
    def verify_boundary_conditions(self, n: int, x: np.ndarray, tolerance: float = 1e-6) -> bool:
        """Verifies whether state n satisfies the physical boundary conditions on grid x."""
        pass

    def energy_spectrum(self, num_states: int) -> np.ndarray:
        """Returns a 1D array of the lowest `num_states` reference energy eigenvalues."""
        if num_states <= 0:
            raise ValueError("num_states must be a positive integer.")
        return np.array([self.energy(n) for n in self.quantum_numbers(num_states)], dtype=np.float64)

    def probability_density(self, n: int, x: np.ndarray) -> np.ndarray:
        """Evaluates the reference probability density |psi_n(x)|^2."""
        psi = self.wavefunction(n, x)
        return np.abs(psi) ** 2

    def eigenstates(self, num_states: int, x: np.ndarray) -> List[Wavefunction]:
        """Constructs Wavefunction objects for the lowest `num_states` reference eigenstates."""
        states = []
        for n in self.quantum_numbers(num_states):
            psi = self.wavefunction(n, x)
            e = self.energy(n)
            states.append(Wavefunction(x=x, psi=psi, energy=e))
        return states


class AnalyticalSolver:
    """
    Factory and registry for instantiating analytical reference models by name.
    """

    _registry: Dict[str, Type[AnalyticalSolution]] = {}

    @classmethod
    def register(cls, key: str, solution_cls: Type[AnalyticalSolution]) -> None:
        cls._registry[key.lower()] = solution_cls

    @classmethod
    def create(cls, model_name: str, **kwargs: Any) -> AnalyticalSolution:
        key = model_name.lower()
        if key not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(f"Unknown analytical model '{model_name}'. Available: {available}")
        return cls._registry[key](**kwargs)

    @classmethod
    def available_models(cls) -> List[str]:
        return sorted(cls._registry.keys())
