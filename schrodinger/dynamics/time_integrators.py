from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Union
import numpy as np
from schrodinger.numerical.grid import SpatialGrid
from schrodinger.potentials.base import Potential
from schrodinger.potentials.time_dependent import TimeDependentPotential, StaticPotentialWrapper


class IntegratorType(Enum):
    """Supported time integration algorithms for the TDSE."""
    RK4 = "rk4"
    CRANK_NICOLSON = "crank_nicolson"
    SPLIT_OPERATOR = "split_operator"


@dataclass
class IntegratorMetadata:
    """Documents mathematical properties, accuracy, and constraints of an integrator."""
    name: str
    temporal_order: int
    is_unconditionally_unitary: bool
    supports_time_dependent_potential: bool
    boundary_assumption: str
    constraints: List[str]


class TimeIntegrator(ABC):
    """
    Abstract base class for single-step time propagators solving:
        i * hbar * d(psi)/dt = H(t) * psi
    """

    def __init__(
        self,
        grid: SpatialGrid,
        potential: Union[Potential, TimeDependentPotential],
        mass: float = 1.0,
        hbar: float = 1.0
    ):
        if mass <= 0 or hbar <= 0:
            raise ValueError("mass and hbar must be strictly positive.")
        if isinstance(potential, TimeDependentPotential):
            self._td_potential = potential
            self._is_time_dependent = True
        elif isinstance(potential, Potential):
            self._td_potential = StaticPotentialWrapper(potential)
            self._is_time_dependent = False
        else:
            raise TypeError(
                f"potential must be Potential or TimeDependentPotential. Received: {type(potential)}"
            )

        self.grid = grid
        self.mass = float(mass)
        self.hbar = float(hbar)

    @property
    def is_time_dependent(self) -> bool:
        return self._is_time_dependent

    @property
    @abstractmethod
    def metadata(self) -> IntegratorMetadata:
        """Returns structured documentation of the integrator's assumptions and constraints."""
        pass

    @abstractmethod
    def step(self, psi: np.ndarray, t: float, dt: float) -> np.ndarray:
        """Advances wavefunction `psi` from time `t` to `t + dt`."""
        pass

    def validate_stability(self, dt: float) -> List[str]:
        """Checks time-step stability constraints and returns any diagnostic warnings."""
        if dt <= 0:
            raise ValueError(f"Time step dt must be strictly positive. Received: {dt}")
        return []
