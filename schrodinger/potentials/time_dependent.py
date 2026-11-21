import math
from abc import ABC, abstractmethod
from typing import Callable, Literal, Union
import numpy as np
from .base import Potential


class _FrozenPotentialSlice(Potential):
    """
    Static spatial Potential snapshot representing V(x, t_fixed) at a single instant.
    Ensures seamless interoperability with PotentialEnergyOperator and Hamiltonian.
    """

    def __init__(self, td_potential: "TimeDependentPotential", fixed_time: float):
        super().__init__(unit_system=td_potential.unit_system)
        self._td_potential = td_potential
        self._fixed_time = float(fixed_time)

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return self._td_potential._evaluate(x, self._fixed_time)


class TimeDependentPotential(ABC):
    """
    Abstract base class for time-dependent quantum potential functions V(x, t).
    Supports scalar and vectorized spatial evaluation at arbitrary times t,
    and generates static Potential snapshots via `at_time(t)`.
    """

    def __init__(self, unit_system: Literal["dimensionless", "si"] = "dimensionless"):
        if unit_system not in ("dimensionless", "si"):
            raise ValueError("unit_system must be 'dimensionless' or 'si'.")
        self.unit_system = unit_system

    def __call__(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        """
        Evaluates V(x, t) safely for scalar or array spatial coordinates x at time t.
        """
        if isinstance(t, bool) or not math.isfinite(t):
            raise ValueError(f"Time 't' must be a finite real number. Received: {t}")

        x_arr = np.atleast_1d(x).astype(np.float64)
        if not np.all(np.isfinite(x_arr)):
            raise ValueError("Input positions 'x' must be finite numbers.")

        v_arr = self._evaluate(x_arr, float(t))
        if np.any(np.isnan(v_arr)):
            raise ArithmeticError("Time-dependent potential evaluation resulted in NaN values.")

        if np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0):
            return float(v_arr[0])
        return v_arr

    @abstractmethod
    def _evaluate(self, x: np.ndarray, t: float) -> np.ndarray:
        """Core vectorized evaluation V(x, t) implemented by subclasses."""
        pass

    def at_time(self, t: float) -> Potential:
        """Returns a static Potential object frozen at time t."""
        if isinstance(t, bool) or not math.isfinite(t):
            raise ValueError(f"Time 't' must be finite. Received: {t}")
        return _FrozenPotentialSlice(self, float(t))


class StaticPotentialWrapper(TimeDependentPotential):
    """Adapts any time-independent Potential V(x) into the V(x, t) interface."""

    def __init__(self, potential: Potential):
        if not isinstance(potential, Potential):
            raise TypeError(f"Expected a Potential instance. Received: {type(potential)}")
        super().__init__(unit_system=potential.unit_system)
        self.potential = potential

    def _evaluate(self, x: np.ndarray, t: float) -> np.ndarray:
        return np.asarray(self.potential(x), dtype=np.float64)

    def at_time(self, t: float) -> Potential:
        return self.potential


class CallableTimeDependentPotential(TimeDependentPotential):
    """Wraps a user-supplied function f(x_array, t) -> array as a TimeDependentPotential."""

    def __init__(
        self,
        func: Callable[[np.ndarray, float], np.ndarray],
        unit_system: Literal["dimensionless", "si"] = "dimensionless"
    ):
        super().__init__(unit_system=unit_system)
        if not callable(func):
            raise TypeError("func must be callable.")
        self.func = func

    def _evaluate(self, x: np.ndarray, t: float) -> np.ndarray:
        out = np.asarray(self.func(x, t), dtype=np.float64)
        if out.shape != x.shape:
            raise ValueError(f"Potential callable returned shape {out.shape}, expected {x.shape}.")
        return out


class LinearlyDrivenPotential(TimeDependentPotential):
    """
    Time-dependent potential with a spatially linear time-varying driving field:
        V(x, t) = V_0(x) - amplitude * sin(drive_frequency * t + phase) * (x - center)
    Models laser-matter dipole interactions and driven oscillators.
    """

    def __init__(
        self,
        base_potential: Potential,
        drive_amplitude: float = 0.1,
        drive_frequency: float = 1.0,
        phase: float = 0.0,
        center: float = 0.0
    ):
        if not isinstance(base_potential, Potential):
            raise TypeError("base_potential must be a Potential instance.")
        if not (
            math.isfinite(drive_amplitude)
            and math.isfinite(drive_frequency)
            and math.isfinite(phase)
            and math.isfinite(center)
        ):
            raise ValueError("Driving parameters must be finite real numbers.")

        super().__init__(unit_system=base_potential.unit_system)
        self.base_potential = base_potential
        self.drive_amplitude = float(drive_amplitude)
        self.drive_frequency = float(drive_frequency)
        self.phase = float(phase)
        self.center = float(center)

    def _evaluate(self, x: np.ndarray, t: float) -> np.ndarray:
        v0 = np.asarray(self.base_potential(x), dtype=np.float64)
        field = self.drive_amplitude * math.sin(self.drive_frequency * t + self.phase)
        return v0 - field * (x - self.center)


class ModulatedPotential(TimeDependentPotential):
    """
    Time-dependent parametrically modulated potential:
        V(x, t) = (1.0 + modulation_depth * sin(frequency * t + phase)) * V_0(x)
    """

    def __init__(
        self,
        base_potential: Potential,
        modulation_depth: float = 0.1,
        frequency: float = 1.0,
        phase: float = 0.0
    ):
        if not isinstance(base_potential, Potential):
            raise TypeError("base_potential must be a Potential instance.")
        if not (math.isfinite(modulation_depth) and math.isfinite(frequency) and math.isfinite(phase)):
            raise ValueError("Modulation parameters must be finite real numbers.")

        super().__init__(unit_system=base_potential.unit_system)
        self.base_potential = base_potential
        self.modulation_depth = float(modulation_depth)
        self.frequency = float(frequency)
        self.phase = float(phase)

    def _evaluate(self, x: np.ndarray, t: float) -> np.ndarray:
        v0 = np.asarray(self.base_potential(x), dtype=np.float64)
        factor = 1.0 + self.modulation_depth * math.sin(self.frequency * t + self.phase)
        return factor * v0
