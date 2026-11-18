import math
from typing import Dict, Any, List, Literal, Optional
import numpy as np
import sympy as sp
from .analytical_solver import AnalyticalSolution, AnalyticalSolver


class FreeParticleAnalytical(AnalyticalSolution):
    """
    Analytical plane-wave reference model for a 1D Free Particle (V(x) = 0).

    Continuous spectrum:
        E(k) = hbar^2 * k^2 / (2 * m)
        p(k) = hbar * k
        psi_k(x) = A * exp(i * k * x)

    When `box_length` is specified, integer quantum numbers n map to periodic
    momentum eigenstates k_n = 2 * pi * n / L with box normalization A = 1/sqrt(L).
    """

    def __init__(
        self,
        box_length: Optional[float] = None,
        mass: float = 1.0,
        hbar: float = 1.0,
        unit_system: Literal["dimensionless", "si"] = "dimensionless"
    ):
        super().__init__(mass=mass, hbar=hbar, unit_system=unit_system)
        if box_length is not None and box_length <= 0:
            raise ValueError(f"box_length must be strictly positive. Received: {box_length}")
        self.box_length = float(box_length) if box_length is not None else None

    @property
    def name(self) -> str:
        return "FreeParticle"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "box_length": self.box_length,
            "mass": self.mass,
            "hbar": self.hbar,
            "unit_system": self.unit_system
        }

    def wavevector(self, n: int) -> float:
        """Maps an integer mode n to periodic wavevector k_n = 2*pi*n / L."""
        if self.box_length is None:
            raise ValueError("Discrete wavevector k_n requires a finite `box_length`.")
        return float(2.0 * math.pi * n / self.box_length)

    def momentum_from_k(self, k: float) -> float:
        """Returns the exact momentum eigenvalue p = hbar * k."""
        return float(self.hbar * k)

    def energy_from_k(self, k: float) -> float:
        """Returns the continuous dispersion energy E(k) = hbar^2 * k^2 / (2m)."""
        return float((self.hbar ** 2) * (k ** 2) / (2.0 * self.mass))

    def plane_wave(
        self,
        k: float,
        x: np.ndarray,
        normalization: Literal["box", "dirac"] = "box"
    ) -> np.ndarray:
        """
        Evaluates the complex momentum eigenstate psi_k(x) = A * exp(i * k * x).
        
        Args:
            k (float): Wavevector.
            x (np.ndarray): Spatial coordinate array.
            normalization (str):
                - 'box': A = 1 / sqrt(L) (uses box_length or domain span x[-1] - x[0]).
                - 'dirac': A = 1 / sqrt(2 * pi) for delta-function normalization.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        if normalization == "dirac":
            amplitude = 1.0 / math.sqrt(2.0 * math.pi)
        elif normalization == "box":
            length = self.box_length if self.box_length is not None else float(x_arr[-1] - x_arr[0])
            if length <= 0:
                raise ValueError("Domain length must be strictly positive for box normalization.")
            amplitude = 1.0 / math.sqrt(length)
        else:
            raise ValueError("normalization must be 'box' or 'dirac'.")

        return amplitude * np.exp(1j * k * x_arr)

    def quantum_numbers(self, num_states: int) -> List[int]:
        """
        Returns periodic mode indices ordered by ascending energy: [0, 1, -1, 2, -2, ...].
        """
        if num_states <= 0:
            raise ValueError("num_states must be strictly positive.")
        modes = [0]
        k_idx = 1
        while len(modes) < num_states:
            modes.append(k_idx)
            if len(modes) < num_states:
                modes.append(-k_idx)
            k_idx += 1
        return modes

    def energy(self, n: int) -> float:
        """Calculates box-quantized periodic free-particle energy for mode n."""
        return self.energy_from_k(self.wavevector(n))

    def normalization_constant(self, n: int) -> float:
        """Returns 1/sqrt(L) for box normalization, or 1/sqrt(2*pi) if unbounded."""
        if self.box_length is not None:
            return float(1.0 / math.sqrt(self.box_length))
        return float(1.0 / math.sqrt(2.0 * math.pi))

    def wavefunction(self, n: int, x: np.ndarray) -> np.ndarray:
        """Evaluates periodic plane-wave eigenstate for integer mode n."""
        k = self.wavevector(n)
        return self.plane_wave(k, x, normalization="box")

    def verify_boundary_conditions(self, n: int, x: np.ndarray, tolerance: float = 1e-10) -> bool:
        """Verifies periodic boundary condition psi(x_min) == psi(x_max)."""
        psi = self.wavefunction(n, np.array([x[0], x[-1]], dtype=np.float64))
        return bool(abs(psi[0] - psi[-1]) <= tolerance)

    @staticmethod
    def symbolic_dispersion() -> sp.Expr:
        """Returns the SymPy symbolic expression for E(k)."""
        k, hbar, m = sp.symbols("k hbar m", real=True, positive=True)
        return (hbar ** 2 * k ** 2) / (2 * m)

    @staticmethod
    def symbolic_plane_wave() -> sp.Expr:
        """Returns the SymPy symbolic expression for Dirac-normalized plane wave."""
        x, k = sp.symbols("x k", real=True)
        return (1 / sp.sqrt(2 * sp.pi)) * sp.exp(sp.I * k * x)


AnalyticalSolver.register("free_particle", FreeParticleAnalytical)
