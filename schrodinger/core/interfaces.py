"""
Abstract Base Classes defining the core architecture of the Schrödinger Solver.
Enforces strict separation of mathematical concepts and numerical implementations.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import numpy as np


class Wavefunction(ABC):
    """Represents a quantum mechanical wavefunction and its properties."""
    
    @property
    @abstractmethod
    def x(self) -> np.ndarray:
        """Position space array."""
        pass

    @property
    @abstractmethod
    def psi(self) -> np.ndarray:
        """Wavefunction amplitude array."""
        pass

    @property
    @abstractmethod
    def energy(self) -> float:
        """Energy eigenvalue associated with this wavefunction."""
        pass

    @abstractmethod
    def probability_density(self) -> np.ndarray:
        """Returns the probability density |psi(x)|^2."""
        pass

    @abstractmethod
    def normalize(self) -> None:
        """Normalizes the wavefunction in place such that integral(|psi|^2 dx) = 1."""
        pass


class Potential(ABC):
    """Represents the potential energy function V(x)."""
    
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the potential energy across the spatial grid.
        
        Args:
            x (np.ndarray): Spatial grid.
            
        Returns:
            np.ndarray: Potential energy evaluated at each point in x.
        """
        pass


class Hamiltonian(ABC):
    """Represents the Hamiltonian operator matrix constructor."""
    
    @abstractmethod
    def construct_matrix(self, x: np.ndarray, potential: np.ndarray, mass: float, hbar: float) -> np.ndarray:
        """
        Constructs the Hamiltonian matrix H = T + V.
        
        Args:
            x (np.ndarray): Discretized position space.
            potential (np.ndarray): Discretized potential V(x).
            mass (float): Particle mass.
            hbar (float): Reduced Planck constant.
            
        Returns:
            np.ndarray: The Hamiltonian matrix (dense or sparse).
        """
        pass


class EigenvalueSolver(ABC):
    """Interface for mathematical eigenvalue solving algorithms."""
    
    @abstractmethod
    def solve(self, hamiltonian_matrix: np.ndarray, num_states: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solves H*psi = E*psi.
        
        Args:
            hamiltonian_matrix (np.ndarray): The matrix to diagonalize.
            num_states (int, optional): Number of lowest energy states to return.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - 1D array of eigenvalues (Energies)
                - 2D array of eigenvectors (Wavefunctions)
        """
        pass


class NumericalEngine(ABC):
    """Handles the discretization and numerical configurations of the space."""
    
    @abstractmethod
    def discretize_space(self, x_min: float, x_max: float, points: int) -> np.ndarray:
        """Generates the position space grid."""
        pass


class QuantumSystem(ABC):
    """
    High-level orchestrator that binds the Potential, Engine, Hamiltonian, 
    and Solver to produce Wavefunctions.
    """
    
    @abstractmethod
    def __init__(self, potential: Potential, mass: float = 1.0, hbar: float = 1.0):
        """Initializes the physical system parameters."""
        pass

    @abstractmethod
    def solve_system(self, engine: NumericalEngine, solver: EigenvalueSolver, 
                     hamiltonian_builder: Hamiltonian) -> List[Wavefunction]:
        """
        Executes the numerical pipeline to solve the Schrödinger equation.
        
        Returns:
            List[Wavefunction]: A list of solved, normalized energy eigenstates.
        """
        pass
