import numpy as np
from typing import Dict, Any, List
from schrodinger.numerical.grid import SpatialGrid, BoundaryCondition
from .wavefunction import Wavefunction
from .phase_manager import PhaseManager
from .normalization import NormalizationEngine, IntegrationMethod

class WavefunctionReconstructor:
    """
    Constructs normalized Wavefunction objects from the raw output of the EigenvalueSolver.
    """

    def __init__(self, grid: SpatialGrid, integration_method: IntegrationMethod = IntegrationMethod.SIMPSON):
        self.grid = grid
        self.integration_method = integration_method

    def reconstruct(self, solver_result: Dict[str, Any]) -> List[Wavefunction]:
        """
        Takes the raw dictionary from EigenvalueSolver and constructs physical Wavefunctions.
        Properly handles boundary padding for Dirichlet conditions.
        """
        energies = solver_result["energies"]
        raw_psis = solver_result["wavefunctions"]
        
        num_states = len(energies)
        wavefunctions = []
        
        # For Dirichlet, the solver might have returned an interior-only N-2 matrix
        is_interior_only = (raw_psis.shape[0] == self.grid.points - 2)

        for i in range(num_states):
            energy = energies[i]
            raw_psi_vector = raw_psis[:, i]

            # 1. Pad boundaries if interior-only matrix was used
            if is_interior_only and self.grid.boundary_condition == BoundaryCondition.DIRICHLET:
                full_psi = np.zeros(self.grid.points, dtype=complex)
                full_psi[1:-1] = raw_psi_vector
            else:
                full_psi = raw_psi_vector.astype(complex)

            # 2. Continuous Normalization
            norm_psi, _ = NormalizationEngine.normalize(full_psi, self.grid.grid, self.integration_method)

            # 3. Global Phase Standardization
            std_psi = PhaseManager.standardize_phase(norm_psi)

            # 4. Object Instantiation
            wf = Wavefunction(x=self.grid.grid, psi=std_psi, energy=energy)
            wavefunctions.append(wf)

        return wavefunctions
