import numpy as np
from typing import List

class DegeneracyAnalyzer:
    """Analyzes computed energy spectrum for identical or nearly identical levels."""

    @staticmethod
    def find_degeneracies(energies: np.ndarray, tolerance: float = 1e-8) -> List[List[int]]:
        """
        Identifies degenerate energy levels.
        
        Args:
            energies (np.ndarray): Sorted 1D array of energy eigenvalues.
            tolerance (float): Maximum energy spacing to be considered degenerate.
            
        Returns:
            List[List[int]]: A list of lists, where each sublist contains the state indices
                             that share the same degenerate energy level.
        """
        if len(energies) < 2:
            return []

        degeneracies = []
        current_group = [0]

        for i in range(1, len(energies)):
            if abs(energies[i] - energies[i - 1]) < tolerance:
                current_group.append(i)
            else:
                if len(current_group) > 1:
                    degeneracies.append(current_group)
                current_group = [i]

        # Catch trailing group
        if len(current_group) > 1:
            degeneracies.append(current_group)

        return degeneracies
