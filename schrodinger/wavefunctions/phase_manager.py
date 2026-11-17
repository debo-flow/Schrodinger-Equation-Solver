import numpy as np

class PhaseManager:
    """Manages and standardizes the global complex phase of wavefunctions."""

    @staticmethod
    def standardize_phase(psi: np.ndarray) -> np.ndarray:
        """
        Rotates the global phase such that the point of maximum amplitude
        is strictly real and positive.
        
        Args:
            psi (np.ndarray): The unstandardized wavefunction array.
            
        Returns:
            np.ndarray: The phase-standardized wavefunction.
        """
        if len(psi) == 0:
            return psi
            
        # Find index of maximum absolute amplitude
        max_idx = np.argmax(np.abs(psi))
        
        # Calculate its phase angle
        phase = np.angle(psi[max_idx])
        
        # Rotate entire wavefunction to cancel this phase
        standardized_psi = psi * np.exp(-1j * phase)
        
        # Eliminate tiny floating point imaginary residuals if it's functionally real
        if np.max(np.abs(np.imag(standardized_psi))) < 1e-12:
            standardized_psi = np.real(standardized_psi).astype(complex)
            
        return standardized_psi
