import numpy as np
from scipy import integrate, sparse
from schrodinger.wavefunctions.wavefunction import Wavefunction

class ObservableEngine:
    """Calculates generalized expectation values <A> = <psi|A|psi>."""

    @staticmethod
    def expectation(operator_matrix: sparse.spmatrix, wf: Wavefunction, is_hermitian: bool = True) -> Union[float, complex]:
        """
        Computes the expectation value of a sparse operator using continuous integration.
        
        Args:
            operator_matrix: The sparse matrix representing the operator.
            wf: The normalized Wavefunction object.
            is_hermitian: If True, forces the return type to float (drops imaginary numerical noise).
            
        Returns:
            The expectation value.
        """
        # Apply operator: |phi> = A |psi>
        A_psi = operator_matrix.dot(wf.psi)
        
        # Integrate: int psi* (A psi) dx
        integrand = np.conj(wf.psi) * A_psi
        exp_val = integrate.simpson(y=integrand, x=wf.x)
        
        if is_hermitian:
            if abs(np.imag(exp_val)) > 1e-5:
                raise ValueError(f"Hermitian operator yielded significant imaginary expectation: {exp_val}")
            return float(np.real(exp_val))
            
        return complex(exp_val)
