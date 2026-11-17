import pytest
import numpy as np
from scipy import sparse
from schrodinger.solvers import (
    SolverConfig, SolverMethod, EigenvalueSolver, 
    DenseSolver, SparseSolver, EigenpairValidator, DegeneracyAnalyzer
)

@pytest.fixture
def diagonal_hamiltonian():
    """A simple 4x4 matrix with known eigenvalues (1.0, 2.0, 2.0, 4.0)"""
    return np.diag([4.0, 1.0, 2.0, 2.0])

@pytest.fixture
def sparse_diagonal_hamiltonian(diagonal_hamiltonian):
    return sparse.csr_matrix(diagonal_hamiltonian)

def test_dense_solver_accuracy(diagonal_hamiltonian):
    energies, wavefunctions = DenseSolver.solve(diagonal_hamiltonian, num_eigenvalues=4)
    
    # Should be sorted ascending
    np.testing.assert_array_almost_equal(energies, [1.0, 2.0, 2.0, 4.0])
    # Check ground state eigenvector (corresponds to index 1 in original matrix)
    assert np.abs(wavefunctions[1, 0]) == 1.0

def test_sparse_solver_accuracy():
    # Make a larger matrix for sparse solver (needs N > k+1)
    large_diag = sparse.diags([np.arange(1, 21, dtype=float)], [0], format='csr')
    config = SolverConfig(num_eigenvalues=3, tolerance=1e-10)
    
    energies, wavefunctions = SparseSolver.solve(large_diag, config)
    
    assert len(energies) == 3
    np.testing.assert_array_almost_equal(energies, [1.0, 2.0, 3.0])

def test_eigenpair_validation(diagonal_hamiltonian):
    energies = np.array([1.0])
    wavefunctions = np.array([[0.0], [1.0], [0.0], [0.0]])
    
    # Correct pair should have residual 0
    res = EigenpairValidator.calculate_residual(diagonal_hamiltonian, energies[0], wavefunctions[:, 0])
    assert res == 0.0
    
    # Invalid pair should fail
    bad_energies = np.array([1.5])
    with pytest.raises(ValueError, match="failed residual validation"):
        EigenpairValidator.validate_all_pairs(diagonal_hamiltonian, bad_energies, wavefunctions)

def test_degeneracy_analyzer():
    energies = np.array([1.0, 2.0, 2.0 + 1e-9, 4.0, 5.0, 5.0])
    degen = DegeneracyAnalyzer.find_degeneracies(energies, tolerance=1e-5)
    
    assert len(degen) == 2
    assert degen[0] == [1, 2]
    assert degen[1] == [4, 5]

def test_eigenvalue_solver_orchestration(diagonal_hamiltonian, sparse_diagonal_hamiltonian):
    # Test Auto Routing & Validation
    config = SolverConfig(num_eigenvalues=3, method=SolverMethod.AUTO)
    solver = EigenvalueSolver(config)
    
    # Pass dense matrix
    result_dense = solver.solve(diagonal_hamiltonian)
    assert result_dense["method_used"] == "dense"
    np.testing.assert_array_almost_equal(result_dense["energies"], [1.0, 2.0, 2.0])
    
    # Pass sparse matrix (Since N=4, it should auto-fallback to Dense because k >= N-2)
    config_sparse = SolverConfig(num_eigenvalues=3, method=SolverMethod.SPARSE)
    solver_sparse = EigenvalueSolver(config_sparse)
    with pytest.raises(ValueError, match="dimension too small for sparse"):
        # Explicitly forcing sparse on a tiny matrix fails the k < N-1 condition
        solver_sparse.solve(sparse_diagonal_hamiltonian)
        
    # Auto should correctly pick dense for this small sparse matrix
    result_auto_sparse = solver.solve(sparse_diagonal_hamiltonian)
    assert result_auto_sparse["method_used"] == "dense"

def test_non_hermitian_rejection():
    bad_matrix = np.array([[1.0, 2.0], [0.0, 1.0]])
    solver = EigenvalueSolver()
    
    with pytest.raises(ValueError, match="not Hermitian"):
        solver.solve(bad_matrix)
