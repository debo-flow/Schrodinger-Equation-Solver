import pytest
import numpy as np
from scipy.sparse import issparse
from schrodinger.numerical.grid import SpatialGrid
from schrodinger.numerical.finite_difference import FiniteDifferenceEngine
from schrodinger.potentials.models import HarmonicOscillator, InfiniteSquareWell
from schrodinger.operators import (
    OperatorValidator, KineticEnergyOperator, PotentialEnergyOperator, Hamiltonian
)

@pytest.fixture
def base_grid():
    return SpatialGrid(x_min=-5.0, x_max=5.0, points=100)

@pytest.fixture
def fd_engine(base_grid):
    return FiniteDifferenceEngine(base_grid)

@pytest.fixture
def harmonic_potential():
    return HarmonicOscillator(mass=1.0, omega=1.0)

def test_physical_constants_validation():
    """Ensure negative mass or hbar raises an error."""
    with pytest.raises(ValueError, match="strictly positive"):
        OperatorValidator.validate_physical_constants(mass=-1.0, hbar=1.0)
    with pytest.raises(ValueError, match="strictly positive"):
        OperatorValidator.validate_physical_constants(mass=1.0, hbar=-1.0)

def test_unit_consistency_validation(fd_engine):
    """Ensure combining SI kinetic with Dimensionless potential raises an error."""
    pot_si = HarmonicOscillator(unit_system="si")
    kin_dim = KineticEnergyOperator(fd_engine, unit_system="dimensionless")
    
    with pytest.raises(ValueError, match="Unit mismatch"):
        Hamiltonian(kin_dim, PotentialEnergyOperator(fd_engine.grid, pot_si))

def test_kinetic_operator_scaling(fd_engine):
    """Verify that changing mass/hbar properly scales the operator matrix."""
    kin_1 = KineticEnergyOperator(fd_engine, mass=1.0, hbar=1.0)
    kin_2 = KineticEnergyOperator(fd_engine, mass=2.0, hbar=1.0)
    
    mat_1 = kin_1.build_matrix()
    mat_2 = kin_2.build_matrix()
    
    # Mass doubled -> kinetic energy matrix should be halved
    np.testing.assert_allclose(mat_1.toarray(), 2.0 * mat_2.toarray())

def test_potential_operator_matrix(base_grid, harmonic_potential):
    """Verify the potential matrix is strictly diagonal and correctly sized."""
    pot_op = PotentialEnergyOperator(base_grid, harmonic_potential)
    v_mat = pot_op.build_matrix(interior_only=False)
    
    assert issparse(v_mat)
    assert v_mat.shape == (100, 100)
    
    # Check that it's strictly diagonal by isolating the diagonal elements
    v_dense = v_mat.toarray()
    assert np.count_nonzero(v_dense - np.diag(np.diagonal(v_dense))) == 0

def test_hamiltonian_assembly_and_hermiticity(fd_engine, harmonic_potential):
    """Test full assembly of H and successful Hermiticity validation."""
    kin_op = KineticEnergyOperator(fd_engine)
    pot_op = PotentialEnergyOperator(fd_engine.grid, harmonic_potential)
    
    H = Hamiltonian(kin_op, pot_op)
    h_sparse = H.build_sparse(accuracy_order=2, interior_only=True)
    
    # Matrix should be strictly symmetric
    # (The check_hermiticity runs inside build_sparse, so passing it implies success)
    assert issparse(h_sparse)
    assert h_sparse.shape == (98, 98) # interior_only reduces N=100 to N=98

def test_dense_sparse_equivalence(fd_engine, harmonic_potential):
    """Verify that build_dense() perfectly matches build_sparse().toarray()."""
    kin_op = KineticEnergyOperator(fd_engine)
    pot_op = PotentialEnergyOperator(fd_engine.grid, harmonic_potential)
    
    H = Hamiltonian(kin_op, pot_op)
    h_dense = H.build_dense(accuracy_order=4)
    h_sparse = H.build_sparse(accuracy_order=4)
    
    np.testing.assert_array_equal(h_dense, h_sparse.toarray())

def test_asymmetric_matrix_rejection():
    """Ensure the validator correctly catches broken (non-Hermitian) matrices."""
    bad_mat = sparse.csr_matrix(np.array([[1.0, 2.0], [0.0, 1.0]]))
    
    with pytest.raises(ValueError, match="not Hermitian"):
        OperatorValidator.check_hermiticity(bad_mat)
