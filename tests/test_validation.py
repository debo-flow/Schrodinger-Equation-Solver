import pytest
import numpy as np
from scipy import sparse
import json
from schrodinger.analysis import (
    HermiticityAnalyzer, SpectralAnalyzer, ExpectationCalculator, 
    MatrixQualityAnalyzer, HamiltonianValidator
)

@pytest.fixture
def hermitian_dense():
    return np.array([[2.0, 1.0j], [-1.0j, 2.0]])

@pytest.fixture
def non_hermitian_dense():
    return np.array([[2.0, 1.0], [0.0, 2.0]])

@pytest.fixture
def hermitian_sparse():
    return sparse.csr_matrix([[2.0, 0.5], [0.5, 2.0]])

def test_hermiticity_analyzer(hermitian_dense, non_hermitian_dense, hermitian_sparse):
    analyzer = HermiticityAnalyzer()
    
    # Check dense Hermitian
    is_herm, dev = analyzer.is_hermitian(hermitian_dense)
    assert is_herm is True
    assert dev == 0.0
    
    # Check dense non-Hermitian
    is_herm, dev = analyzer.is_hermitian(non_hermitian_dense)
    assert is_herm is False
    assert dev == 1.0
    
    # Check sparse Hermitian
    is_herm, dev = analyzer.is_hermitian(hermitian_sparse)
    assert is_herm is True
    assert dev == 0.0

def test_spectral_analyzer(hermitian_dense, hermitian_sparse):
    # Trace tests
    assert SpectralAnalyzer.calculate_trace(hermitian_dense) == 4.0
    assert SpectralAnalyzer.calculate_trace(hermitian_sparse) == 4.0
    
    # Norm tests
    expected_dense_norm = np.sqrt(2**2 + 1**2 + 1**2 + 2**2) # sqrt(10)
    assert np.isclose(SpectralAnalyzer.calculate_norm(hermitian_dense), expected_dense_norm)
    
    # Dense properties (rank & det)
    props = SpectralAnalyzer.analyze_dense_properties(hermitian_dense)
    assert props["rank"] == 2
    assert np.isclose(props["determinant"], 3.0) # 4 - 1

def test_expectation_calculator(hermitian_dense, non_hermitian_dense):
    calc = ExpectationCalculator()
    psi = np.array([1.0, 0.0]) # unnormalized state |0>
    
    # Expected value <psi|H|psi> / <psi|psi> for [[2, i], [-i, 2]] is 2.0
    energy = calc.evaluate(hermitian_dense, psi, normalize=True)
    assert energy == 2.0
    
    # Complex state
    psi_complex = np.array([1.0j, 1.0])
    energy_c = calc.evaluate(hermitian_dense, psi_complex, normalize=True)
    # H @ psi = [2i+i, 1+2] = [3i, 3]
    # psi.conj().T @ H_psi = [-i, 1] @ [3i, 3] = 3 + 3 = 6
    # norm = 2. Result = 3.0
    assert energy_c == 3.0
    
    # Test imaginary warning trigger
    psi_bad = np.array([1.0, 1.0])
    with pytest.raises(ValueError, match="significant imaginary component"):
        calc.evaluate(non_hermitian_dense, psi_bad)

def test_matrix_quality():
    good_mat = np.array([[1.0, 2.0], [2.0, 1.0]])
    bad_mat = np.array([[1.0, np.inf], [np.nan, 1.0]])
    sparse_bad = sparse.csr_matrix(bad_mat)
    
    assert MatrixQualityAnalyzer.check_finiteness(good_mat) is True
    assert MatrixQualityAnalyzer.check_finiteness(bad_mat) is False
    assert MatrixQualityAnalyzer.check_finiteness(sparse_bad) is False
    
    warnings = MatrixQualityAnalyzer.find_invalid_entries(bad_mat)
    assert "Matrix contains NaN or Inf values." in warnings

def test_validation_report_generation(hermitian_sparse):
    validator = HamiltonianValidator()
    report = validator.generate_report(hermitian_sparse)
    
    assert report.is_valid is True
    assert report.is_hermitian is True
    assert report.is_finite is True
    assert report.trace == 4.0
    
    # JSON Serialization test
    json_str = report.to_json()
    assert isinstance(json_str, str)
    data = json.loads(json_str)
    assert data["is_valid"] is True
    assert data["dimensions"] == [2, 2]
