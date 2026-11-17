import json
from dataclasses import dataclass, asdict
from typing import Union, List, Optional
import numpy as np
from scipy import sparse
from .hermiticity import HermiticityAnalyzer
from .spectral_analysis import SpectralAnalyzer
from .matrix_quality import MatrixQualityAnalyzer

@dataclass
class ValidationReport:
    """Structured data container for Hamiltonian validation results."""
    dimensions: List[int]
    is_square: bool
    is_hermitian: bool
    hermiticity_deviation: float
    trace: float
    norm: float
    rank: Optional[float]
    determinant: Optional[float]
    is_finite: bool
    warnings: List[str]
    is_valid: bool

    def to_json(self) -> str:
        """Serializes the report to a formatted JSON string."""
        return json.dumps(asdict(self), indent=4)


class HamiltonianValidator:
    """Orchestrates all analyzers to generate a comprehensive matrix report."""

    def __init__(self, tolerance: float = 1e-10):
        self.tolerance = tolerance
        self.herm_analyzer = HermiticityAnalyzer(tolerance=tolerance)

    def generate_report(self, matrix: Union[np.ndarray, sparse.spmatrix]) -> ValidationReport:
        """Runs full suite of analyses and compiles the ValidationReport."""
        
        # Dimensions & Quality
        dims = list(matrix.shape)
        is_square = (dims[0] == dims[1])
        quality_warnings = MatrixQualityAnalyzer.find_invalid_entries(matrix)
        is_finite = MatrixQualityAnalyzer.check_finiteness(matrix)
        
        # Hermiticity
        is_herm, deviation = self.herm_analyzer.is_hermitian(matrix)
        if not is_herm and is_square:
            quality_warnings.append(f"Matrix is non-Hermitian. Deviation: {deviation:.2e}")

        # Spectral Properties
        trace = SpectralAnalyzer.calculate_trace(matrix) if is_square else 0.0
        norm = SpectralAnalyzer.calculate_norm(matrix)
        
        dense_props = SpectralAnalyzer.analyze_dense_properties(matrix) if is_square else {}
        
        # Final validity gate
        is_valid = is_square and is_herm and is_finite and (len(quality_warnings) == 0)
        
        return ValidationReport(
            dimensions=dims,
            is_square=is_square,
            is_hermitian=is_herm,
            hermiticity_deviation=deviation,
            trace=trace,
            norm=norm,
            rank=dense_props.get("rank"),
            determinant=dense_props.get("determinant"),
            is_finite=is_finite,
            warnings=quality_warnings,
            is_valid=is_valid
        )
