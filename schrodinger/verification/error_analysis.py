from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union
import numpy as np
from scipy import integrate


@dataclass
class ErrorSummary:
    """Structured container for scalar and array error statistics."""
    max_absolute_error: float
    rmse: float
    mean_absolute_error: float
    max_relative_error: Optional[float]
    mean_relative_error: Optional[float]


class ErrorAnalyzer:
    """
    Provides numerically stable error metrics for comparing numerical results
    against analytical reference values.
    """

    @staticmethod
    def _validate_shapes(numerical: np.ndarray, analytical: np.ndarray) -> None:
        if numerical.shape != analytical.shape:
            raise ValueError(
                f"Shape mismatch: numerical {numerical.shape} != analytical {analytical.shape}"
            )
        if not (np.all(np.isfinite(numerical)) and np.all(np.isfinite(analytical))):
            raise ValueError("Inputs to ErrorAnalyzer must contain strictly finite values.")

    @staticmethod
    def absolute_error(
        numerical: Union[float, np.ndarray],
        analytical: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Computes |numerical - analytical|."""
        if np.isscalar(numerical) and np.isscalar(analytical):
            return float(abs(numerical - analytical))
        num_arr = np.asarray(numerical)
        ana_arr = np.asarray(analytical)
        ErrorAnalyzer._validate_shapes(num_arr, ana_arr)
        return np.abs(num_arr - ana_arr)

    @staticmethod
    def relative_error(
        numerical: Union[float, np.ndarray],
        analytical: Union[float, np.ndarray],
        zero_threshold: float = 1e-12
    ) -> Union[Optional[float], np.ndarray]:
        """
        Computes |numerical - analytical| / |analytical| safely.

        Near-zero reference handling:
            - Scalar inputs: returns 0.0 if both error and reference are below zero_threshold,
              otherwise returns None when |analytical| < zero_threshold.
            - Array inputs: assigns 0.0 where both error and reference are near zero,
              and np.nan where |analytical| < zero_threshold but error is non-negligible.
        """
        if zero_threshold <= 0:
            raise ValueError("zero_threshold must be strictly positive.")

        if np.isscalar(numerical) and np.isscalar(analytical):
            abs_err = abs(numerical - analytical)
            ref_mag = abs(analytical)
            if ref_mag < zero_threshold:
                return 0.0 if abs_err < zero_threshold else None
            return float(abs_err / ref_mag)

        num_arr = np.asarray(numerical)
        ana_arr = np.asarray(analytical)
        ErrorAnalyzer._validate_shapes(num_arr, ana_arr)

        abs_err = np.abs(num_arr - ana_arr)
        ref_mag = np.abs(ana_arr)

        rel = np.full(abs_err.shape, np.nan, dtype=np.float64)
        valid_mask = ref_mag >= zero_threshold
        both_zero_mask = (~valid_mask) & (abs_err < zero_threshold)

        rel[valid_mask] = abs_err[valid_mask] / ref_mag[valid_mask]
        rel[both_zero_mask] = 0.0
        return rel

    @staticmethod
    def rmse(numerical: np.ndarray, analytical: np.ndarray) -> float:
        """Computes the Root Mean Square Error."""
        num_arr = np.asarray(numerical)
        ana_arr = np.asarray(analytical)
        ErrorAnalyzer._validate_shapes(num_arr, ana_arr)
        return float(np.sqrt(np.mean(np.abs(num_arr - ana_arr) ** 2)))

    @staticmethod
    def max_absolute_error(numerical: np.ndarray, analytical: np.ndarray) -> float:
        """Computes the infinity norm max(|numerical - analytical|)."""
        num_arr = np.asarray(numerical)
        ana_arr = np.asarray(analytical)
        ErrorAnalyzer._validate_shapes(num_arr, ana_arr)
        return float(np.max(np.abs(num_arr - ana_arr)))

    @staticmethod
    def l2_wavefunction_error(
        psi_numerical: np.ndarray,
        psi_analytical: np.ndarray,
        x: np.ndarray
    ) -> float:
        """
        Computes the continuous L2 norm error:
            sqrt( integral |psi_num(x) - psi_ana(x)|^2 dx )
        """
        num_arr = np.asarray(psi_numerical)
        ana_arr = np.asarray(psi_analytical)
        x_arr = np.asarray(x, dtype=np.float64)
        ErrorAnalyzer._validate_shapes(num_arr, ana_arr)
        if len(x_arr) != len(num_arr):
            raise ValueError("Spatial grid length must match wavefunction length.")

        integrand = np.abs(num_arr - ana_arr) ** 2
        integral = float(integrate.simpson(y=integrand, x=x_arr))
        return float(np.sqrt(max(0.0, integral)))

    @staticmethod
    def energy_spectrum_deviation(
        numerical_energies: np.ndarray,
        analytical_energies: np.ndarray,
        zero_threshold: float = 1e-12
    ) -> Dict[str, Any]:
        """
        Computes detailed per-level and aggregate deviations between energy spectra.
        """
        e_num = np.asarray(numerical_energies, dtype=np.float64)
        e_ana = np.asarray(analytical_energies, dtype=np.float64)
        ErrorAnalyzer._validate_shapes(e_num, e_ana)

        abs_errors = ErrorAnalyzer.absolute_error(e_num, e_ana)
        rel_errors_list: List[Optional[float]] = [
            ErrorAnalyzer.relative_error(float(en), float(ea), zero_threshold=zero_threshold)
            for en, ea in zip(e_num, e_ana)
        ]
        valid_rel = [r for r in rel_errors_list if r is not None]

        return {
            "absolute_errors": [float(v) for v in abs_errors],
            "relative_errors": rel_errors_list,
            "max_absolute_error": float(np.max(abs_errors)),
            "rmse": ErrorAnalyzer.rmse(e_num, e_ana),
            "max_relative_error": float(max(valid_rel)) if valid_rel else None,
            "mean_relative_error": float(np.mean(valid_rel)) if valid_rel else None
        }

    @staticmethod
    def summarize_array_errors(
        numerical: np.ndarray,
        analytical: np.ndarray,
        zero_threshold: float = 1e-12
    ) -> ErrorSummary:
        """Returns an ErrorSummary dataclass for two arrays."""
        abs_err = ErrorAnalyzer.absolute_error(numerical, analytical)
        rel_err = ErrorAnalyzer.relative_error(numerical, analytical, zero_threshold=zero_threshold)
        finite_rel = rel_err[np.isfinite(rel_err)]

        return ErrorSummary(
            max_absolute_error=float(np.max(abs_err)),
            rmse=ErrorAnalyzer.rmse(numerical, analytical),
            mean_absolute_error=float(np.mean(abs_err)),
            max_relative_error=float(np.max(finite_rel)) if len(finite_rel) > 0 else None,
            mean_relative_error=float(np.mean(finite_rel)) if len(finite_rel) > 0 else None
        )
