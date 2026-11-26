from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence
import numpy as np
from .perturbation_operator import PerturbationMatrix
from .non_degenerate import NonDegeneratePerturbationSolver


@dataclass
class LambdaScanPoint:
    """Comparison of 1st-order, 2nd-order, and exact numerical energies at strength lambda."""
    lambda_strength: float
    exact_numerical_energy: float
    first_order_energy: float
    second_order_energy: float
    first_order_abs_error: float
    second_order_abs_error: float
    first_order_rel_error: Optional[float]
    second_order_rel_error: Optional[float]
    first_order_shift_magnitude: float
    second_order_shift_magnitude: float
    second_order_improves_over_first: bool
    is_perturbatively_valid: bool


@dataclass
class PerturbationConvergenceReport:
    """Summary of perturbation accuracy and breakdown behavior across lambda values."""
    state_index: int
    unperturbed_energy: float
    points: List[LambdaScanPoint]
    estimated_second_order_scaling: Optional[float]
    breakdown_lambda: Optional[float]
    warnings: List[str]


class PerturbationConvergenceAnalyzer:
    """
    Scans perturbation strength lambda to compare 1st-order vs 2nd-order
    Rayleigh-Schrödinger approximations against exact numerical eigenvalues
    of H_0 + lambda * H', identifying series convergence and breakdown regimes.
    """

    def __init__(self, solver: Optional[NonDegeneratePerturbationSolver] = None):
        self.solver = solver if solver is not None else NonDegeneratePerturbationSolver()

    @staticmethod
    def _safe_rel_error(approx: float, ref: float, zero_threshold: float = 1e-12) -> Optional[float]:
        abs_err = abs(approx - ref)
        if abs(ref) < zero_threshold:
            return 0.0 if abs_err < zero_threshold else None
        return float(abs_err / abs(ref))

    def scan_lambda(
        self,
        state_index: int,
        unperturbed_energies: Sequence[float],
        perturbation_matrix: PerturbationMatrix,
        lambda_values: Sequence[float],
        exact_energy_fn: Callable[[float], float]
    ) -> PerturbationConvergenceReport:
        """
        Evaluates 1st- and 2nd-order approximations against `exact_energy_fn(lam)`
        across all `lambda_values`.
        """
        if not lambda_values:
            raise ValueError("lambda_values must be non-empty.")

        sorted_lams = sorted([float(v) for v in lambda_values], key=lambda x: abs(x))
        e0 = float(unperturbed_energies[state_index])
        points: List[LambdaScanPoint] = []
        warnings: List[str] = []
        breakdown_lam: Optional[float] = None

        for lam in sorted_lams:
            r1 = self.solver.solve_state(
                state_index, unperturbed_energies, perturbation_matrix, lambda_strength=lam, order=1
            )
            r2 = self.solver.solve_state(
                state_index, unperturbed_energies, perturbation_matrix, lambda_strength=lam, order=2
            )
            e_exact = float(exact_energy_fn(lam))

            err1 = abs(r1.total_approximate_energy - e_exact)
            err2 = abs(r2.total_approximate_energy - e_exact)
            rel1 = self._safe_rel_error(r1.total_approximate_energy, e_exact)
            rel2 = self._safe_rel_error(r2.total_approximate_energy, e_exact)
            improves = err2 <= err1 + 1e-14

            if breakdown_lam is None and abs(lam) > 0 and (not r2.is_perturbatively_valid or not improves):
                breakdown_lam = lam

            points.append(
                LambdaScanPoint(
                    lambda_strength=lam,
                    exact_numerical_energy=e_exact,
                    first_order_energy=r1.total_approximate_energy,
                    second_order_energy=r2.total_approximate_energy,
                    first_order_abs_error=float(err1),
                    second_order_abs_error=float(err2),
                    first_order_rel_error=rel1,
                    second_order_rel_error=rel2,
                    first_order_shift_magnitude=float(abs(lam * r1.first_order_correction)),
                    second_order_shift_magnitude=float(abs((lam ** 2) * r2.second_order_correction)),
                    second_order_improves_over_first=improves,
                    is_perturbatively_valid=r2.is_perturbatively_valid
                )
            )

        # Estimate empirical scaling exponent of 2nd-order residual error vs |lambda| (expect ~3)
        pos_lams = np.array([abs(p.lambda_strength) for p in points if abs(p.lambda_strength) > 0 and p.second_order_abs_error > 1e-13])
        pos_errs = np.array([p.second_order_abs_error for p in points if abs(p.lambda_strength) > 0 and p.second_order_abs_error > 1e-13])

        scaling: Optional[float] = None
        if len(pos_lams) >= 2:
            # Use smallest 3 non-zero points for asymptotic small-lambda slope
            k_fit = min(len(pos_lams), 3)
            slope, _ = np.polyfit(np.log(pos_lams[:k_fit]), np.log(pos_errs[:k_fit]), 1)
            scaling = float(slope)

        if breakdown_lam is not None:
            warnings.append(
                f"Perturbative validity or 2nd-order improvement broke down starting at lambda={breakdown_lam:.4g}."
            )

        return PerturbationConvergenceReport(
            state_index=int(state_index),
            unperturbed_energy=e0,
            points=points,
            estimated_second_order_scaling=scaling,
            breakdown_lambda=breakdown_lam,
            warnings=warnings
        )
