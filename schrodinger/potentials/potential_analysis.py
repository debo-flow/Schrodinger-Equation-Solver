from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple
import numpy as np
from schrodinger.numerical.grid import SpatialGrid
from .base import Potential


SymmetryType = Literal["even", "odd", "asymmetric"]


@dataclass
class ExtremumPoint:
    """Represents a local or global extremum of V(x)."""
    position: float
    value: float
    kind: Literal["minimum", "maximum"]


@dataclass
class PotentialAnalysisReport:
    """Structured analysis summary for a quantum potential over a spatial grid."""
    global_minimum: ExtremumPoint
    global_maximum: ExtremumPoint
    local_minima: List[ExtremumPoint]
    local_maxima: List[ExtremumPoint]
    symmetry: SymmetryType
    symmetry_center: float
    finite_min: Optional[float]
    finite_max: Optional[float]
    finite_span: Optional[float]
    has_infinite_walls: bool
    is_continuous: bool
    discontinuity_positions: List[float]
    max_step_jump: float


class PotentialAnalyzer:
    """
    Analyzes geometric, topological, and numerical properties of quantum
    potentials evaluated across a discretized SpatialGrid.
    """

    @staticmethod
    def find_global_extrema(grid: SpatialGrid, potential: Potential) -> Tuple[ExtremumPoint, ExtremumPoint]:
        """Finds global minimum and global maximum across the spatial grid."""
        x = grid.grid
        v = potential(x)

        min_idx = int(np.argmin(v))
        max_idx = int(np.argmax(v))

        g_min = ExtremumPoint(position=float(x[min_idx]), value=float(v[min_idx]), kind="minimum")
        g_max = ExtremumPoint(position=float(x[max_idx]), value=float(v[max_idx]), kind="maximum")
        return g_min, g_max

    @staticmethod
    def find_local_extrema(
        grid: SpatialGrid,
        potential: Potential,
        tolerance: float = 1e-12
    ) -> Tuple[List[ExtremumPoint], List[ExtremumPoint]]:
        """
        Identifies interior local minima and local maxima on the discrete grid.
        """
        x = grid.grid
        v = potential(x)

        local_mins: List[ExtremumPoint] = []
        local_maxs: List[ExtremumPoint] = []

        for i in range(1, len(v) - 1):
            if not (np.isfinite(v[i - 1]) and np.isfinite(v[i]) and np.isfinite(v[i + 1])):
                continue
            left_diff = v[i] - v[i - 1]
            right_diff = v[i] - v[i + 1]

            if left_diff < -tolerance and right_diff <= tolerance:
                local_mins.append(ExtremumPoint(position=float(x[i]), value=float(v[i]), kind="minimum"))
            elif left_diff > tolerance and right_diff >= -tolerance:
                local_maxs.append(ExtremumPoint(position=float(x[i]), value=float(v[i]), kind="maximum"))

        return local_mins, local_maxs

    @staticmethod
    def classify_symmetry(
        grid: SpatialGrid,
        potential: Potential,
        center: Optional[float] = None,
        tolerance: float = 1e-7
    ) -> Tuple[SymmetryType, float]:
        """
        Classifies potential parity around `center` as 'even', 'odd', or 'asymmetric'.
        """
        sym_center = (
            float(center)
            if center is not None
            else float(getattr(potential, "center", 0.5 * (grid.x_min + grid.x_max)))
        )
        half_span = max(abs(grid.x_max - sym_center), abs(sym_center - grid.x_min))
        if half_span == 0.0:
            return "even", sym_center

        # Evaluate on an exact symmetric probe grid around sym_center
        offsets = np.linspace(0.0, half_span, max(grid.points // 2, 50))
        v_pos = potential(sym_center + offsets)
        v_neg = potential(sym_center - offsets)

        inf_pos = np.isinf(v_pos)
        inf_neg = np.isinf(v_neg)
        if not np.array_equal(inf_pos, inf_neg):
            return "asymmetric", sym_center

        finite_mask = np.isfinite(v_pos) & np.isfinite(v_neg)
        if not np.any(finite_mask):
            # All points +inf symmetrically -> even
            return ("even" if np.array_equal(v_pos, v_neg) else "asymmetric"), sym_center

        vp = v_pos[finite_mask]
        vn = v_neg[finite_mask]
        scale = max(1.0, float(np.max(np.abs(vp))))

        is_even = bool(np.all(np.abs(vp - vn) <= tolerance * scale))
        if is_even:
            return "even", sym_center

        is_odd = bool(np.all(np.abs(vp + vn) <= tolerance * scale))
        if is_odd:
            return "odd", sym_center

        return "asymmetric", sym_center

    @staticmethod
    def estimate_range(grid: SpatialGrid, potential: Potential) -> Tuple[Optional[float], Optional[float], bool]:
        """
        Returns (finite_min, finite_max, has_infinite_walls) across the grid.
        """
        v = potential(grid.grid)
        has_inf = bool(np.any(np.isinf(v)))
        finite_v = v[np.isfinite(v)]
        if len(finite_v) == 0:
            return None, None, has_inf
        return float(np.min(finite_v)), float(np.max(finite_v)), has_inf

    @staticmethod
    def check_continuity(
        grid: SpatialGrid,
        potential: Potential,
        jump_factor: float = 10.0,
        abs_jump_threshold: float = 1e-2
    ) -> Tuple[bool, List[float], float]:
        """
        Detects numerical jump discontinuities by comparing adjacent finite differences
        against the median interior step variation.

        Returns:
            (is_continuous, discontinuity_positions, max_step_jump)
        """
        x = grid.grid
        v = potential(x)

        discontinuities: List[float] = []
        finite_mask = np.isfinite(v[:-1]) & np.isfinite(v[1:])

        # Infinite-to-finite wall transitions are discontinuities
        inf_transitions = np.where(np.isinf(v[:-1]) != np.isinf(v[1:]))[0]
        for idx in inf_transitions:
            discontinuities.append(float(0.5 * (x[idx] + x[idx + 1])))

        if not np.any(finite_mask):
            return (len(discontinuities) == 0), sorted(discontinuities), float("inf")

        diffs = np.abs(v[1:][finite_mask] - v[:-1][finite_mask])
        if len(diffs) == 0:
            return (len(discontinuities) == 0), sorted(discontinuities), 0.0

        max_jump = float(np.max(diffs))
        median_step = float(np.median(diffs))
        threshold = max(abs_jump_threshold, jump_factor * median_step)

        finite_indices = np.where(finite_mask)[0]
        for local_i, diff_val in enumerate(diffs):
            if diff_val > threshold:
                idx = finite_indices[local_i]
                discontinuities.append(float(0.5 * (x[idx] + x[idx + 1])))

        discontinuities = sorted(set(discontinuities))
        return (len(discontinuities) == 0), discontinuities, max_jump

    @classmethod
    def analyze(
        cls,
        grid: SpatialGrid,
        potential: Potential,
        symmetry_center: Optional[float] = None
    ) -> PotentialAnalysisReport:
        """Executes full geometric and numerical analysis of a potential on `grid`."""
        g_min, g_max = cls.find_global_extrema(grid, potential)
        local_mins, local_maxs = cls.find_local_extrema(grid, potential)
        sym, sym_center = cls.classify_symmetry(grid, potential, center=symmetry_center)
        f_min, f_max, has_inf = cls.estimate_range(grid, potential)
        f_span = (f_max - f_min) if (f_min is not None and f_max is not None) else None
        is_cont, disc_pos, max_jump = cls.check_continuity(grid, potential)

        return PotentialAnalysisReport(
            global_minimum=g_min,
            global_maximum=g_max,
            local_minima=local_mins,
            local_maxima=local_maxs,
            symmetry=sym,
            symmetry_center=sym_center,
            finite_min=f_min,
            finite_max=f_max,
            finite_span=f_span,
            has_infinite_walls=has_inf,
            is_continuous=is_cont,
            discontinuity_positions=disc_pos,
            max_step_jump=max_jump
        )
