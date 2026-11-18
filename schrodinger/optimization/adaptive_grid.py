from dataclasses import dataclass
from typing import Callable, List, Optional
import numpy as np
from schrodinger.numerical.grid import SpatialGrid, BoundaryCondition
from schrodinger.potentials.base import Potential
from schrodinger.wavefunctions.wavefunction import Wavefunction


@dataclass
class AdaptiveGridConfig:
    """Hard bounds preventing uncontrolled grid refinement."""
    min_points: int = 50
    max_points: int = 4000
    max_refinement_steps: int = 5
    refinement_factor: float = 1.5
    max_wavefunction_step: float = 0.15
    min_points_per_oscillation: int = 20
    boundary_leakage_tolerance: float = 1e-3
    energy_convergence_tolerance: float = 1e-4

    def __post_init__(self) -> None:
        if self.min_points < 5:
            raise ValueError("min_points must be >= 5.")
        if self.max_points < self.min_points:
            raise ValueError("max_points must be >= min_points.")
        if self.max_refinement_steps < 1:
            raise ValueError("max_refinement_steps must be >= 1.")
        if self.refinement_factor <= 1.0:
            raise ValueError("refinement_factor must be strictly greater than 1.0.")


@dataclass
class GridAssessment:
    """Quantitative assessment of spatial grid adequacy."""
    current_points: int
    dx: float
    potential_variation_score: float
    max_wavefunction_variation: float
    min_points_per_oscillation: Optional[float]
    max_boundary_amplitude: float
    needs_refinement: bool
    needs_domain_expansion: bool
    reasons: List[str]


@dataclass
class RefinementRecommendation:
    """Bounded recommendation for adjusting spatial grid resolution."""
    recommended_points: int
    recommended_x_min: float
    recommended_x_max: float
    hit_max_points_cap: bool
    assessment: GridAssessment


@dataclass
class AdaptiveRefinementResult:
    """Result of a bounded iterative grid refinement convergence run."""
    final_grid: SpatialGrid
    final_wavefunctions: List[Wavefunction]
    iterations: int
    point_history: List[int]
    energy_delta_history: List[float]
    converged: bool
    stopped_due_to_limit: bool


class AdaptiveGridFramework:
    """
    Extensible adaptive spatial grid analyzer and bounded refinement controller.
    Evaluates potential steepness, wavefunction oscillation resolution, and
    iterative eigenvalue convergence without allowing runaway refinement.
    """

    def __init__(self, config: AdaptiveGridConfig = AdaptiveGridConfig()):
        self.config = config

    def analyze_potential_variation(self, grid: SpatialGrid, potential: Potential) -> float:
        """
        Measures normalized potential roughness over finite interior values.
        Returns a dimensionless score in [0, inf); values > 0.25 indicate sharp variation.
        """
        v_vals = potential(grid.interior_points)
        finite_v = v_vals[np.isfinite(v_vals)]
        if len(finite_v) < 3:
            return 0.0

        v_span = float(np.max(finite_v) - np.min(finite_v))
        if v_span < 1e-12:
            return 0.0

        max_step = float(np.max(np.abs(np.diff(finite_v))))
        return max_step / v_span

    def analyze_wavefunctions(
        self,
        grid: SpatialGrid,
        wavefunctions: List[Wavefunction]
    ) -> tuple:
        """
        Evaluates point-to-point amplitude variation, oscillation resolution,
        and near-boundary amplitude decay across computed eigenstates.
        """
        if not wavefunctions:
            return 0.0, None, 0.0

        max_step = 0.0
        min_ppo: Optional[float] = None
        max_boundary_amp = 0.0

        for wf in wavefunctions:
            psi_abs = np.abs(wf.psi)
            peak = float(np.max(psi_abs))
            if peak > 0:
                norm_amp = psi_abs / peak
                step = float(np.max(np.abs(np.diff(norm_amp))))
                max_step = max(max_step, step)

                # Check interior-adjacent boundary decay (index 1 and -2 for Dirichlet, 0 and -1 otherwise)
                if grid.boundary_condition == BoundaryCondition.DIRICHLET and len(norm_amp) > 4:
                    edge_amp = max(float(norm_amp[1]), float(norm_amp[-2]))
                else:
                    edge_amp = max(float(norm_amp[0]), float(norm_amp[-1]))
                max_boundary_amp = max(max_boundary_amp, edge_amp)

            nodes = wf.nodes
            half_waves = max(1, nodes + 1)
            ppo = float(grid.points) / float(half_waves)
            min_ppo = ppo if min_ppo is None else min(min_ppo, ppo)

        return max_step, min_ppo, max_boundary_amp

    def assess_grid(
        self,
        grid: SpatialGrid,
        potential: Potential,
        wavefunctions: Optional[List[Wavefunction]] = None
    ) -> GridAssessment:
        """Assesses whether the current grid satisfies accuracy heuristics."""
        pot_var = self.analyze_potential_variation(grid, potential)
        wf_var, min_ppo, max_edge = self.analyze_wavefunctions(grid, wavefunctions or [])

        reasons: List[str] = []
        needs_refinement = False
        needs_expansion = False

        if grid.points < self.config.min_points:
            needs_refinement = True
            reasons.append(f"Grid points ({grid.points}) below minimum ({self.config.min_points}).")

        if pot_var > 0.25:
            needs_refinement = True
            reasons.append(f"Sharp potential variation detected (normalized max step={pot_var:.3f}).")

        if wf_var > self.config.max_wavefunction_step:
            needs_refinement = True
            reasons.append(
                f"Wavefunction amplitude changes too rapidly per grid step ({wf_var:.3f} > "
                f"{self.config.max_wavefunction_step:.3f})."
            )

        if min_ppo is not None and min_ppo < self.config.min_points_per_oscillation:
            needs_refinement = True
            reasons.append(
                f"Insufficient points per half-oscillation ({min_ppo:.1f} < "
                f"{self.config.min_points_per_oscillation})."
            )

        # Domain expansion is only suggested when potential is finite at edges (not infinite well)
        edge_v = potential(np.array([grid.x_min, grid.x_max]))
        if np.all(np.isfinite(edge_v)) and max_edge > self.config.boundary_leakage_tolerance:
            # Ignore pure free particle / flat potential where wave naturally spans domain
            if pot_var > 1e-6:
                needs_expansion = True
                reasons.append(
                    f"Wavefunction amplitude near domain edge ({max_edge:.2e}) exceeds "
                    f"tolerance ({self.config.boundary_leakage_tolerance:.2e})."
                )

        return GridAssessment(
            current_points=grid.points,
            dx=grid.dx,
            potential_variation_score=pot_var,
            max_wavefunction_variation=wf_var,
            min_points_per_oscillation=min_ppo,
            max_boundary_amplitude=max_edge,
            needs_refinement=needs_refinement,
            needs_domain_expansion=needs_expansion,
            reasons=reasons
        )

    def recommend_refinement(
        self,
        grid: SpatialGrid,
        potential: Potential,
        wavefunctions: Optional[List[Wavefunction]] = None
    ) -> RefinementRecommendation:
        """Produces a bounded refinement recommendation without mutating the original grid."""
        assessment = self.assess_grid(grid, potential, wavefunctions)

        target_points = grid.points
        if assessment.needs_refinement:
            scaled = int(np.ceil(grid.points * self.config.refinement_factor))
            target_points = max(self.config.min_points, scaled)

        hit_cap = target_points >= self.config.max_points
        recommended_points = min(target_points, self.config.max_points)

        rec_x_min, rec_x_max = grid.x_min, grid.x_max
        if assessment.needs_domain_expansion:
            span = grid.x_max - grid.x_min
            rec_x_min -= 0.15 * span
            rec_x_max += 0.15 * span

        return RefinementRecommendation(
            recommended_points=recommended_points,
            recommended_x_min=rec_x_min,
            recommended_x_max=rec_x_max,
            hit_max_points_cap=hit_cap,
            assessment=assessment
        )

    def refine_until_converged(
        self,
        initial_grid: SpatialGrid,
        potential: Potential,
        solve_fn: Callable[[SpatialGrid], List[Wavefunction]]
    ) -> AdaptiveRefinementResult:
        """
        Iteratively refines the grid while monitoring energy convergence,
        strictly respecting `max_refinement_steps` and `max_points`.
        """
        current_grid = initial_grid
        current_wfs = solve_fn(current_grid)
        point_history = [current_grid.points]
        delta_history: List[float] = []

        converged = False
        stopped_due_to_limit = False

        for _ in range(self.config.max_refinement_steps):
            rec = self.recommend_refinement(current_grid, potential, current_wfs)

            # Determine next point count (either from heuristic or convergence verification step)
            next_points = rec.recommended_points
            if not rec.assessment.needs_refinement and len(delta_history) == 0:
                next_points = min(
                    self.config.max_points,
                    int(np.ceil(current_grid.points * self.config.refinement_factor))
                )

            if next_points <= current_grid.points:
                stopped_due_to_limit = rec.hit_max_points_cap
                break

            next_grid = SpatialGrid(
                x_min=current_grid.x_min,
                x_max=current_grid.x_max,
                points=next_points,
                boundary_condition=current_grid.boundary_condition
            )
            next_wfs = solve_fn(next_grid)

            prev_e = np.array([wf.energy for wf in current_wfs], dtype=np.float64)
            next_e = np.array([wf.energy for wf in next_wfs], dtype=np.float64)
            max_delta = float(np.max(np.abs(next_e - prev_e)))

            current_grid = next_grid
            current_wfs = next_wfs
            point_history.append(current_grid.points)
            delta_history.append(max_delta)

            post_assess = self.assess_grid(current_grid, potential, current_wfs)
            if max_delta <= self.config.energy_convergence_tolerance and not post_assess.needs_refinement:
                converged = True
                break
        else:
            stopped_due_to_limit = True

        return AdaptiveRefinementResult(
            final_grid=current_grid,
            final_wavefunctions=current_wfs,
            iterations=len(delta_history),
            point_history=point_history,
            energy_delta_history=delta_history,
            converged=converged,
            stopped_due_to_limit=stopped_due_to_limit
        )
