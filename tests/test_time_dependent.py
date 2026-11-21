import json
import math
import pytest
import numpy as np

from schrodinger.numerical.grid import SpatialGrid, BoundaryCondition
from schrodinger.potentials.models import FreeParticle, HarmonicOscillator, InfiniteSquareWell
from schrodinger.potentials.time_dependent import (
    LinearlyDrivenPotential,
    ModulatedPotential,
    CallableTimeDependentPotential
)
from schrodinger.optimization import OptimizationEngine, OptimizationConfig
from schrodinger.verification.error_analysis import ErrorAnalyzer

from schrodinger.dynamics import (
    InitialStateBuilder,
    IntegratorType,
    RK4Integrator,
    CrankNicolsonIntegrator,
    SplitOperatorIntegrator,
    TimeObservablesAnalyzer,
    EvolutionValidator,
    TimeEvolutionConfig,
    TimeEvolutionEngine,
    TimeDependentSchrodingerEngine
)


def test_stationary_eigenstate_evolution():
    """Verify stationary eigenstates evolve purely by global phase exp(-i*E*t/hbar) with constant density."""
    grid = SpatialGrid(x_min=-7.0, x_max=7.0, points=300, boundary_condition=BoundaryCondition.DIRICHLET)
    ho = HarmonicOscillator(mass=1.0, omega=1.0)

    opt = OptimizationEngine(OptimizationConfig(enable_cache=False, monitor_performance=False))
    eigenstates = opt.solve_system(grid, ho, num_eigenvalues=2).wavefunctions
    psi_1 = eigenstates[1]  # First excited state (E_1 ≈ 1.5)

    t_end = 1.2
    engine = TimeDependentSchrodingerEngine(mass=1.0, hbar=1.0)
    res = engine.propagate(
        grid, ho, psi_1, t_end=t_end, dt=1e-3, method=IntegratorType.CRANK_NICOLSON, save_every=20
    )

    assert res.validation.is_valid is True
    assert res.validation.norm_report.max_norm_deviation < 1e-10

    expected_psi = EvolutionValidator.analytical_stationary_state(psi_1.psi, psi_1.energy, t_end)
    l2_err = ErrorAnalyzer.l2_wavefunction_error(res.evolution.final_state.psi, expected_psi, grid.grid)
    assert l2_err < 1e-3

    # Probability density must remain invariant in time
    rho_init = np.abs(psi_1.psi) ** 2
    rho_final = res.evolution.final_state.probability_density
    np.testing.assert_allclose(rho_final, rho_init, atol=1e-5)


def test_free_particle_gaussian_wavepacket_against_analytical():
    """Compare free-particle wavepacket propagation & dispersion across all three integrators."""
    grid = SpatialGrid(x_min=-15.0, x_max=15.0, points=512, boundary_condition=BoundaryCondition.DIRICHLET)
    fp = FreeParticle()

    x0 = -3.0
    sigma = 0.8
    k0 = 3.0
    t_end = 1.0
    init_wf = InitialStateBuilder.gaussian_wavepacket(grid, x0=x0, sigma=sigma, k0=k0)
    exact_final = EvolutionValidator.analytical_free_gaussian_wavepacket(
        grid.grid, t=t_end, x0=x0, sigma=sigma, k0=k0
    )

    engine = TimeDependentSchrodingerEngine(mass=1.0, hbar=1.0)
    for method, dt in [
        (IntegratorType.SPLIT_OPERATOR, 2e-3),
        (IntegratorType.CRANK_NICOLSON, 1e-3),
        (IntegratorType.RK4, 5e-4)
    ]:
        res = engine.propagate(
            grid, fp, init_wf, t_end=t_end, dt=dt, method=method, save_every=50
        )
        assert res.validation.is_valid is True
        l2_err = ErrorAnalyzer.l2_wavefunction_error(res.evolution.final_state.psi, exact_final, grid.grid)
        assert l2_err < 2e-2

        # Group velocity check: <x(t_end)> ≈ x0 + (hbar * k0 / m) * t_end = 0.0
        assert res.observables is not None
        assert np.isclose(res.observables.expectation_x[-1], x0 + k0 * t_end, atol=2e-2)
        assert np.isclose(res.observables.expectation_p[-1], k0, atol=2e-2)


def test_harmonic_oscillator_coherent_state_dynamics():
    """Verify QHO coherent state reproduces analytical oscillating <x(t)>, <p(t)>, and complex phase."""
    grid = SpatialGrid(x_min=-10.0, x_max=10.0, points=400, boundary_condition=BoundaryCondition.DIRICHLET)
    omega = 1.2
    ho = HarmonicOscillator(mass=1.0, omega=omega, center=0.0)

    alpha = 1.5 + 0.5j
    t_end = 1.5
    init_wf = InitialStateBuilder.coherent_state(grid, alpha=alpha, mass=1.0, omega=omega)
    exact_final = EvolutionValidator.analytical_coherent_oscillator_state(
        grid.grid, t=t_end, alpha=alpha, mass=1.0, omega=omega
    )

    engine = TimeDependentSchrodingerEngine(mass=1.0, hbar=1.0)
    res = engine.propagate(
        grid, ho, init_wf, t_end=t_end, dt=1e-3, method=IntegratorType.SPLIT_OPERATOR, save_every=25
    )

    assert res.validation.is_valid is True
    assert res.validation.norm_report.max_norm_deviation < 1e-12
    assert res.validation.max_energy_drift is not None and res.validation.max_energy_drift < 1e-3

    # Full complex wavefunction (including global phase) must match analytical coherent state
    l2_err = ErrorAnalyzer.l2_wavefunction_error(res.evolution.final_state.psi, exact_final, grid.grid)
    assert l2_err < 5e-3


def test_superposition_state_beating_dynamics():
    """Verify two-state superposition oscillates <x(t)> at Bohr transition frequency (E1 - E0)/hbar."""
    grid = SpatialGrid(x_min=0.0, x_max=math.pi, points=200, boundary_condition=BoundaryCondition.DIRICHLET)
    isw = InfiniteSquareWell(width=math.pi, center=0.5 * math.pi)

    opt = OptimizationEngine(OptimizationConfig(enable_cache=False, monitor_performance=False))
    states = opt.solve_system(grid, isw, num_eigenvalues=2).wavefunctions

    sup_wf = InitialStateBuilder.superposition_state(states, [1.0, 1.0])
    # Bohr frequency omega_10 = E_1 - E_0 = 2.0 - 0.5 = 1.5 -> half-period T/2 = pi / 1.5
    t_half = math.pi / (states[1].energy - states[0].energy)

    engine = TimeDependentSchrodingerEngine(mass=1.0, hbar=1.0)
    res = engine.propagate(
        grid, isw, sup_wf, t_end=t_half, dt=5e-4, method=IntegratorType.CRANK_NICOLSON, save_every=20
    )

    assert res.observables is not None
    x_init = res.observables.expectation_x[0]
    x_final = res.observables.expectation_x[-1]
    # After half a Bohr period, expectation position flips symmetrically across pi/2
    assert np.isclose((x_init + x_final) / 2.0, 0.5 * math.pi, atol=1e-2)


def test_time_dependent_potential_and_energy_transfer():
    """Verify driven time-dependent potential V(x, t) preserves norm while pumping energy."""
    grid = SpatialGrid(x_min=-8.0, x_max=8.0, points=300, boundary_condition=BoundaryCondition.DIRICHLET)
    base_ho = HarmonicOscillator(mass=1.0, omega=1.0)
    driven_pot = LinearlyDrivenPotential(base_ho, drive_amplitude=0.4, drive_frequency=1.0)

    # Verify V(x, t) snapshot compatibility with Potential base class
    slice_t = driven_pot.at_time(0.5)
    np.testing.assert_allclose(slice_t(grid.grid), driven_pot(grid.grid, 0.5))

    init_wf = InitialStateBuilder.coherent_state(grid, alpha=0.0 + 0.0j, mass=1.0, omega=1.0)
    engine = TimeDependentSchrodingerEngine(mass=1.0, hbar=1.0)

    for method in (IntegratorType.CRANK_NICOLSON, IntegratorType.SPLIT_OPERATOR, IntegratorType.RK4):
        res = engine.propagate(
            grid, driven_pot, init_wf, t_end=2.0, dt=1e-3, method=method, save_every=50
        )
        assert res.validation.norm_report.is_conserved is True
        assert res.observables is not None
        # Resonant dipole drive must excite the ground state and displace <x(t)>
        assert res.observables.total_energy[-1] > res.observables.total_energy[0] + 0.05
        assert np.max(np.abs(res.observables.expectation_x)) > 0.1


def test_timestep_convergence_orders():
    """Verify RK4 exhibits ~4th-order convergence and Crank-Nicolson / Split-Operator exhibit ~2nd-order."""
    grid = SpatialGrid(x_min=-10.0, x_max=10.0, points=256, boundary_condition=BoundaryCondition.DIRICHLET)
    ho = HarmonicOscillator(mass=1.0, omega=1.0)
    init_wf = InitialStateBuilder.coherent_state(grid, alpha=1.0 + 0.0j)
    t_end = 0.5

    # Reference converged solution
    ref_engine = TimeDependentSchrodingerEngine()
    ref_psi = ref_engine.propagate(
        grid, ho, init_wf, t_end=t_end, dt=1e-4, method=IntegratorType.SPLIT_OPERATOR, store_intermediate_states=False
    ).evolution.final_state.psi

    # Crank-Nicolson temporal order (~2)
    def _run_cn(dt: float) -> np.ndarray:
        return ref_engine.propagate(
            grid, ho, init_wf, t_end=t_end, dt=dt, method=IntegratorType.CRANK_NICOLSON, store_intermediate_states=False
        ).evolution.final_state.psi

    cn_conv = EvolutionValidator.analyze_timestep_convergence(
        _run_cn, dt_values=[0.05, 0.025, 0.0125], reference_psi=ref_psi, x=grid.grid, min_expected_order=1.8
    )
    assert cn_conv.converged is True
    assert cn_conv.estimated_temporal_order is not None and 1.85 <= cn_conv.estimated_temporal_order <= 2.15

    # RK4 temporal order (~4)
    def _run_rk4(dt: float) -> np.ndarray:
        return ref_engine.propagate(
            grid, ho, init_wf, t_end=t_end, dt=dt, method=IntegratorType.RK4, store_intermediate_states=False
        ).evolution.final_state.psi

    rk4_conv = EvolutionValidator.analyze_timestep_convergence(
        _run_rk4, dt_values=[0.01, 0.005, 0.0025], reference_psi=ref_psi, x=grid.grid, min_expected_order=3.5
    )
    assert rk4_conv.converged is True
    assert rk4_conv.estimated_temporal_order is not None and 3.6 <= rk4_conv.estimated_temporal_order <= 4.3


def test_stability_diagnostics_and_json_report():
    """Verify RK4 CFL instability detection and EvolutionValidationReport JSON serialization."""
    grid = SpatialGrid(x_min=-5.0, x_max=5.0, points=200, boundary_condition=BoundaryCondition.DIRICHLET)
    ho = HarmonicOscillator(mass=1.0, omega=1.0)
    init_wf = InitialStateBuilder.gaussian_wavepacket(grid, x0=0.0, sigma=0.5)

    # Excessively large dt for explicit RK4 on fine grid must trigger instability abort
    engine = TimeDependentSchrodingerEngine()
    with pytest.raises(RuntimeError, match="Aborted unstable integration"):
        engine.propagate(grid, ho, init_wf, t_end=1.0, dt=0.1, method=IntegratorType.RK4)

    # InfiniteSquareWell rejected by SplitOperatorIntegrator
    isw = InfiniteSquareWell(width=2.0, center=0.0)
    with pytest.raises(ValueError, match="finite potential values"):
        SplitOperatorIntegrator(grid, isw)

    # JSON serialization of validation report
    res = engine.propagate(grid, ho, init_wf, t_end=0.2, dt=1e-3, method=IntegratorType.CRANK_NICOLSON, save_every=20)
    parsed = json.loads(res.validation.to_json())
    assert parsed["is_valid"] is True
    assert parsed["integrator_name"] == "Crank-Nicolson"
    assert parsed["norm_report"]["is_conserved"] is True
