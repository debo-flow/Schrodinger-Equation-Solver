import math
import pytest
import numpy as np
from scipy import integrate

from schrodinger.numerical.grid import SpatialGrid, BoundaryCondition
from schrodinger.numerical.finite_difference import FiniteDifferenceEngine
from schrodinger.operators import KineticEnergyOperator, PotentialEnergyOperator, Hamiltonian
from schrodinger.solvers import EigenvalueSolver, SolverConfig
from schrodinger.wavefunctions import WavefunctionReconstructor
from schrodinger.optimization import OptimizationEngine, OptimizationConfig, ComputationCache

from schrodinger.potentials import (
    HarmonicOscillator,
    InfiniteSquareWell,
    MorsePotential,
    PoschlTellerPotential,
    QuarticAnharmonicOscillator,
    PeriodicPotential,
    StepPotential,
    BarrierPotential,
    CoulombLikePotential,
    PotentialSum,
    ScaledPotential,
    PiecewisePotential,
    ParameterizedPotentialCombination,
    PotentialAnalyzer
)
from schrodinger.verification import (
    AnalyticalSolver,
    MorseAnalytical,
    PoschlTellerAnalytical,
    NumericalReferenceSolution,
    NumericalComparisonEngine
)


def _solve_with_engine(potential, x_min, x_max, points, num_states=3, mass=1.0, hbar=1.0):
    grid = SpatialGrid(x_min=x_min, x_max=x_max, points=points, boundary_condition=BoundaryCondition.DIRICHLET)
    engine = OptimizationEngine(OptimizationConfig(enable_cache=False, monitor_performance=False))
    return engine.solve_system(grid, potential, num_eigenvalues=num_states, mass=mass, hbar=hbar).wavefunctions


def test_morse_potential_evaluation_and_validation():
    with pytest.raises(ValueError):
        MorsePotential(dissociation_energy=-5.0)
    with pytest.raises(ValueError):
        MorsePotential(width_parameter=0.0)

    morse = MorsePotential(dissociation_energy=10.0, width_parameter=0.5, equilibrium_position=1.5)
    assert isinstance(morse(1.5), float)
    assert morse(1.5) == 0.0

    # Far to the right -> approaches D_e = 10.0
    assert np.isclose(morse(100.0), 10.0, rtol=1e-10)

    # Far to the left -> large positive value without overflow/NaN
    repulsive = morse(-2000.0)
    assert np.isfinite(repulsive) and repulsive > 1e100


def test_poschl_teller_potential_and_factory():
    with pytest.raises(ValueError):
        PoschlTellerPotential(depth=-1.0)
    with pytest.raises(ValueError):
        PoschlTellerPotential.from_lambda(lam=0.8)

    pt = PoschlTellerPotential.from_lambda(lam=3.0, alpha=1.0, mass=1.0, hbar=1.0)
    # V_0 = (1/2) * 3 * (3 - 1) = 3.0
    assert np.isclose(pt(0.0), -3.0)
    assert np.isclose(pt(50.0), 0.0, atol=1e-12)


def test_anharmonic_oscillator():
    with pytest.raises(ValueError):
        QuarticAnharmonicOscillator(quadratic_coeff=0.0, quartic_coeff=0.0)
    with pytest.raises(ValueError):
        QuarticAnharmonicOscillator(quadratic_coeff=-1.0, quartic_coeff=1.0)

    qao = QuarticAnharmonicOscillator(quadratic_coeff=0.5, quartic_coeff=0.25, center=1.0)
    assert qao(1.0) == 0.0
    # At x = 3.0 (dx = 2): 0.5*4 + 0.25*16 = 2 + 4 = 6.0
    assert np.isclose(qao(3.0), 6.0)


def test_periodic_step_barrier_and_coulomb_potentials():
    # Periodic
    with pytest.raises(ValueError):
        PeriodicPotential(period=-2.0)
    per = PeriodicPotential(amplitude=2.0, period=4.0, offset=1.0)
    assert np.isclose(per(0.0), 3.0)
    assert np.isclose(per(2.0), -1.0)
    assert np.isclose(per(4.0), 3.0)

    # Step
    step = StepPotential(step_position=0.5, left_height=-1.0, right_height=3.0)
    vals = step(np.array([0.0, 0.5, 1.0]))
    np.testing.assert_array_equal(vals, [-1.0, 3.0, 3.0])

    # Barrier
    with pytest.raises(ValueError):
        BarrierPotential(height=0.0, width=1.0, baseline=0.0)
    bar = BarrierPotential(height=5.0, width=2.0, center=0.0, baseline=-1.0)
    np.testing.assert_array_equal(bar(np.array([-2.0, -1.0, 0.0, 1.0, 2.0])), [-1.0, 5.0, 5.0, 5.0, -1.0])

    # Coulomb-like
    with pytest.raises(ValueError):
        CoulombLikePotential(strength=1.0, softening=0.0)
    with pytest.raises(ValueError):
        CoulombLikePotential(strength=0.0, softening=0.1)
    coul = CoulombLikePotential(strength=2.0, softening=0.5, center=1.0)
    assert np.isclose(coul(1.0), -4.0)  # -2.0 / 0.5
    assert np.isclose(coul(1.0 + math.sqrt(3.75)), -1.0)  # -2.0 / sqrt(3.75 + 0.25)


def test_potential_composition_and_unit_enforcement():
    ho = HarmonicOscillator(mass=1.0, omega=1.0)
    bar = BarrierPotential(height=2.0, width=0.5, center=0.0)

    # Operator overloading (+, -, *, unary -)
    combined = 2.0 * ho + bar - 0.5 * (-bar)
    x = np.array([-2.0, 0.0, 2.0])
    expected = 2.0 * ho(x) + 1.5 * bar(x)
    np.testing.assert_allclose(combined(x), expected)

    # Unit mismatch rejection
    ho_si = HarmonicOscillator(unit_system="si")
    with pytest.raises(ValueError, match="Unit mismatch"):
        _ = ho + ho_si

    # Piecewise composition
    pw = PiecewisePotential(
        pieces=[
            (-5.0, -0.001, StepPotential(left_height=2.0, right_height=2.0)),
            (0.0, 5.0, HarmonicOscillator(mass=1.0, omega=1.0))
        ]
    )
    np.testing.assert_allclose(pw(np.array([-2.0, 0.0, 2.0])), [2.0, 0.0, 2.0])

    # Parameterized linear combination with dynamic weight update
    combo = ParameterizedPotentialCombination([ho, bar], weights=[1.0, 0.0], offset=0.5)
    assert np.isclose(combo(0.0), 0.5)
    combo.update_weights([1.0, 2.0])
    assert np.isclose(combo(0.0), 4.5)


def test_potential_analysis_suite():
    grid = SpatialGrid(x_min=-4.0, x_max=4.0, points=401)

    # Symmetric double-well-like quartic potential: V(x) = -2x^2 + 0.5x^4 -> minima at +/- sqrt(2)
    left_well = HarmonicOscillator(mass=1.0, omega=2.0, center=-1.5)
    right_well = HarmonicOscillator(mass=1.0, omega=2.0, center=1.5)
    double_well = PiecewisePotential([(-4.0, 0.0, left_well), (0.0, 4.0, right_well)])

    report = PotentialAnalyzer.analyze(grid, double_well, symmetry_center=0.0)
    assert report.symmetry == "even"
    assert len(report.local_minima) == 2
    assert np.isclose(report.local_minima[0].position, -1.5, atol=0.05)
    assert np.isclose(report.local_minima[1].position, 1.5, atol=0.05)
    assert report.is_continuous is True
    assert report.has_infinite_walls is False

    # Step potential discontinuity & asymmetry detection
    step = StepPotential(step_position=0.0, left_height=0.0, right_height=5.0)
    step_report = PotentialAnalyzer.analyze(grid, step, symmetry_center=0.0)
    assert step_report.symmetry == "asymmetric"
    assert step_report.is_continuous is False
    assert np.isclose(step_report.discontinuity_positions[0], 0.0, atol=0.05)

    # Infinite square well symmetry and infinite wall detection
    isw = InfiniteSquareWell(width=4.0, center=0.0)
    isw_report = PotentialAnalyzer.analyze(grid, isw, symmetry_center=0.0)
    assert isw_report.symmetry == "even"
    assert isw_report.has_infinite_walls is True


def test_exact_morse_and_poschl_teller_analytical_verification():
    assert "morse" in AnalyticalSolver.available_models()
    assert "poschl_teller" in AnalyticalSolver.available_models()

    # 1. Exact Morse Benchmark
    morse_pot = MorsePotential(dissociation_energy=12.0, width_parameter=0.5, equilibrium_position=0.0)
    morse_ana = MorseAnalytical(dissociation_energy=12.0, width_parameter=0.5, equilibrium_position=0.0)
    assert morse_ana.is_exact is True

    morse_wfs = _solve_with_engine(morse_pot, x_min=-5.0, x_max=22.0, points=500, num_states=3)
    morse_comp = NumericalComparisonEngine.compare_spectrum(morse_wfs, morse_ana)
    assert morse_comp.max_energy_relative_error < 1e-3
    assert morse_comp.max_wavefunction_l2_error < 1e-2

    # 2. Exact Pöschl–Teller Benchmark (lambda = 4 -> 3 bound states: n = 0, 1, 2)
    pt_pot = PoschlTellerPotential.from_lambda(lam=4.0, alpha=1.0)
    pt_ana = PoschlTellerAnalytical(depth=pt_pot.depth, alpha=1.0)
    assert pt_ana.is_exact is True
    assert pt_ana.num_bound_states == 3

    pt_wfs = _solve_with_engine(pt_pot, x_min=-10.0, x_max=10.0, points=500, num_states=3)
    pt_comp = NumericalComparisonEngine.compare_spectrum(pt_wfs, pt_ana)
    assert pt_comp.max_energy_relative_error < 1e-3
    assert pt_comp.max_wavefunction_l2_error < 1e-2


def test_numerical_reference_comparison_for_non_exact_potentials():
    # Quartic anharmonic oscillator & regularized Coulomb-like potential
    qao = QuarticAnharmonicOscillator(quadratic_coeff=0.5, quartic_coeff=0.25)
    num_ref = NumericalReferenceSolution(qao, x_min=-6.0, x_max=6.0, num_states=3, reference_points=801)
    assert num_ref.is_exact is False

    coarse_wfs = _solve_with_engine(qao, x_min=-6.0, x_max=6.0, points=250, num_states=3)
    comp = NumericalComparisonEngine.compare_spectrum(coarse_wfs, num_ref)
    assert comp.max_energy_relative_error < 2e-3
    assert comp.max_wavefunction_l2_error < 2e-2

    # Regularized Coulomb-like ground state integration
    coul = CoulombLikePotential(strength=1.0, softening=0.2, center=0.0)
    coul_wfs = _solve_with_engine(coul, x_min=-15.0, x_max=15.0, points=400, num_states=2)
    assert coul_wfs[0].energy < 0.0
    assert np.isclose(integrate.simpson(y=coul_wfs[0].probability_density, x=coul_wfs[0].x), 1.0, atol=1e-4)


def test_hamiltonian_and_cache_compatibility_with_composed_potentials():
    grid = SpatialGrid(x_min=-6.0, x_max=6.0, points=150)
    combo = ParameterizedPotentialCombination(
        [HarmonicOscillator(omega=1.0), PeriodicPotential(amplitude=0.2, period=2.0)],
        weights=[1.0, 1.0]
    )

    # Full Milestone 5 pipeline compatibility
    fd = FiniteDifferenceEngine(grid)
    ham = Hamiltonian(KineticEnergyOperator(fd), PotentialEnergyOperator(grid, combo))
    h_mat = ham.build_sparse(accuracy_order=2, interior_only=True)
    res = EigenvalueSolver(SolverConfig(num_eigenvalues=2)).solve(h_mat)
    wfs = WavefunctionReconstructor(grid).reconstruct(res)
    assert len(wfs) == 2

    # Cache invalidation when ParameterizedPotentialCombination weights change
    cache = ComputationCache()
    opt = OptimizationEngine(OptimizationConfig(enable_cache=True), cache=cache)
    r1 = opt.solve_system(grid, combo, num_eigenvalues=2)
    assert r1.cache_hit is False
    r2 = opt.solve_system(grid, combo, num_eigenvalues=2)
    assert r2.cache_hit is True

    combo.update_weights([1.0, 3.0])
    r3 = opt.solve_system(grid, combo, num_eigenvalues=2)
    assert r3.cache_hit is False
    assert not np.allclose(r1.energies, r3.energies)
