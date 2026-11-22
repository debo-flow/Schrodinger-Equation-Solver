import math
import pytest
import numpy as np

from schrodinger.numerical.grid import SpatialGrid, BoundaryCondition
from schrodinger.potentials.models import FreeParticle, HarmonicOscillator, InfiniteSquareWell
from schrodinger.wavefunctions.normalization import NormalizationEngine
from schrodinger.dynamics.time_integrators import IntegratorType

from schrodinger.wavepackets import (
    GaussianWavepacketSpec,
    GaussianWavepacketGenerator,
    PlaneWaveSpec,
    PlaneWaveGenerator,
    SuperpositionEngine,
    CoherentStateGenerator,
    InitialStateType,
    InitialStateEngine,
    MomentumSpaceEngine,
    WavefunctionCharacterizer,
    WavepacketDynamicsAnalyzer
)


def test_gaussian_wavepacket_normalization_and_moments():
    grid = SpatialGrid(x_min=-10.0, x_max=10.0, points=512)
    x0 = -1.5
    sigma = 0.75
    k0 = 2.5

    wf = GaussianWavepacketGenerator.generate(grid, x0=x0, sigma=sigma, k0=k0, mass=1.0, hbar=1.0)
    assert NormalizationEngine.verify_normalization(wf.psi, grid.grid, tolerance=1e-6)

    char = WavefunctionCharacterizer.characterize(wf)
    assert np.isclose(char.expectation_x, x0, atol=1e-4)
    assert np.isclose(char.spatial_width, sigma, atol=1e-4)
    assert np.isclose(char.peak_position, x0, atol=grid.dx)

    # Analytical vs numerical momentum-space moments
    mom = MomentumSpaceEngine.to_momentum_space(wf, grid, hbar=1.0)
    assert np.isclose(mom.fourier_norm, 1.0, atol=1e-6)
    assert np.isclose(mom.expectation_p, k0, atol=1e-4)
    assert np.isclose(mom.uncertainty_p, 1.0 / (2.0 * sigma), atol=1e-4)
    assert np.isclose(char.spatial_width * mom.uncertainty_p, 0.5, atol=1e-4)

    # Invalid parameters & Nyquist violation
    with pytest.raises(ValueError):
        GaussianWavepacketGenerator.generate(grid, sigma=-0.1)
    with pytest.raises(ValueError, match="Nyquist"):
        GaussianWavepacketGenerator.generate(grid, k0=1000.0)


def test_plane_wave_construction_and_periodicity():
    grid = SpatialGrid(x_min=0.0, x_max=2.0 * math.pi, points=256)

    # Periodic mode index n = 3 -> k = 3.0, E = 4.5
    pw = PlaneWaveGenerator.generate(grid, mode_index=3, mass=1.0, hbar=1.0)
    assert NormalizationEngine.verify_normalization(pw.psi, grid.grid, tolerance=1e-6)
    assert np.isclose(pw.energy, 4.5)
    assert PlaneWaveGenerator.is_periodic_on_grid(pw, tolerance=1e-10)

    # Apodized plane wave for Dirichlet boundaries
    pw_tapered = PlaneWaveGenerator.generate(grid, k=4.0, taper_fraction=0.2)
    assert NormalizationEngine.verify_normalization(pw_tapered.psi, grid.grid, tolerance=1e-5)
    assert abs(pw_tapered.psi[0]) < 1e-12
    assert abs(pw_tapered.psi[-1]) < 1e-12

    with pytest.raises(ValueError):
        PlaneWaveGenerator.generate(grid, k=1.0, mode_index=1)


def test_superposition_engine_normalization_and_orthogonality():
    grid = SpatialGrid(x_min=0.0, x_max=math.pi, points=300, boundary_condition=BoundaryCondition.DIRICHLET)
    isw = InfiniteSquareWell(width=math.pi, center=0.5 * math.pi)
    init_engine = InitialStateEngine(mass=1.0, hbar=1.0)

    psi_0, _ = init_engine.create_stationary_eigenstate(grid, isw, state_index=0)  # E_0 = 0.5
    psi_1, _ = init_engine.create_stationary_eigenstate(grid, isw, state_index=1)  # E_1 = 2.0

    sup_wf, report = SuperpositionEngine.create_with_report(
        [psi_0, psi_1],
        coefficients=[3.0, 4.0j],
        enforce_orthogonality=True
    )

    assert NormalizationEngine.verify_normalization(sup_wf.psi, grid.grid, tolerance=1e-6)
    assert report.basis_is_orthogonal is True
    # Probabilities: 9/25 = 0.36 and 16/25 = 0.64 -> <E> = 0.36*0.5 + 0.64*2.0 = 1.46
    np.testing.assert_allclose(report.occupation_probabilities, [0.36, 0.64], atol=1e-6)
    assert np.isclose(report.expected_energy, 1.46, atol=1e-3)

    # Non-orthogonal basis rejection when enforce_orthogonality=True
    with pytest.raises(ValueError, match="orthogonality"):
        SuperpositionEngine.create([psi_0, psi_0], [1.0, 1.0], enforce_orthogonality=True)


def test_coherent_state_analytical_properties():
    grid = SpatialGrid(x_min=-10.0, x_max=10.0, points=512)
    alpha = 1.2 - 0.8j
    omega = 1.5
    mass = 1.0
    hbar = 1.0

    wf = CoherentStateGenerator.generate(grid, alpha=alpha, mass=mass, omega=omega, hbar=hbar)
    assert NormalizationEngine.verify_normalization(wf.psi, grid.grid, tolerance=1e-6)

    validation = CoherentStateGenerator.validate_coherent_state(
        wf, grid, alpha=alpha, mass=mass, omega=omega, hbar=hbar, tolerance=1e-3
    )
    assert validation.is_valid is True
    assert validation.heisenberg_minimum_error < 1e-4

    # Check Poissonian Fock probabilities sum to 1
    probs = CoherentStateGenerator.fock_occupation_probabilities(alpha, max_n=25)
    assert np.isclose(sum(probs.values()), 1.0, atol=1e-10)


def test_fourier_transform_roundtrip_and_parseval():
    grid = SpatialGrid(x_min=-12.0, x_max=12.0, points=512)
    wf = GaussianWavepacketGenerator.generate(grid, x0=2.3, sigma=0.9, k0=-1.7)

    mom = MomentumSpaceEngine.to_momentum_space(wf, grid, hbar=1.0)
    assert np.isclose(mom.fourier_norm, 1.0, atol=1e-6)

    psi_reconstructed = MomentumSpaceEngine.to_position_space(mom, grid)
    np.testing.assert_allclose(psi_reconstructed, wf.psi, atol=1e-12)


def test_wavefunction_characterization_nodes_and_symmetry():
    grid = SpatialGrid(x_min=-6.0, x_max=6.0, points=401, boundary_condition=BoundaryCondition.DIRICHLET)
    ho = HarmonicOscillator(mass=1.0, omega=1.0, center=0.0)
    init_engine = InitialStateEngine()

    wf_0, _ = init_engine.create_stationary_eigenstate(grid, ho, state_index=0)
    wf_1, _ = init_engine.create_stationary_eigenstate(grid, ho, state_index=1)
    wf_2, _ = init_engine.create_stationary_eigenstate(grid, ho, state_index=2)

    c0 = WavefunctionCharacterizer.characterize(wf_0, symmetry_center=0.0)
    c1 = WavefunctionCharacterizer.characterize(wf_1, symmetry_center=0.0)
    c2 = WavefunctionCharacterizer.characterize(wf_2, symmetry_center=0.0)

    assert c0.node_count == 0 and c0.symmetry == "even"
    assert c1.node_count == 1 and c1.symmetry == "odd"
    assert np.isclose(c1.node_positions[0], 0.0, atol=0.05)
    assert c2.node_count == 2 and c2.symmetry == "even"


def test_initial_state_engine_custom_and_dispatcher():
    grid = SpatialGrid(x_min=-5.0, x_max=5.0, points=201)
    engine = InitialStateEngine()

    wf_custom, report = engine.create_custom(grid, lambda x: np.exp(-x ** 2) * np.cos(2.0 * x), energy=1.2)
    assert NormalizationEngine.verify_normalization(wf_custom.psi, grid.grid)
    assert report.boundary_leakage_ok is True

    wf_disp = engine.generate(InitialStateType.GAUSSIAN, grid=grid, x0=0.0, sigma=0.6, k0=1.0)
    assert NormalizationEngine.verify_normalization(wf_disp.psi, grid.grid)


def test_free_particle_wavepacket_spreading_dynamics():
    grid = SpatialGrid(x_min=-16.0, x_max=16.0, points=512, boundary_condition=BoundaryCondition.DIRICHLET)
    fp = FreeParticle()
    sigma0 = 0.7
    x0 = -2.0
    k0 = 2.0
    t_end = 1.2

    init_wf = GaussianWavepacketGenerator.generate(grid, x0=x0, sigma=sigma0, k0=k0, mass=1.0, hbar=1.0)
    analyzer = WavepacketDynamicsAnalyzer(mass=1.0, hbar=1.0)

    report = analyzer.simulate(
        grid=grid,
        potential=fp,
        initial_state=init_wf,
        t_end=t_end,
        dt=2e-3,
        method=IntegratorType.SPLIT_OPERATOR,
        save_every=30,
        initial_gaussian_sigma=sigma0
    )

    assert report.solver_result.validation.is_valid is True
    assert report.boundary_reflection_detected is False
    assert report.analytical_free_widths is not None

    # Wavepacket must spread monotonically and match analytical sigma(t)
    assert report.spatial_widths[-1] > sigma0
    np.testing.assert_allclose(report.spatial_widths, report.analytical_free_widths, rtol=1e-2)

    # Center moves at group velocity v_g = hbar * k0 / m = 2.0
    expected_x_final = x0 + k0 * t_end
    assert np.isclose(report.expectation_x[-1], expected_x_final, atol=2e-2)

    # Position-space <p(t)> and Fourier-space <p(t)> must agree and stay conserved
    np.testing.assert_allclose(report.fourier_expectation_p, k0, atol=1e-2)
    np.testing.assert_allclose(report.expectation_p, k0, atol=2e-2)
