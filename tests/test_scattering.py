import math
import pytest
import numpy as np

from schrodinger.potentials.models import FiniteSquareWell, FreeParticle
from schrodinger.potentials.barrier import BarrierPotential
from schrodinger.potentials.step import StepPotential
from schrodinger.potentials.poschl_teller import PoschlTellerPotential
from schrodinger.potentials.potential_composition import PiecewisePotential

from schrodinger.scattering import (
    TransmissionCalculator,
    ReflectionCalculator,
    PotentialSlab,
    TransferMatrixEngine,
    ScatteringConfig,
    ScatteringEngine,
    TunnelingEngine,
    ResonanceAnalyzer
)


def test_rectangular_barrier_analytical_and_numerical_agreement():
    """Verify T and R across sub-barrier (E < V0), threshold (E = V0), and above-barrier (E > V0)."""
    v0 = 5.0
    width = 1.5
    barrier = BarrierPotential(height=v0, width=width, center=0.0, baseline=0.0)
    engine = ScatteringEngine(ScatteringConfig(mass=1.0, hbar=1.0, method="transfer_matrix"))

    for energy in [1.5, 3.5, 5.0, 6.5, 10.0]:
        t_ana = TransmissionCalculator.analytical_rectangular_barrier(energy, v0, width)
        r_ana = ReflectionCalculator.analytical_rectangular_barrier(energy, v0, width)
        assert TransmissionCalculator.verify_probability_conservation(r_ana, t_ana, tolerance=1e-12)

        state = engine.solve_scattering(barrier, energy=energy)
        assert np.isclose(state.transmission, t_ana, rtol=1e-10, atol=1e-12)
        assert np.isclose(state.reflection, r_ana, rtol=1e-10, atol=1e-12)
        assert TransmissionCalculator.verify_probability_conservation(
            state.reflection, state.transmission, tolerance=1e-12
        )
        assert state.current_conservation_error < 1e-10


def test_potential_step_sub_and_above_threshold():
    """Verify total reflection for E <= V_R and flux-conserving transmission for E > V_R."""
    step = StepPotential(step_position=0.0, left_height=0.0, right_height=4.0)
    engine = ScatteringEngine(ScatteringConfig(mass=1.0, hbar=1.0))

    # Sub-step (evanescent right channel): R = 1, T = 0
    sub_state = engine.solve_scattering(step, energy=2.5)
    assert np.isclose(sub_state.reflection, 1.0, atol=1e-12)
    assert sub_state.transmission == 0.0
    assert sub_state.current_conservation_error < 1e-10

    # Above-step (propagating right channel): R + T = 1 with k_R / k_L flux factor
    above_state = engine.solve_scattering(step, energy=9.0)
    t_ana = TransmissionCalculator.analytical_potential_step(9.0, 0.0, 4.0)
    r_ana = ReflectionCalculator.analytical_potential_step(9.0, 0.0, 4.0)
    assert np.isclose(above_state.transmission, t_ana, rtol=1e-10)
    assert np.isclose(above_state.reflection, r_ana, rtol=1e-10)
    assert TransmissionCalculator.verify_probability_conservation(
        above_state.reflection, above_state.transmission, tolerance=1e-12
    )


def test_finite_square_well_ramsauer_townsend_resonances():
    """Verify Ramsauer-Townsend T = 1 resonant transmission over a finite square well."""
    depth = 6.0
    width = 2.0
    well = FiniteSquareWell(depth=depth, width=width, center=0.0)
    engine = ScatteringEngine(ScatteringConfig(mass=1.0, hbar=1.0))

    # Resonance condition: q * L = n * pi => E_n = (n^2 * pi^2 * hbar^2)/(2*m*L^2) - V_0 > 0
    # For L=2, m=1, hbar=1, V_0=6: n=3 gives E_3 = 9*pi^2 / 8 - 6 ≈ 5.1033 > 0
    e_res = (9.0 * (math.pi ** 2)) / (2.0 * (width ** 2)) - depth
    assert e_res > 0.0

    t_ana = TransmissionCalculator.analytical_finite_square_well(e_res, depth, width)
    assert np.isclose(t_ana, 1.0, atol=1e-12)

    state = engine.solve_scattering(well, energy=e_res)
    assert np.isclose(state.transmission, 1.0, atol=1e-10)
    assert np.isclose(state.reflection, 0.0, atol=1e-10)


def test_transfer_matrix_determinant_and_boundary_matching():
    """Verify TMM composition, determinant invariance det(M) = k_R / k_L, and interface continuity."""
    tmm = TransferMatrixEngine(mass=1.0, hbar=1.0)
    slabs = [
        PotentialSlab(-2.0, -1.0, 4.0),
        PotentialSlab(-1.0, 1.0, 0.0),
        PotentialSlab(1.0, 2.0, 4.0)
    ]

    res = tmm.solve(energy=2.2, slabs=slabs, v_left=0.0, v_right=1.0)
    assert res.stability.is_stable is True
    assert res.stability.determinant_error < 1e-12
    assert res.stability.boundary_matching_error < 1e-12
    assert res.probability_conservation_error < 1e-12

    # Non-contiguous slabs must raise ValueError
    bad_slabs = [PotentialSlab(0.0, 1.0, 2.0), PotentialSlab(1.5, 2.5, 2.0)]
    with pytest.raises(ValueError, match="contiguous"):
        tmm.solve(energy=3.0, slabs=bad_slabs)


def test_ode_vs_transfer_matrix_and_smooth_poschl_teller():
    """Verify backward ODE agrees with TMM and reproduces reflectionless Poschl-Teller well."""
    barrier = BarrierPotential(height=4.0, width=1.2, center=0.0)
    tmm_engine = ScatteringEngine(ScatteringConfig(method="transfer_matrix"))
    ode_engine = ScatteringEngine(ScatteringConfig(method="ode", ode_rtol=1e-10, ode_atol=1e-12))

    s_tmm = tmm_engine.solve_scattering(barrier, energy=2.5)
    s_ode = ode_engine.solve_scattering(barrier, energy=2.5)

    assert np.isclose(s_ode.transmission, s_tmm.transmission, rtol=1e-7)
    assert np.isclose(s_ode.reflection, s_tmm.reflection, rtol=1e-7)
    assert s_ode.current_conservation_error < 1e-6

    # Reflectionless Pöschl-Teller well (integer lambda = 3 -> V_0 = 3 -> T = 1 for all E > 0)
    pt = PoschlTellerPotential.from_lambda(lam=3.0, alpha=1.0)
    pt_state = ode_engine.solve_scattering(pt, energy=1.5)
    assert np.isclose(pt_state.transmission, 1.0, atol=1e-6)
    assert np.isclose(pt_state.reflection, 0.0, atol=1e-6)


def test_tunneling_engine_parameter_sweeps():
    """Verify energy, barrier-width, and barrier-height tunneling dependence."""
    tun = TunnelingEngine(ScatteringConfig(mass=1.0, hbar=1.0))
    barrier = BarrierPotential(height=5.0, width=1.0, center=0.0)

    # Energy sweep: sub-barrier transmission strictly increases with energy
    e_sweep = tun.energy_dependence(barrier, energies=[0.5, 1.5, 2.5, 3.5, 4.5])
    assert e_sweep.all_conserved is True
    assert np.all(np.diff(e_sweep.transmissions) > 0.0)

    # Width sweep: sub-barrier transmission strictly decreases with width
    w_sweep = tun.width_dependence(widths=[0.5, 1.0, 1.5, 2.0], energy=2.0, barrier_height=5.0)
    assert w_sweep.all_conserved is True
    assert np.all(np.diff(w_sweep.transmissions) < 0.0)

    # Height sweep: sub-barrier transmission strictly decreases with barrier height
    h_sweep = tun.height_dependence(heights=[3.0, 4.0, 5.0, 6.0], energy=2.0, barrier_width=1.0)
    assert h_sweep.all_conserved is True
    assert np.all(np.diff(h_sweep.transmissions) < 0.0)


def test_double_barrier_resonant_tunneling_detection():
    """Verify ResonanceAnalyzer detects sub-barrier resonant tunneling peaks and reflection bands."""
    double_barrier = PiecewisePotential(
        pieces=[
            (-1.5, -1.0, BarrierPotential(height=10.0, width=0.5, center=-1.25)),
            (-1.0, 1.0, FreeParticle()),
            (1.0, 1.5, BarrierPotential(height=10.0, width=0.5, center=1.25))
        ]
    )
    analyzer = ResonanceAnalyzer(ScatteringConfig(mass=1.0, hbar=1.0, method="transfer_matrix"))
    energies = np.linspace(0.2, 6.0, 600)

    report = analyzer.analyze_spectrum(
        double_barrier,
        energies=energies,
        x_left=-1.5,
        x_right=1.5,
        v_left=0.0,
        v_right=0.0,
        resonance_min_transmission=0.95,
        reflection_max_transmission=0.05
    )

    assert report.max_conservation_error < 1e-11
    # Bound-like quasi-stationary states in well of width 2: E_1 ≈ pi^2 / 8 ≈ 1.23, E_2 ≈ 4*pi^2 / 8 ≈ 4.93
    assert len(report.resonances) >= 2
    assert np.isclose(report.resonances[0].energy, (math.pi ** 2) / 8.0, atol=0.15)
    assert report.resonances[0].fwhm is not None and report.resonances[0].fwhm > 0.0
    assert len(report.reflection_regions) >= 1


def test_scattering_grid_convergence():
    """Verify smooth potential slab discretization converges monotonically with num_slabs."""
    pt = PoschlTellerPotential(depth=2.5, alpha=1.0)
    energy = 1.2
    ref_state = ScatteringEngine(ScatteringConfig(method="ode", ode_rtol=1e-11, ode_atol=1e-13)).solve_scattering(
        pt, energy=energy
    )

    errors = []
    for n_slabs in [25, 50, 100, 200]:
        tmm_state = ScatteringEngine(
            ScatteringConfig(method="transfer_matrix", num_slabs=n_slabs)
        ).solve_scattering(pt, energy=energy)
        errors.append(abs(tmm_state.transmission - ref_state.transmission))

    assert errors[-1] < errors[0]
    assert errors[-1] < 1e-4


def test_invalid_scattering_inputs():
    barrier = BarrierPotential(height=5.0, width=1.0, center=0.0)
    engine = ScatteringEngine()

    # Incident energy <= v_left must fail
    with pytest.raises(ValueError, match="must exceed left asymptotic potential"):
        engine.solve_scattering(barrier, energy=0.0)
    with pytest.raises(ValueError, match="must exceed left asymptotic potential"):
        engine.solve_scattering(barrier, energy=-1.0)
