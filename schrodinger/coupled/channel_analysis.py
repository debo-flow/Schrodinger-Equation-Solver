from dataclasses import dataclass
from typing import List, Sequence
import numpy as np
from .multi_wavefunction import MultiComponentWavefunction


@dataclass
class ChannelPopulationReport:
    """Stationary channel occupation and conservation summary for a coupled state."""
    num_channels: int
    channel_labels: List[str]
    channel_populations: List[float]
    total_population: float
    conservation_error: float
    dominant_channel: int
    bounds_valid: bool


@dataclass
class ChannelTrajectoryReport:
    """Time-dependent channel population and population-transfer diagnostics."""
    times: np.ndarray
    channel_labels: List[str]
    populations_over_time: np.ndarray
    total_population_over_time: np.ndarray
    max_conservation_error: float
    initial_populations: np.ndarray
    final_populations: np.ndarray
    net_population_transfer: np.ndarray
    max_population_transfer: np.ndarray
    bounds_valid: bool


class ChannelPopulationAnalyzer:
    """
    Analyzes channel occupation probabilities, total probability conservation,
    population transfer across channels, state overlaps, and transition probabilities.
    """

    @staticmethod
    def analyze_state(
        state: MultiComponentWavefunction,
        tolerance: float = 1e-4
    ) -> ChannelPopulationReport:
        """Computes per-channel populations and verifies bounds [0, 1] and sum = 1."""
        pops = state.channel_populations
        total_pop = float(np.sum(pops))
        cons_err = abs(total_pop - 1.0)

        within_bounds = bool(
            np.all(pops >= -tolerance)
            and np.all(pops <= 1.0 + tolerance)
            and (cons_err <= tolerance)
        )

        return ChannelPopulationReport(
            num_channels=state.num_channels,
            channel_labels=state.channel_labels,
            channel_populations=[float(p) for p in pops],
            total_population=total_pop,
            conservation_error=float(cons_err),
            dominant_channel=int(np.argmax(pops)),
            bounds_valid=within_bounds
        )

    @staticmethod
    def analyze_trajectory(
        times: Sequence[float],
        states: Sequence[MultiComponentWavefunction],
        tolerance: float = 1e-4
    ) -> ChannelTrajectoryReport:
        """Evaluates time-dependent channel populations P_c(t) and population transfer."""
        if len(times) != len(states):
            raise ValueError("times and states must have identical lengths.")
        if len(times) == 0:
            raise ValueError("Cannot analyze an empty trajectory.")

        t_arr = np.asarray(times, dtype=np.float64)
        n_steps = len(states)
        n_channels = states[0].num_channels

        pops_matrix = np.zeros((n_steps, n_channels), dtype=np.float64)
        for idx, st in enumerate(states):
            if st.num_channels != n_channels:
                raise ValueError("All trajectory states must have identical channel counts.")
            pops_matrix[idx, :] = st.channel_populations

        totals = np.sum(pops_matrix, axis=1)
        cons_errors = np.abs(totals - 1.0)
        max_cons_err = float(np.max(cons_errors))

        init_pops = pops_matrix[0, :].copy()
        final_pops = pops_matrix[-1, :].copy()
        net_transfer = final_pops - init_pops
        max_transfer = np.max(np.abs(pops_matrix - init_pops[np.newaxis, :]), axis=0)

        bounds_ok = bool(
            np.all(pops_matrix >= -tolerance)
            and np.all(pops_matrix <= 1.0 + tolerance)
            and (max_cons_err <= tolerance)
        )

        return ChannelTrajectoryReport(
            times=t_arr,
            channel_labels=states[0].channel_labels,
            populations_over_time=pops_matrix,
            total_population_over_time=totals,
            max_conservation_error=max_cons_err,
            initial_populations=init_pops,
            final_populations=final_pops,
            net_population_transfer=net_transfer,
            max_population_transfer=max_transfer,
            bounds_valid=bounds_ok
        )

    @staticmethod
    def state_overlap(
        state_a: MultiComponentWavefunction,
        state_b: MultiComponentWavefunction
    ) -> complex:
        """Computes complex overlap amplitude <Psi_a | Psi_b>."""
        return state_a.inner_product(state_b)

    @staticmethod
    def transition_probability(
        initial_state: MultiComponentWavefunction,
        target_state: MultiComponentWavefunction,
        tolerance: float = 1e-4
    ) -> float:
        """
        Calculates transition probability P_{i -> f} = |<Psi_target | Psi_initial>|^2
        and verifies it obeys valid probability bounds [0, 1].
        """
        prob = initial_state.overlap_probability(target_state)
        if prob < -tolerance or prob > 1.0 + tolerance:
            raise ValueError(f"Transition probability {prob:.6f} out of physical bounds [0, 1].")
        return float(np.clip(prob, 0.0, 1.0))
