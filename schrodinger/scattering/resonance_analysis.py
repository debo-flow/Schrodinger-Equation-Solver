from dataclasses import dataclass
from typing import List, Optional, Sequence
import numpy as np
from schrodinger.potentials.base import Potential
from .tunneling_engine import TunnelingEngine
from .scattering_engine import ScatteringConfig


@dataclass
class ResonancePeak:
    """Represents a resonant transmission peak in T(E)."""
    energy: float
    transmission: float
    reflection: float
    fwhm: Optional[float]
    lifetime_estimate: Optional[float]


@dataclass
class ReflectionRegion:
    """Contiguous energy interval where reflection dominates (T <= threshold)."""
    energy_start: float
    energy_end: float
    mean_reflection: float
    min_transmission: float


@dataclass
class ResonanceAnalysisReport:
    """Comprehensive energy-dependent scattering and resonance report."""
    energies: np.ndarray
    transmissions: np.ndarray
    reflections: np.ndarray
    resonances: List[ResonancePeak]
    reflection_regions: List[ReflectionRegion]
    max_conservation_error: float


class ResonanceAnalyzer:
    """
    Identifies transmission resonances, resonance widths (FWHM), and high-reflection
    regimes across configurable incident energy scans.
    """

    def __init__(self, config: ScatteringConfig = ScatteringConfig()):
        self.config = config
        self.tunneling_engine = TunnelingEngine(config)

    @staticmethod
    def _estimate_fwhm(energies: np.ndarray, transmissions: np.ndarray, peak_idx: int) -> Optional[float]:
        """Estimates Full Width at Half Maximum (FWHM) around a transmission peak."""
        peak_t = transmissions[peak_idx]
        left_min = float(np.min(transmissions[: peak_idx + 1]))
        right_min = float(np.min(transmissions[peak_idx:]))
        baseline = max(left_min, right_min)
        if peak_t - baseline < 0.1:
            return None
        half_max = 0.5 * (peak_t + baseline)

        # Search left crossing
        e_left: Optional[float] = None
        for i in range(peak_idx, 0, -1):
            if transmissions[i - 1] <= half_max <= transmissions[i]:
                dt = transmissions[i] - transmissions[i - 1]
                frac = (half_max - transmissions[i - 1]) / dt if dt > 0 else 0.0
                e_left = float(energies[i - 1] + frac * (energies[i] - energies[i - 1]))
                break

        # Search right crossing
        e_right: Optional[float] = None
        for i in range(peak_idx, len(energies) - 1):
            if transmissions[i] >= half_max >= transmissions[i + 1]:
                dt = transmissions[i] - transmissions[i + 1]
                frac = (transmissions[i] - half_max) / dt if dt > 0 else 0.0
                e_right = float(energies[i] + frac * (energies[i + 1] - energies[i]))
                break

        if e_left is None or e_right is None or e_right <= e_left:
            return None
        return float(e_right - e_left)

    @classmethod
    def find_resonances(
        cls,
        energies: np.ndarray,
        transmissions: np.ndarray,
        min_transmission: float = 0.9,
        prominence: float = 0.05,
        hbar: float = 1.0
    ) -> List[ResonancePeak]:
        """Detects local maxima in T(E) exceeding `min_transmission` with sufficient prominence."""
        peaks: List[ResonancePeak] = []
        n = len(energies)
        if n < 3:
            return peaks

        for i in range(1, n - 1):
            t_curr = transmissions[i]
            if t_curr < min_transmission:
                continue
            if t_curr >= transmissions[i - 1] and t_curr > transmissions[i + 1]:
                local_floor = min(float(np.min(transmissions[:i])), float(np.min(transmissions[i + 1 :])))
                if (t_curr - local_floor) >= prominence:
                    fwhm = cls._estimate_fwhm(energies, transmissions, i)
                    lifetime = float(hbar / fwhm) if (fwhm is not None and fwhm > 0.0) else None
                    peaks.append(
                        ResonancePeak(
                            energy=float(energies[i]),
                            transmission=float(t_curr),
                            reflection=float(max(0.0, 1.0 - t_curr)),
                            fwhm=fwhm,
                            lifetime_estimate=lifetime
                        )
                    )
        return peaks

    @staticmethod
    def find_reflection_regions(
        energies: np.ndarray,
        transmissions: np.ndarray,
        reflections: np.ndarray,
        max_transmission_threshold: float = 0.1
    ) -> List[ReflectionRegion]:
        """Groups contiguous energy intervals where T(E) <= max_transmission_threshold."""
        regions: List[ReflectionRegion] = []
        in_region = False
        start_idx = 0

        for i, t_val in enumerate(transmissions):
            if t_val <= max_transmission_threshold and not in_region:
                in_region = True
                start_idx = i
            elif t_val > max_transmission_threshold and in_region:
                in_region = False
                end_idx = i - 1
                regions.append(
                    ReflectionRegion(
                        energy_start=float(energies[start_idx]),
                        energy_end=float(energies[end_idx]),
                        mean_reflection=float(np.mean(reflections[start_idx : end_idx + 1])),
                        min_transmission=float(np.min(transmissions[start_idx : end_idx + 1]))
                    )
                )

        if in_region:
            end_idx = len(energies) - 1
            regions.append(
                ReflectionRegion(
                    energy_start=float(energies[start_idx]),
                    energy_end=float(energies[end_idx]),
                    mean_reflection=float(np.mean(reflections[start_idx : end_idx + 1])),
                    min_transmission=float(np.min(transmissions[start_idx : end_idx + 1]))
                )
            )
        return regions

    def analyze_spectrum(
        self,
        potential: Potential,
        energies: Sequence[float],
        x_left: Optional[float] = None,
        x_right: Optional[float] = None,
        v_left: Optional[float] = None,
        v_right: Optional[float] = None,
        resonance_min_transmission: float = 0.9,
        reflection_max_transmission: float = 0.1
    ) -> ResonanceAnalysisReport:
        """Sweeps incident energies and extracts resonances and reflection bands."""
        sweep = self.tunneling_engine.energy_dependence(
            potential, energies, x_left=x_left, x_right=x_right, v_left=v_left, v_right=v_right
        )
        resonances = self.find_resonances(
            sweep.parameter_values,
            sweep.transmissions,
            min_transmission=resonance_min_transmission,
            hbar=self.config.hbar
        )
        ref_regions = self.find_reflection_regions(
            sweep.parameter_values,
            sweep.transmissions,
            sweep.reflections,
            max_transmission_threshold=reflection_max_transmission
        )
        return ResonanceAnalysisReport(
            energies=sweep.parameter_values,
            transmissions=sweep.transmissions,
            reflections=sweep.reflections,
            resonances=resonances,
            reflection_regions=ref_regions,
            max_conservation_error=float(np.max(sweep.conservation_errors))
        )
