from dataclasses import dataclass
from typing import List, Literal, Optional
import numpy as np
from scipy import integrate
from schrodinger.wavefunctions.wavefunction import Wavefunction
from schrodinger.wavefunctions.quantum_observables import QuantumObservables


SymmetryClassification = Literal["even", "odd", "hermitian_symmetric", "asymmetric"]


@dataclass
class WavefunctionCharacterizationReport:
    """Morphological and statistical characterization of a quantum wavefunction."""
    expectation_x: float
    spatial_width: float
    fwhm: Optional[float]
    peak_position: float
    peak_density: float
    node_count: int
    node_positions: List[float]
    symmetry: SymmetryClassification
    symmetry_deviation: float
    boundary_amplitude: float
    probability_inside_core: float


class WavefunctionCharacterizer:
    """
    Analyzes spatial distribution, width, center, peak position, zero-crossing nodes,
    parity symmetry, and boundary confinement of quantum wavefunctions.
    """

    @staticmethod
    def estimate_fwhm(x: np.ndarray, density: np.ndarray) -> Optional[float]:
        """Estimates Full Width at Half Maximum (FWHM) of |psi(x)|^2."""
        if len(x) < 3:
            return None
        peak_idx = int(np.argmax(density))
        peak_val = float(density[peak_idx])
        if peak_val <= 0.0:
            return None
        half_max = 0.5 * peak_val

        left_x: Optional[float] = None
        for i in range(peak_idx, 0, -1):
            if density[i - 1] <= half_max <= density[i]:
                denom = density[i] - density[i - 1]
                frac = (half_max - density[i - 1]) / denom if denom > 0 else 0.0
                left_x = float(x[i - 1] + frac * (x[i] - x[i - 1]))
                break

        right_x: Optional[float] = None
        for i in range(peak_idx, len(x) - 1):
            if density[i] >= half_max >= density[i + 1]:
                denom = density[i] - density[i + 1]
                frac = (density[i] - half_max) / denom if denom > 0 else 0.0
                right_x = float(x[i] + frac * (x[i + 1] - x[i]))
                break

        if left_x is None or right_x is None or right_x <= left_x:
            return None
        return float(right_x - left_x)

    @staticmethod
    def detect_nodes(x: np.ndarray, psi: np.ndarray) -> List[float]:
        """
        Detects interior zero-crossing node positions via linear interpolation
        on the phase-dominant component of psi(x).
        """
        max_idx = int(np.argmax(np.abs(psi)))
        phase = np.angle(psi[max_idx]) if abs(psi[max_idx]) > 0 else 0.0
        aligned_real = np.real(psi * np.exp(-1j * phase))

        peak = float(np.max(np.abs(aligned_real)))
        if peak <= 0.0:
            return []
        threshold = 1e-7 * peak

        node_positions: List[float] = []
        for i in range(1, len(x) - 2):
            y0, y1 = aligned_real[i], aligned_real[i + 1]
            if (y0 > threshold and y1 < -threshold) or (y0 < -threshold and y1 > threshold):
                frac = abs(y0) / (abs(y0) + abs(y1))
                node_positions.append(float(x[i] + frac * (x[i + 1] - x[i])))
            elif abs(y0) <= threshold and (aligned_real[i - 1] * aligned_real[i + 1] < -(threshold ** 2)):
                node_positions.append(float(x[i]))

        return node_positions

    @staticmethod
    def classify_symmetry(
        x: np.ndarray,
        psi: np.ndarray,
        center: Optional[float] = None,
        tolerance: float = 1e-4
    ) -> tuple:
        """
        Classifies spatial symmetry around `center` (defaults to grid midpoint):
          - 'even': psi(center + dx) == +psi(center - dx)
          - 'odd':  psi(center + dx) == -psi(center - dx)
          - 'hermitian_symmetric': psi(center + dx) == conj(psi(center - dx))
          - 'asymmetric': none of the above
        Returns (SymmetryClassification, deviation).
        """
        sym_center = float(center) if center is not None else float(0.5 * (x[0] + x[-1]))
        half_span = min(abs(x[-1] - sym_center), abs(sym_center - x[0]))
        if half_span <= 0.0:
            return "even", 0.0

        offsets = np.linspace(0.0, half_span, max(len(x) // 2, 50))
        psi_pos = (
            np.interp(sym_center + offsets, x, np.real(psi))
            + 1j * np.interp(sym_center + offsets, x, np.imag(psi))
        )
        psi_neg = (
            np.interp(sym_center - offsets, x, np.real(psi))
            + 1j * np.interp(sym_center - offsets, x, np.imag(psi))
        )

        norm_scale = max(float(np.max(np.abs(psi))), 1e-15)
        even_dev = float(np.max(np.abs(psi_pos - psi_neg)) / norm_scale)
        if even_dev <= tolerance:
            return "even", even_dev

        odd_dev = float(np.max(np.abs(psi_pos + psi_neg)) / norm_scale)
        if odd_dev <= tolerance:
            return "odd", odd_dev

        herm_dev = float(np.max(np.abs(psi_pos - np.conj(psi_neg))) / norm_scale)
        if herm_dev <= tolerance:
            return "hermitian_symmetric", herm_dev

        return "asymmetric", min(even_dev, odd_dev, herm_dev)

    @classmethod
    def characterize(
        cls,
        wf: Wavefunction,
        symmetry_center: Optional[float] = None
    ) -> WavefunctionCharacterizationReport:
        """Produces a complete morphological and statistical characterization report."""
        x = wf.x
        psi = wf.psi
        density = wf.probability_density

        exp_x = QuantumObservables.expectation_x(psi, x)
        var_x = QuantumObservables.variance_x(psi, x)
        width = float(np.sqrt(var_x))

        peak_idx = int(np.argmax(density))
        peak_pos = float(x[peak_idx])
        peak_dens = float(density[peak_idx])
        fwhm = cls.estimate_fwhm(x, density)

        nodes = cls.detect_nodes(x, psi)
        sym, sym_dev = cls.classify_symmetry(x, psi, center=symmetry_center)
        edge_amp = float(max(abs(psi[0]), abs(psi[-1])))

        # Probability contained within [exp_x - width, exp_x + width]
        mask = (x >= exp_x - width) & (x <= exp_x + width)
        core_prob = float(integrate.simpson(y=density[mask], x=x[mask])) if np.sum(mask) >= 2 else 0.0

        return WavefunctionCharacterizationReport(
            expectation_x=exp_x,
            spatial_width=width,
            fwhm=fwhm,
            peak_position=peak_pos,
            peak_density=peak_dens,
            node_count=len(nodes),
            node_positions=nodes,
            symmetry=sym,
            symmetry_deviation=sym_dev,
            boundary_amplitude=edge_amp,
            probability_inside_core=core_prob
        )
