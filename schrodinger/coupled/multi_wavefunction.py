from typing import List, Optional, Sequence
import numpy as np
from scipy import integrate
from schrodinger.wavefunctions.wavefunction import Wavefunction
from schrodinger.wavefunctions.normalization import IntegrationMethod


class MultiComponentWavefunction:
    """
    Represents an N-channel quantum state:
        Psi(x) = [psi_1(x), psi_2(x), ..., psi_N(x)]^T

    Normalization is strictly enforced across the COMPLETE multi-component state:
        integral Psi^dagger(x) Psi(x) dx = sum_c integral |psi_c(x)|^2 dx = 1
    """

    def __init__(
        self,
        x: np.ndarray,
        components: np.ndarray,
        energy: float = 0.0,
        channel_labels: Optional[Sequence[str]] = None,
        normalize: bool = True,
        integration_method: IntegrationMethod = IntegrationMethod.SIMPSON,
        norm_tolerance: float = 1e-4
    ):
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim != 1 or len(x_arr) < 3:
            raise ValueError("Spatial grid x must be a 1D array with at least 3 points.")

        comp_arr = np.asarray(components, dtype=np.complex128)
        if comp_arr.ndim != 2:
            raise ValueError(
                f"components must be a 2D array of shape (num_channels, num_points). Received shape: {comp_arr.shape}"
            )
        if comp_arr.shape[0] < 1:
            raise ValueError("MultiComponentWavefunction requires at least 1 channel.")
        if comp_arr.shape[1] != len(x_arr):
            raise ValueError(
                f"Component spatial length ({comp_arr.shape[1]}) must match grid length ({len(x_arr)})."
            )
        if not np.all(np.isfinite(comp_arr)):
            raise ValueError("MultiComponentWavefunction components contain NaN or Inf values.")

        num_channels = comp_arr.shape[0]
        if channel_labels is not None:
            if len(channel_labels) != num_channels:
                raise ValueError(
                    f"channel_labels length ({len(channel_labels)}) must match num_channels ({num_channels})."
                )
            self._channel_labels = [str(lbl) for lbl in channel_labels]
        else:
            self._channel_labels = [f"channel_{i}" for i in range(num_channels)]

        self._x = x_arr.copy()
        self._energy = float(energy)
        self._integration_method = integration_method

        total_integral = self._integrate_total_density(comp_arr, self._x, integration_method)
        if total_integral <= 0.0 or not np.isfinite(total_integral):
            raise ValueError(f"Complete-state probability integral is non-positive ({total_integral}).")

        if normalize:
            self._components = comp_arr / np.sqrt(total_integral)
        else:
            if abs(total_integral - 1.0) > norm_tolerance:
                raise ValueError(
                    f"Multi-component state is not normalized: integral = {total_integral:.6f} != 1.0."
                )
            self._components = comp_arr.copy()

    @staticmethod
    def _integrate_total_density(
        components: np.ndarray,
        x: np.ndarray,
        method: IntegrationMethod
    ) -> float:
        total_density = np.sum(np.abs(components) ** 2, axis=0)
        if method == IntegrationMethod.SIMPSON:
            return float(integrate.simpson(y=total_density, x=x))
        return float(integrate.trapezoid(y=total_density, x=x))

    @property
    def num_channels(self) -> int:
        return int(self._components.shape[0])

    @property
    def num_points(self) -> int:
        return int(self._components.shape[1])

    @property
    def x(self) -> np.ndarray:
        return self._x

    @property
    def components(self) -> np.ndarray:
        """Returns 2D complex128 array of shape (num_channels, num_points)."""
        return self._components

    @property
    def energy(self) -> float:
        return self._energy

    @property
    def channel_labels(self) -> List[str]:
        return list(self._channel_labels)

    def channel(self, index: int) -> np.ndarray:
        """Returns the 1D wavefunction component psi_index(x)."""
        if index < 0 or index >= self.num_channels:
            raise IndexError(f"Channel index {index} out of bounds for {self.num_channels} channels.")
        return self._components[index]

    @property
    def channel_probability_densities(self) -> np.ndarray:
        """Returns per-channel probability densities |psi_c(x)|^2 of shape (num_channels, num_points)."""
        return np.abs(self._components) ** 2

    @property
    def total_probability_density(self) -> np.ndarray:
        """Returns total probability density sum_c |psi_c(x)|^2 of shape (num_points,)."""
        return np.sum(self.channel_probability_densities, axis=0)

    @property
    def channel_populations(self) -> np.ndarray:
        """
        Calculates occupation probability P_c = integral |psi_c(x)|^2 dx for each channel.
        Sums to 1.0 across all channels.
        """
        densities = self.channel_probability_densities
        if self._integration_method == IntegrationMethod.SIMPSON:
            pops = integrate.simpson(y=densities, x=self._x, axis=1)
        else:
            pops = integrate.trapezoid(y=densities, x=self._x, axis=1)
        return np.asarray(pops, dtype=np.float64)

    @property
    def total_norm(self) -> float:
        """Returns complete state L2 norm sqrt(integral Psi^dagger(x) Psi(x) dx)."""
        integral = self._integrate_total_density(self._components, self._x, self._integration_method)
        return float(np.sqrt(max(0.0, integral)))

    def inner_product(self, other: "MultiComponentWavefunction") -> complex:
        """
        Computes coupled-state inner product:
            <Psi_self | Psi_other> = sum_c integral psi_{self,c}^*(x) * psi_{other,c}(x) dx
        """
        if not isinstance(other, MultiComponentWavefunction):
            raise TypeError("Expected a MultiComponentWavefunction instance.")
        if self.num_channels != other.num_channels:
            raise ValueError(
                f"Channel count mismatch: {self.num_channels} != {other.num_channels}."
            )
        if self._x.shape != other._x.shape or not np.allclose(self._x, other._x):
            raise ValueError("Spatial grids must match to compute inner product.")

        integrand = np.sum(np.conj(self._components) * other._components, axis=0)
        if self._integration_method == IntegrationMethod.SIMPSON:
            return complex(integrate.simpson(y=integrand, x=self._x))
        return complex(integrate.trapezoid(y=integrand, x=self._x))

    def overlap_probability(self, other: "MultiComponentWavefunction") -> float:
        """Returns transition/overlap probability |<Psi_self | Psi_other>|^2."""
        return float(abs(self.inner_product(other)) ** 2)

    def is_orthogonal_to(
        self,
        other: "MultiComponentWavefunction",
        tolerance: float = 1e-4
    ) -> bool:
        """Verifies whether |<Psi_self | Psi_other>| <= tolerance."""
        return abs(self.inner_product(other)) <= tolerance

    @classmethod
    def from_array(
        cls,
        x: np.ndarray,
        components: np.ndarray,
        energy: float = 0.0,
        channel_labels: Optional[Sequence[str]] = None,
        normalize: bool = True
    ) -> "MultiComponentWavefunction":
        return cls(
            x=x,
            components=components,
            energy=energy,
            channel_labels=channel_labels,
            normalize=normalize
        )

    @classmethod
    def from_single_channel(
        cls,
        wf: Wavefunction,
        num_channels: int,
        target_channel: int = 0,
        channel_labels: Optional[Sequence[str]] = None
    ) -> "MultiComponentWavefunction":
        """Embeds a single-channel Wavefunction into `target_channel` of an N-channel state."""
        if num_channels < 1:
            raise ValueError("num_channels must be >= 1.")
        if target_channel < 0 or target_channel >= num_channels:
            raise IndexError(f"target_channel {target_channel} out of bounds for {num_channels} channels.")

        comps = np.zeros((num_channels, len(wf.x)), dtype=np.complex128)
        comps[target_channel, :] = wf.psi
        return cls(
            x=wf.x,
            components=comps,
            energy=wf.energy,
            channel_labels=channel_labels,
            normalize=True
        )

    @classmethod
    def superposition(
        cls,
        states: Sequence["MultiComponentWavefunction"],
        coefficients: Sequence[complex],
        channel_labels: Optional[Sequence[str]] = None
    ) -> "MultiComponentWavefunction":
        """
        Constructs a normalized linear superposition of compatible MultiComponentWavefunctions:
            Psi(x) = sum_k c_k * Psi_k(x)
        """
        if not states:
            raise ValueError("superposition requires at least one MultiComponentWavefunction.")
        if len(states) != len(coefficients):
            raise ValueError("Number of states must match number of coefficients.")

        base = states[0]
        coeff_arr = np.asarray(coefficients, dtype=np.complex128)
        if not np.all(np.isfinite(coeff_arr)) or np.linalg.norm(coeff_arr) <= 1e-15:
            raise ValueError("Coefficients must be finite with at least one non-zero amplitude.")

        combined = np.zeros_like(base.components, dtype=np.complex128)
        weight_sum = 0.0
        weighted_energy = 0.0

        for st, c in zip(states, coeff_arr):
            if st.num_channels != base.num_channels:
                raise ValueError("All states in superposition must have identical channel counts.")
            if st.x.shape != base.x.shape or not np.allclose(st.x, base.x):
                raise ValueError("All states in superposition must share the same spatial grid.")
            combined += c * st.components
            mag2 = float(abs(c) ** 2)
            weight_sum += mag2
            weighted_energy += mag2 * st.energy

        labels = channel_labels if channel_labels is not None else base.channel_labels
        return cls(
            x=base.x,
            components=combined,
            energy=weighted_energy / weight_sum,
            channel_labels=labels,
            normalize=True
        )
