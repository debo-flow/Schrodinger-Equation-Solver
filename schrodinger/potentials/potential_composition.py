import math
from typing import List, Sequence, Tuple, Optional
import numpy as np
from .base import Potential


class PotentialSum(Potential):
    """
    Represents the algebraic sum of two or more potential functions:
        V_total(x) = sum_i V_i(x)
    Enforces identical unit systems across all constituent potentials.
    """

    def __init__(self, potentials: Sequence[Potential]):
        if not potentials:
            raise ValueError("PotentialSum requires at least one Potential.")
        for pot in potentials:
            if not isinstance(pot, Potential):
                raise TypeError(f"All items must be Potential instances. Received: {type(pot)}")

        base_unit = potentials[0].unit_system
        for pot in potentials[1:]:
            if pot.unit_system != base_unit:
                raise ValueError(
                    f"Unit mismatch in PotentialSum: '{pot.unit_system}' != '{base_unit}'."
                )

        super().__init__(unit_system=base_unit)
        # Flatten nested PotentialSum instances cleanly
        flattened: List[Potential] = []
        for pot in potentials:
            if isinstance(pot, PotentialSum):
                flattened.extend(pot.potentials)
            else:
                flattened.append(pot)
        self.potentials: List[Potential] = flattened

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        total = np.zeros_like(x, dtype=np.float64)
        for pot in self.potentials:
            total = total + pot._evaluate(x)
        return total

    def __repr__(self) -> str:
        return f"PotentialSum({self.potentials!r})"


class ScaledPotential(Potential):
    """
    Represents a scalar-multiplied potential function:
        V_scaled(x) = scale * V(x)
    """

    def __init__(self, potential: Potential, scale: float):
        if not isinstance(potential, Potential):
            raise TypeError(f"Expected a Potential instance. Received: {type(potential)}")
        if isinstance(scale, bool) or not math.isfinite(scale):
            raise ValueError(f"scale must be a finite real number. Received: {scale}")

        super().__init__(unit_system=potential.unit_system)
        if isinstance(potential, ScaledPotential):
            self.potential = potential.potential
            self.scale = float(potential.scale * scale)
        else:
            self.potential = potential
            self.scale = float(scale)

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        if self.scale == 0.0:
            return np.zeros_like(x, dtype=np.float64)
        return self.scale * self.potential._evaluate(x)

    def __repr__(self) -> str:
        return f"ScaledPotential(potential={self.potential!r}, scale={self.scale!r})"


class PiecewisePotential(Potential):
    """
    Constructs a spatial piecewise potential from ordered (x_min, x_max, potential) intervals.
    Points falling outside all specified intervals use `default_potential` (or V=0 if None).
    """

    def __init__(
        self,
        pieces: Sequence[Tuple[float, float, Potential]],
        default_potential: Optional[Potential] = None
    ):
        if not pieces:
            raise ValueError("PiecewisePotential requires at least one (x_min, x_max, potential) piece.")

        unit = pieces[0][2].unit_system
        validated_pieces: List[Tuple[float, float, Potential]] = []

        for idx, (x_start, x_end, pot) in enumerate(pieces):
            if not isinstance(pot, Potential):
                raise TypeError(f"Piece {idx} must contain a Potential instance.")
            if x_end < x_start:
                raise ValueError(f"Piece {idx} has invalid interval [{x_start}, {x_end}].")
            if pot.unit_system != unit:
                raise ValueError(f"Unit mismatch in PiecewisePotential: '{pot.unit_system}' != '{unit}'.")
            validated_pieces.append((float(x_start), float(x_end), pot))

        if default_potential is not None:
            if not isinstance(default_potential, Potential):
                raise TypeError("default_potential must be a Potential instance or None.")
            if default_potential.unit_system != unit:
                raise ValueError(
                    f"Unit mismatch in default_potential: '{default_potential.unit_system}' != '{unit}'."
                )

        super().__init__(unit_system=unit)
        self.pieces = validated_pieces
        self.default_potential = default_potential

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        result = (
            self.default_potential._evaluate(x).copy()
            if self.default_potential is not None
            else np.zeros_like(x, dtype=np.float64)
        )
        assigned = np.zeros(len(x), dtype=bool)

        for x_start, x_end, pot in self.pieces:
            mask = (~assigned) & (x >= x_start) & (x <= x_end)
            if np.any(mask):
                result[mask] = pot._evaluate(x[mask])
                assigned[mask] = True

        return result

    def __repr__(self) -> str:
        return f"PiecewisePotential(pieces={self.pieces!r}, default_potential={self.default_potential!r})"


class ParameterizedPotentialCombination(Potential):
    """
    Parameterized linear combination of multiple basis potentials:
        V(x) = offset + sum_i weights[i] * potentials[i](x)
    Supports dynamic weight updates while preserving unit consistency.
    """

    def __init__(
        self,
        potentials: Sequence[Potential],
        weights: Sequence[float],
        offset: float = 0.0
    ):
        if not potentials:
            raise ValueError("ParameterizedPotentialCombination requires at least one potential.")
        if len(potentials) != len(weights):
            raise ValueError(
                f"Number of potentials ({len(potentials)}) must match number of weights ({len(weights)})."
            )
        if not math.isfinite(offset):
            raise ValueError(f"offset must be finite. Received: {offset}")

        unit = potentials[0].unit_system
        for pot in potentials:
            if not isinstance(pot, Potential):
                raise TypeError(f"Expected Potential instances. Received: {type(pot)}")
            if pot.unit_system != unit:
                raise ValueError(f"Unit mismatch: '{pot.unit_system}' != '{unit}'.")

        super().__init__(unit_system=unit)
        self.potentials: List[Potential] = list(potentials)
        self.weights: List[float] = self._validate_weights(weights)
        self.offset = float(offset)

    @staticmethod
    def _validate_weights(weights: Sequence[float]) -> List[float]:
        clean: List[float] = []
        for w in weights:
            if isinstance(w, bool) or not math.isfinite(w):
                raise ValueError(f"All weights must be finite numbers. Received: {w}")
            clean.append(float(w))
        return clean

    def update_weights(self, new_weights: Sequence[float]) -> None:
        """Updates linear combination coefficients with full validation."""
        if len(new_weights) != len(self.potentials):
            raise ValueError(
                f"Expected {len(self.potentials)} weights, received {len(new_weights)}."
            )
        self.weights = self._validate_weights(new_weights)

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        result = np.full_like(x, self.offset, dtype=np.float64)
        for weight, pot in zip(self.weights, self.potentials):
            if weight != 0.0:
                result = result + weight * pot._evaluate(x)
        return result

    def __repr__(self) -> str:
        return (
            f"ParameterizedPotentialCombination(potentials={self.potentials!r}, "
            f"weights={self.weights!r}, offset={self.offset!r})"
        )
