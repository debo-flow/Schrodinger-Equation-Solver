from typing import Literal, Optional
import numpy as np
from scipy import sparse
from schrodinger.numerical.grid import SpatialGrid
from schrodinger.numerical.finite_difference import FiniteDifferenceEngine
from schrodinger.operators.kinetic_operator import KineticEnergyOperator
from schrodinger.operators.operator_validation import OperatorValidator
from schrodinger.analysis.hermiticity import HermiticityAnalyzer
from .coupled_potential import CoupledPotential


class CoupledHamiltonian:
    """
    Constructs the N-channel block Hamiltonian matrix:
        H = T (x) I_N + V(x)
    in channel-major ordering of dimension (N * M_act) x (N * M_act).
    Supports sparse CSR and dense representations while preserving Hermiticity.
    """

    def __init__(
        self,
        grid: SpatialGrid,
        coupled_potential: CoupledPotential,
        mass: float = 1.0,
        hbar: float = 1.0,
        unit_system: Literal["dimensionless", "si"] = "dimensionless",
        hermitian_tolerance: float = 1e-10
    ):
        OperatorValidator.validate_physical_constants(mass, hbar)
        if not isinstance(coupled_potential, CoupledPotential):
            raise TypeError(f"Expected CoupledPotential instance. Received: {type(coupled_potential)}")
        OperatorValidator.validate_unit_consistency(unit_system, coupled_potential.unit_system)

        self.grid = grid
        self.coupled_potential = coupled_potential
        self.num_channels = coupled_potential.num_channels
        self.mass = float(mass)
        self.hbar = float(hbar)
        self.unit_system = unit_system
        self.hermitian_tolerance = float(hermitian_tolerance)

        self._fd_engine = FiniteDifferenceEngine(grid)
        self._kinetic_op = KineticEnergyOperator(
            self._fd_engine,
            mass=self.mass,
            hbar=self.hbar,
            unit_system=self.unit_system
        )
        self._hermiticity_analyzer = HermiticityAnalyzer(tolerance=self.hermitian_tolerance)

    def build_kinetic_block(
        self,
        accuracy_order: int = 2,
        interior_only: bool = True
    ) -> sparse.csr_matrix:
        """Builds I_N (x) T as a block-diagonal sparse CSR matrix."""
        t_single = self._kinetic_op.build_matrix(
            accuracy_order=accuracy_order,
            interior_only=interior_only
        )
        identity_channels = sparse.eye(self.num_channels, format="csr", dtype=t_single.dtype)
        return sparse.kron(identity_channels, t_single, format="csr")

    def build_potential_block(self, interior_only: bool = True) -> sparse.csr_matrix:
        """
        Builds the N x N block potential matrix where block (i, j) is
        diag(V_{i,j}(x_0), ..., V_{i,j}(x_{M_act-1})).
        """
        x_pts = self.grid.interior_points if interior_only else self.grid.grid
        v_eval = self.coupled_potential(x_pts)  # Shape: (N, N, M_act)

        if not np.all(np.isfinite(v_eval)):
            raise ValueError(
                "Coupled potential contains non-finite values (Inf/NaN) on active grid nodes. "
                "Use interior_only=True for infinite square well boundaries."
            )

        n = self.num_channels
        m_act = len(x_pts)
        is_complex = np.iscomplexobj(v_eval) and np.any(np.imag(v_eval) != 0.0)
        mat_dtype = np.complex128 if is_complex else self.grid.grid.dtype

        blocks = [[None for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                diag_vals = v_eval[i, j, :].astype(mat_dtype)
                if np.any(diag_vals != 0):
                    blocks[i][j] = sparse.diags(diag_vals, offsets=0, shape=(m_act, m_act), format="csr")
                else:
                    blocks[i][j] = sparse.csr_matrix((m_act, m_act), dtype=mat_dtype)

        return sparse.bmat(blocks, format="csr")

    def build_sparse(
        self,
        accuracy_order: int = 2,
        interior_only: bool = True
    ) -> sparse.csr_matrix:
        """
        Assembles and validates H = T (x) I_N + V(x) as a sparse CSR matrix.
        """
        t_block = self.build_kinetic_block(accuracy_order=accuracy_order, interior_only=interior_only)
        v_block = self.build_potential_block(interior_only=interior_only)

        if t_block.shape != v_block.shape:
            raise ValueError(f"Dimension mismatch: Kinetic {t_block.shape} != Potential {v_block.shape}")

        h_mat = (t_block + v_block).tocsr()

        is_herm, dev = self._hermiticity_analyzer.is_hermitian(h_mat)
        if not is_herm:
            raise ValueError(
                f"Coupled Hamiltonian failed Hermiticity check: max |H - H^dagger| = {dev:.2e}."
            )

        return h_mat

    def build_dense(
        self,
        accuracy_order: int = 2,
        interior_only: bool = True
    ) -> np.ndarray:
        """Assembles H = T (x) I_N + V(x) as a dense NumPy array."""
        return self.build_sparse(accuracy_order=accuracy_order, interior_only=interior_only).toarray()

    def uncoupled_hamiltonian(self, name_suffix: str = "uncoupled") -> "CoupledHamiltonian":
        """Returns a CoupledHamiltonian with identical diagonal potentials and zero coupling."""
        uncoupled_pot = self.coupled_potential.uncoupled_copy()
        uncoupled_pot.name = f"{self.coupled_potential.name}_{name_suffix}"
        return CoupledHamiltonian(
            grid=self.grid,
            coupled_potential=uncoupled_pot,
            mass=self.mass,
            hbar=self.hbar,
            unit_system=self.unit_system,
            hermitian_tolerance=self.hermitian_tolerance
        )
