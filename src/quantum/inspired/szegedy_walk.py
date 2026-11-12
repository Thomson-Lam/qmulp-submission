"""
The Szegedy quantum walk over the MuLP Google matrix

    G = alpha * T + (1 - alpha) * P,

where T is the column-normalized MuLP supra-transition matrix and P is the full
MuLP restart matrix: each defaulter's column restarts onto its own layer copies
(the influence structure), and every remaining column falls back to the
validated personalization vector v (see ``quantum.inspired.google_matrix``).
``QuantumWalkRanker.buildSzegedyWalk`` passes P in via
``build_restart_matrix_from_influence``, matching
``classical.mulp.MultiLayerRanker.pageRank`` operator for operator: the
manuscript's central claim is a *controlled* comparison in which the classical
and quantum arms differ only in the diffusion, so both are built on the same G.

``SzegedyWalk`` supports two representations of the same walk. The default
constructor accepts a materialized dense ``G`` for small graphs and Qiskit
parity references. The full graph uses the matrix-free Google-operator
constructor ``SzegedyWalk.implicit(...)``, which combines sparse ``T``, restart
data, and ``v`` without materializing dense ``G``. Here "matrix-free" applies to
``G``, ``sqrt(G)``, and the ``N²×N²`` walk operator; the evolving state remains
an ``N×N`` matrix. Only the two methods that touch ``G`` (``_build_initial_state``,
``_apply_R_Pi``) branch on the mode; everything else operates on ``Psi``.

State representation:
  Psi[source, target] = <source| ⊗ <target| Psi_t>

The frozen research specification: source-register measurement, W^2 per
recorded step, personalized initialization from v, and the reported score is
the time average over ``t = 0 .. tau-1`` including ``t = 0``. The borrower score
aggregates the borrower's layer-copy supra-nodes.

Siblings: ``dense_reference`` (explicit ``Pi``, ``S_swap``, ``W`` for tiny
graphs), ``walk_loop`` (the instrumented loop behind ``run``), ``finite_shot``
(the shot estimator and its ensemble), ``checkpoint`` (the resumable runner).
"""

from __future__ import annotations

import numpy as np

from .google_matrix import GoogleMatrixBuilder
from .validation import (
    DEFAULT_MAX_DENSE_BYTES,
    check_dense_allocation,
    check_state_allocation,
    validate_column_stochastic_matrix,
    validate_probability_vector,
)
from .walk_loop import run_loop


class SzegedyWalk:
    """
    Szegedy-type quantum walk diffusion on the MuLP supra-Google matrix.

    Build with the default constructor for materialized dense ``G``, or use
    the matrix-free Google-operator constructor ``implicit`` for full-graph
    execution without materializing dense G.

    Parameters
    ----------
    G : np.ndarray, shape (n, n)
        Column-stochastic personalized Google matrix.
    v : np.ndarray, shape (n,)
        Personalization vector (sums to 1).
    borrower_layer1_indices, borrower_layer2_indices : array-like of int
        Supra-node indices of borrower copies in the MuLP layers; the second
        is ``None`` on a single-layer network.
    measure_register : str
        Default register used by ``_marginal_supra_node_probs``/
        ``_marginal_borrower_probs`` when no per-call override is given;
        "source" (default) matches the frozen research specification.
    Every recorded time step applies W².
    """

    def __init__(
        self,
        G: np.ndarray,
        v: np.ndarray,
        borrower_layer1_indices: np.ndarray,
        borrower_layer2_indices: "np.ndarray | None" = None,
        *,
        measure_register: str = "source",
        validate_inputs: bool = True,
        atol: float = 1e-8,
        max_dense_bytes: int | None = DEFAULT_MAX_DENSE_BYTES,
    ) -> None:
        self.G = np.asarray(G, dtype=np.float64)
        self.v = np.asarray(v, dtype=np.float64)
        self.n = self.G.shape[0]
        self.atol = atol

        if validate_inputs:
            self.G = validate_column_stochastic_matrix(self.G, name="G", atol=atol)
            self.v = validate_probability_vector(self.v, name="v", n=self.n, atol=atol)
        elif self.G.ndim != 2 or self.G.shape[0] != self.G.shape[1]:
            raise ValueError(f"G must be square; got shape={self.G.shape}")

        check_dense_allocation("sqrt(G)", self.G.shape, np.float64, max_dense_bytes)
        check_dense_allocation("state matrix Psi", self.G.shape, np.complex128, max_dense_bytes)
        # Reports the full working set, not one copy: a job sized for one is
        # killed mid-walk with no diagnosis.
        self.state_allocation = check_state_allocation(self.n)

        self._mode = "dense"
        self._operator = None
        self._G_sqrt = np.sqrt(np.maximum(self.G, 0.0))
        # Only the matrix-free constructor records builder diagnostics.
        self.operator_diagnostics = None

        self._init_borrower_indices(borrower_layer1_indices, borrower_layer2_indices)
        self.measure_register = measure_register

    @classmethod
    def implicit(
        cls,
        T,
        v: np.ndarray,
        borrower_layer1_indices: np.ndarray,
        borrower_layer2_indices: "np.ndarray | None" = None,
        *,
        alpha: float = 0.85,
        measure_register: str = "source",
        atol: float = 1e-8,
        max_dense_bytes: int | None = DEFAULT_MAX_DENSE_BYTES,
        restart_matrix=None,
    ) -> "SzegedyWalk":
        """Build the matrix-free Google-operator walk without dense G."""
        self = cls.__new__(cls)
        builder = GoogleMatrixBuilder()
        operator = builder.build_implicit(
            T,
            v,
            alpha=alpha,
            atol=atol,
            restart_matrix=restart_matrix,
        )
        self.operator_diagnostics = builder.diagnostics
        self.n = operator.n
        self.atol = atol

        check_dense_allocation("state matrix Psi", (self.n, self.n), np.complex128, max_dense_bytes)
        self.state_allocation = check_state_allocation(self.n)

        self.G = None
        self._mode = "implicit"
        self._operator = operator
        self._G_sqrt = None
        self.v = validate_probability_vector(operator.v, name="v", n=self.n, atol=atol)

        self._init_borrower_indices(borrower_layer1_indices, borrower_layer2_indices)
        self.measure_register = measure_register
        return self

    def _init_borrower_indices(
        self,
        borrower_layer1_indices: np.ndarray,
        borrower_layer2_indices: "np.ndarray | None" = None,
    ) -> None:
        """Record the supra-node index array for each borrower layer copy.

        Two arrays for the paper's two-layer network; ``None`` for the second
        on a single-layer network, where a borrower has one copy only.
        """
        layers = [np.asarray(borrower_layer1_indices, dtype=int)]
        if borrower_layer2_indices is not None:
            layers.append(np.asarray(borrower_layer2_indices, dtype=int))

        for position, arr in enumerate(layers):
            if arr.ndim != 1:
                raise ValueError("borrower layer indices must be 1-dimensional")
            if len(arr) != len(layers[0]):
                raise ValueError("borrower layer index arrays must have the same length")
            if np.any(arr < 0) or np.any(arr >= self.n):
                raise ValueError(
                    f"borrower layer {position} indices outside [0, {self.n})"
                )
        if len(layers[0]) == 0:
            raise ValueError("at least one borrower index is required")

        self.borrower_layer_indices = layers
        self.bl1 = layers[0]
        self.bl2 = layers[1] if len(layers) > 1 else None
        self.n_layers = len(layers)
        self.n_borrowers = len(layers[0])

    # ------------------------------------------------------------------
    # Core walk mechanics
    # ------------------------------------------------------------------

    def _build_initial_state(self) -> np.ndarray:
        """Build |Psi_0> = sum_j sqrt(v_j) |j>|g_j>."""
        sqrt_v = np.sqrt(self.v)
        if self._mode == "dense":
            Psi = self._G_sqrt * sqrt_v[np.newaxis, :]
            return Psi.T.astype(complex)  # Psi[source, target]

        Psi = np.zeros((self.n, self.n), dtype=complex)
        for source in range(self.n):
            Psi[source, :] = sqrt_v[source] * self._operator.sqrt_column(source)
        return Psi

    def _apply_R_Pi(self, Psi: np.ndarray) -> np.ndarray:
        """Apply R_Pi = 2*Pi - I without materializing the projector."""
        if self._mode == "dense":
            c = (self._G_sqrt.T * Psi).sum(axis=1)
            Pi_Psi = c[:, np.newaxis] * self._G_sqrt.T
            return 2.0 * Pi_Psi - Psi

        out = np.empty_like(Psi)
        for source in range(self.n):
            g = self._operator.sqrt_column(source)
            c = np.vdot(g, Psi[source, :])
            out[source, :] = 2.0 * c * g - Psi[source, :]
        return out

    def _apply_W(self, Psi: np.ndarray) -> np.ndarray:
        """Apply one step of W = S_swap * R_Pi."""
        return self._apply_R_Pi(Psi).T

    def _advance(self, Psi: np.ndarray) -> np.ndarray:
        """Apply the fixed W² evolution for one recorded step."""
        return self._apply_W(self._apply_W(Psi))

    def _marginal_supra_node_probs(
        self, Psi: np.ndarray, *, register: str | None = None
    ) -> np.ndarray:
        """Return supra-node marginal probabilities for target or source register.

        Register naming follows the paper: the *first* factor of H = C^n (x) C^n
        is the "position" register (which node the walker is at, i.e. ``source``
        in this module's ``Psi[source, target]`` convention); the *second*
        factor is the "destination"/target register. Eq. 17 of the paper fixes
        the position register and marginalizes over the destination register,
        i.e. it is the ``source``/``position`` marginal, not ``target``.

        Defaults to this walk's ``measure_register`` when ``register`` is not
        given explicitly.
        """
        if register is None:
            register = self.measure_register
        power = np.abs(Psi) ** 2
        if register in {"target", "second", "destination"}:
            return power.sum(axis=0)
        if register in {"source", "first", "position"}:
            return power.sum(axis=1)
        raise ValueError(
            "register must be 'target'/'second'/'destination' or 'source'/'first'/'position'"
        )

    def _marginal_borrower_probs(self, Psi: np.ndarray) -> np.ndarray:
        """Source-register borrower probabilities summed over all L layer copies."""
        probs_all = self._marginal_supra_node_probs(Psi)
        total = probs_all[self.borrower_layer_indices[0]].copy()
        for layer_indices in self.borrower_layer_indices[1:]:
            total += probs_all[layer_indices]
        return total

    def _assert_probability_normalized(self, Psi: np.ndarray, *, context: str) -> None:
        total = float(self._marginal_supra_node_probs(Psi, register="source").sum())
        if not np.isclose(total, 1.0, atol=max(self.atol, 1e-10)):
            raise AssertionError(f"State probability not normalized at {context}: total={total}")

    def supra_node_history(
        self,
        tau: int,
        *,
        register: str | None = None,
        validate_probabilities: bool = False,
    ) -> np.ndarray:
        """Exact per-step supra-node marginals, shape ``(tau, n)`` -- not borrower-aggregated.

        The same exact matrix-free evolution as :meth:`run`, recording the raw
        supra-node marginal at each recorded step (before advancing) instead of
        aggregating to borrower scores. The lumped-operator reference and the
        Aer arms are scored against it.
        """
        if tau <= 0:
            raise ValueError(f"tau must be positive; got {tau}")
        if register is None:
            register = self.measure_register
        Psi = self._build_initial_state()
        history = np.zeros((tau, self.n), dtype=float)
        for t in range(tau):
            if validate_probabilities:
                self._assert_probability_normalized(Psi, context=f"step {t}")
            history[t] = self._marginal_supra_node_probs(Psi, register=register)
            Psi = self._advance(Psi)
        return history

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        tau: int = 50,
        verbose: bool = False,
        *,
        validate_probabilities: bool = False,
    ) -> np.ndarray:
        """Time-averaged borrower scores ``(1/tau) sum_{t<tau} q_t``, ``t = 0`` included."""
        omega, _ = run_loop(
            self,
            tau,
            validate_probabilities=validate_probabilities,
            collect_history=False,
            verbose=verbose,
        )
        return omega

    def run_with_history(
        self,
        tau: int = 50,
        *,
        validate_probabilities: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (time-average borrower scores, per-step borrower history)."""
        omega, history = run_loop(
            self,
            tau,
            validate_probabilities=validate_probabilities,
            collect_history=True,
        )
        return omega, history
