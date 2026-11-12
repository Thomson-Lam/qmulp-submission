"""The limiting distribution of the measured register: the Cesàro limit of the source marginal.

The score the walk reports is a time average of the measured distribution,
``(1/tau) sum_{t<tau} p_t``, and that average converges even though the state
of a unitary walk oscillates and never does. Its limit is

    omega[j] = lim_{tau->inf} (1/tau) sum_{t<tau} p_t[j],
    p_t[j]   = sum_k |<j,k| U^t |Psi_0>|^2,

and for a unitary ``U = sum_lambda lambda P_lambda`` over its *distinct*
eigenvalues the cross terms ``lambda^t conj(mu)^t`` with ``lambda != mu``
average to zero, so

    omega[j] = sum_lambda sum_k |<j,k| P_lambda |Psi_0>|^2.                  (L)

This is the limiting distribution of Aharonov, Ambainis, Kempe and Vazirani
(2001). It is **not** the projection onto the eigenvalue-one space alone,
which is the limit of the averaged *state*, a different object: every distinct
eigenvalue contributes its own squared projection.

Why it is an N x N problem
--------------------------
Let ``A`` be the ``N^2 x N`` isometry ``A|x> = sum_j x_j |psi_j>`` with
``|psi_j> = |j> (x) |g_j>`` and ``|g_j> = sum_k sqrt(G[k,j]) |k>``, so that
``Pi = A A^†`` and ``Psi_0 = A sqrt(v)``. The discriminant

    D = A^† S A,   D[j,k] = sqrt(G[j,k] G[k,j])

is real symmetric with eigenvalues ``d`` in ``[-1, 1]``. For each eigenvector
``x_l`` put ``a_l = A x_l`` and ``b_l = S a_l``: ``<a_l|b_l> = d_l``,
``Pi a_l = a_l`` and ``Pi b_l = d_l a_l``, so ``V_l = span{a_l, b_l}`` is
invariant under ``S`` and ``Pi``, hence under ``W = S (2 Pi - I)`` and
``U = W^2``; the ``V_l`` are mutually orthogonal and contain
``Psi_0 = sum_l c_l a_l`` with ``c = X^T sqrt(v)``. Inside ``V_l`` with
``|d_l| < 1`` and ``cos theta_l = d_l``, ``W`` has eigenvalues
``exp(+-i theta_l)`` on

    u_l^+- = (a_l - exp(+-i theta_l) b_l) / (sqrt(2) sin theta_l),

and ``U`` has ``exp(+-2i theta_l)`` on the same vectors. With ``|d_l| = 1``,
``b_l = +-a_l`` and ``U a_l = a_l``. Different ``l`` can share one ``U``
eigenvalue (``d`` and ``-d``; ``d = 0``; every ``|d| = 1``), and components
with the same eigenvalue belong to one ``P_lambda``: they are summed as
vectors *before* squaring. That grouping is the one subtle step.

On a lumped operator the role of ``D`` is played by the discriminant the
router chose (``quantum.lumping_router``): ``K = D-bar`` or the compressed
``K = H``. Every formula below takes ``K``; passing nothing uses ``D`` of the
operator handed in, which is only right for a full (unlumped) operator.

Coordinates
-----------
A vector of the busy subspace is ``Phi = A y + S A z`` with ``y, z`` in
``C^N``; as the ``Psi[source, target]`` matrix of
``quantum.inspired.szegedy_walk``,

    Phi[j, k] = y[j] M[k, j] + M[j, k] z[k],

with inner product ``y^† y' + z^† z' + y^† K z' + z^† K y'``. The source
marginal ``sum_k |Phi[j,k]|^2`` has the closed form of :func:`closed_row_mass`
(``sum_k G[k,j] = 1`` removes one factor); :func:`_direct_row_mass` builds
``Phi`` and squares it, and ``tests/test_mechanism.py`` asserts the two agree.

Convergence constant
--------------------
``||omega-bar_tau - omega||_1 <= c / tau`` with
``c = 2 sum_{lambda != mu active} ||Phi_lambda|| ||Phi_mu|| / |1 - lambda conj(mu)|``
over ordered pairs of active groups (``||P_lambda Psi_0||^2 > active_tol``);
``|1 - lambda conj(mu)|`` is the chord between the two eigenvalues, and
``c <= 2 (r - 1) / delta_chord`` with ``r`` active groups and ``delta_chord``
the smallest chord between active groups.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from quantum.lumped_operator import borrower_layer_positions, unfold_class_scores
from quantum.lumping_router import RoutedOperator, discriminant

TWO_PI = 2.0 * np.pi
REGISTERS = ("source", "target")
# A group whose squared overlap with the initial state is below this is inactive.
ACTIVE_TOL = 1e-14
# A discriminant eigenvalue that is *not* folded into the eigenvalue-one block (1 - |d| >= d_tol) but
# lies this close to the edge makes the chord between its eigenphases tiny and the convergence
# constant huge; the limit warns (it never fails on it). Diagnostic only: no number depends on it.
INNER_GAP_WARN = 1e-6


@dataclass(frozen=True)
class LimitResult:
    """The limiting distributions of both registers and the spectrum behind them."""

    # Source-register limit over compact positions; sums to one.
    omega: np.ndarray
    # Target-register limit over compact positions; sums to one.
    omega_target: np.ndarray
    # One U-eigenphase in [0, 2 pi) per distinct eigenvalue met by Psi_0.
    phases: np.ndarray
    # ||P_lambda Psi_0||^2 per distinct eigenvalue; sums to one.
    group_masses: np.ndarray
    # Every distinct eigenvalue the spectrum allows, active or not.
    n_groups_all: int
    # Smallest circular distance (angle) between any two distinct eigenphases.
    min_angle_gap_all: float
    # Groups the initial state actually reaches, and the gaps among them.
    r_active: int
    delta_chord_active: float
    delta_angle_active: float
    # The exact convergence constant and its spectral bound.
    c_exact: float
    c_bound: float
    # Tolerances the grouping and the activity threshold used.
    d_tol: float
    phase_tol: float
    active_tol: float
    # Discriminant (K) eigenvalues, ascending.
    d: np.ndarray
    # The |d| = 1 collapse, reported (see :func:`collapse_diagnostics`): the largest |d| among the
    # eigenvalues kept strictly inside the unit interval and its distance 1 - |d| from the edge, the
    # number of eigenvalues folded into the eigenvalue-one block and the largest 1 - |d| among them.
    d_inner_max: float = float("nan")
    d_gap_min: float = float("nan")
    d_collapsed: int = 0
    d_collapse_dev_max: float = float("nan")


@dataclass(frozen=True)
class LimitingScores:
    """The limit on one lumped operator, unfolded to borrowers for both registers."""

    G: np.ndarray
    v: np.ndarray
    K: np.ndarray
    route: str
    limit: LimitResult
    borrower_ids: np.ndarray
    omega_source: np.ndarray
    omega_target: np.ndarray
    layer_positions: list


def _check_register(register: str) -> None:
    if register not in REGISTERS:
        raise ValueError(f"register must be one of {REGISTERS}; got {register!r}")


def _probability(vector, n: int, name: str, tol: float) -> np.ndarray:
    out = np.asarray(vector, dtype=float)
    if out.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},); got {out.shape}")
    if out.min() < 0.0 or abs(out.sum() - 1.0) > tol:
        raise ValueError(f"{name} must be nonnegative and sum to one")
    return out


def _components(K: np.ndarray, sv: np.ndarray, *, d_tol: float):
    """One ``(phase, Y, Z)`` per eigenvector of ``U`` on the busy subspace, scaled by its overlap.

    ``(Y, Z)`` are the coordinates of ``<u|Psi_0> u`` for the eigenvector ``u``
    of ``U`` with eigenphase ``phase``; a block with ``|d| >= 1 - d_tol`` is
    one-dimensional and carries eigenvalue ``+1``.
    """
    d, X = np.linalg.eigh(K)
    c = X.T @ sv
    Ksv = K @ sv
    n = K.shape[0]
    phases: list[float] = []
    Ys: list[np.ndarray] = []
    Zs: list[np.ndarray] = []
    for index in range(n):
        dl = float(np.clip(d[index], -1.0, 1.0))
        x = X[:, index].astype(complex)
        if 1.0 - abs(dl) < d_tol:
            phases.append(0.0)
            Ys.append(c[index] * x)
            Zs.append(np.zeros(n, dtype=complex))
            continue
        theta = float(np.arccos(dl))
        scale = 1.0 / (np.sqrt(2.0) * np.sin(theta))
        for sign in (1.0, -1.0):
            y = scale * x
            z = -np.exp(1j * sign * theta) * scale * x
            overlap = np.conj(y) @ sv + np.conj(z) @ Ksv
            phases.append(float((2.0 * sign * theta) % TWO_PI))
            Ys.append(overlap * y)
            Zs.append(overlap * z)
    return np.asarray(phases), np.asarray(Ys), np.asarray(Zs), d


def _group(phases: np.ndarray, Ys: np.ndarray, Zs: np.ndarray, *, phase_tol: float):
    """Sum the components of one distinct ``U``-eigenvalue as vectors, before squaring."""
    order = np.argsort(phases, kind="stable")
    members: list[list[int]] = []
    refs: list[float] = []
    for index in order:
        if members and phases[index] - refs[-1] < phase_tol:
            members[-1].append(int(index))
        else:
            members.append([int(index)])
            refs.append(float(phases[index]))
    # A phase just below 2 pi is the eigenvalue of phase 0.
    if len(members) > 1 and refs[0] + TWO_PI - refs[-1] < phase_tol:
        members[0].extend(members.pop())
        refs.pop()
    Yg = np.stack([Ys[m].sum(axis=0) for m in members])
    Zg = np.stack([Zs[m].sum(axis=0) for m in members])
    return np.asarray(refs), Yg, Zg


def closed_row_mass(Y, Z, G: np.ndarray, K: np.ndarray, *, register: str = "source"):
    """``sum_k |Phi[j,k]|^2`` (source) or ``sum_j |Phi[j,k]|^2`` (target) for ``Phi = A Y + S A Z``.

    Closed form: ``|Y|^2 + G |Z|^2 + 2 Re(conj(Y) * (K Z))`` on the source
    register, using ``sum_k G[k,j] = 1``; ``G |Y|^2 + |Z|^2 + 2 Re(conj(Z) * (K Y))``
    on the target register, using ``sum_j G[j,k] = 1`` (``G`` is column-stochastic,
    so both marginals carry ``G`` itself). ``Y`` and ``Z`` may be one vector or a stack of
    them, one row per group; the result has the same leading shape. This is
    the one marginal formula the limit and the running average share.
    """
    _check_register(register)
    Y2, Z2 = np.atleast_2d(Y), np.atleast_2d(Z)
    absY, absZ = np.abs(Y2) ** 2, np.abs(Z2) ** 2
    if register == "source":
        out = absY + absZ @ G.T + 2.0 * np.real(np.conj(Y2) * (Z2 @ K))
    else:
        out = absY @ G.T + absZ + 2.0 * np.real(np.conj(Z2) * (Y2 @ K))
    return out[0] if np.ndim(Y) == 1 else out


def _direct_row_mass(Y, Z, M: np.ndarray, *, register: str = "source") -> np.ndarray:
    """The same marginal by building ``Phi[j,k] = Y[j] M[k,j] + M[j,k] Z[k]`` and squaring it."""
    _check_register(register)
    Phi = np.asarray(Y)[:, np.newaxis] * M.T + M * np.asarray(Z)[np.newaxis, :]
    power = np.abs(Phi) ** 2
    return power.sum(axis=1) if register == "source" else power.sum(axis=0)


def _group_masses(Yg: np.ndarray, Zg: np.ndarray, K: np.ndarray) -> np.ndarray:
    """``||A Y + S A Z||^2 = Y^†Y + Z^†Z + 2 Re(Y^† K Z)`` per row."""
    return np.real(
        (np.abs(Yg) ** 2).sum(axis=1)
        + (np.abs(Zg) ** 2).sum(axis=1)
        + 2.0 * (np.conj(Yg) * (Zg @ K)).sum(axis=1)
    )


def _min_circular_gap(phases: np.ndarray) -> float:
    if phases.size < 2:
        return float("nan")
    ordered = np.sort(phases)
    gaps = np.diff(ordered)
    wrap = ordered[0] + TWO_PI - ordered[-1]
    return float(min(float(gaps.min()), float(wrap)))


def collapse_diagnostics(d: np.ndarray, d_tol: float) -> dict[str, float]:
    """How the ``|d| = 1`` collapse treated the spectrum of ``K``.

    An eigenvalue with ``1 - |d| < d_tol`` is folded into the eigenvalue-one block of ``U`` (see
    :func:`_components`); every other one contributes a pair ``exp(+-2 i theta)``. Returns
    ``d_inner_max`` (the largest ``|d|`` among the eigenvalues kept inside), ``d_gap_min = 1 -
    d_inner_max``, ``d_collapsed`` (how many were folded) and ``d_collapse_dev_max`` (the largest ``1 -
    |d|`` among the folded ones: how far from exactly ``+-1`` the fold reached). Values with nothing
    to report are ``nan``.
    """
    gap = 1.0 - np.abs(np.clip(np.asarray(d, dtype=float), -1.0, 1.0))
    inner = gap >= float(d_tol)
    return {
        "d_inner_max": float(1.0 - gap[inner].min()) if inner.any() else float("nan"),
        "d_gap_min": float(gap[inner].min()) if inner.any() else float("nan"),
        "d_collapsed": int((~inner).sum()),
        "d_collapse_dev_max": float(gap[~inner].max()) if (~inner).any() else float("nan"),
    }


def convergence_constant(phases: np.ndarray, masses: np.ndarray, *, active_tol: float) -> dict[str, float]:
    """``r_active``, the chord and angle gaps over active groups, ``c_exact`` and ``c_bound``."""
    active = np.flatnonzero(masses > float(active_tol))
    r = int(active.size)
    if r < 2:
        return {"r_active": r, "delta_chord_active": float("nan"), "delta_angle_active": float("nan"),
                "c_exact": 0.0, "c_bound": 0.0}
    norms = np.sqrt(masses[active])
    diff = phases[active][:, np.newaxis] - phases[active][np.newaxis, :]
    chord = 2.0 * np.abs(np.sin(0.5 * diff))
    off = ~np.eye(r, dtype=bool)
    c_exact = float(2.0 * np.sum(np.outer(norms, norms)[off] / chord[off]))
    delta_chord = float(chord[off].min())
    return {
        "r_active": r,
        "delta_chord_active": delta_chord,
        "delta_angle_active": _min_circular_gap(phases[active]),
        "c_exact": c_exact,
        "c_bound": float(2.0 * (r - 1) / delta_chord),
    }


def limiting_distribution(
    G: np.ndarray,
    v: np.ndarray,
    *,
    d_tol: float,
    phase_tol: float,
    mass_tol: float,
    K: np.ndarray | None = None,
    initial_amplitudes: np.ndarray | None = None,
    active_tol: float = ACTIVE_TOL,
) -> LimitResult:
    """The Cesàro limit (L) of both register marginals, from one ``N x N`` eigendecomposition.

    ``K`` is the discriminant the walk runs on: the router's choice on a lumped
    operator, or (when omitted) ``D`` of ``G`` itself, which is only right for a
    full operator. ``initial_amplitudes`` replaces ``v`` in
    ``Psi_0 = sum_j sqrt(a_j) |psi_j>`` while the operator, and so its restart,
    is unchanged (the initial-state ablation); it defaults to ``v``.
    """
    G = np.asarray(G, dtype=float)
    n = G.shape[0]
    if G.ndim != 2 or G.shape[1] != n:
        raise ValueError(f"G must be square; got shape {G.shape}")
    v = _probability(v, n, "v", mass_tol)
    start = v if initial_amplitudes is None else _probability(initial_amplitudes, n, "a", mass_tol)
    K = discriminant(G) if K is None else np.asarray(K, dtype=float)
    if K.shape != (n, n) or not np.allclose(K, K.T, atol=1e-12):
        raise ValueError("K must be a real symmetric matrix of the operator's size")

    phases, Ys, Zs, d = _components(K, np.sqrt(start), d_tol=d_tol)
    spectrum = collapse_diagnostics(d, d_tol)
    if spectrum["d_gap_min"] < INNER_GAP_WARN:
        warnings.warn(
            f"the discriminant has an eigenvalue with |d| = 1 - {spectrum['d_gap_min']:.3e}: inside the unit interval "
            f"(d_tol = {float(d_tol):g}) but closer to it than {INNER_GAP_WARN:g}; its eigenphase chord is tiny and the "
            "running average converges very slowly",
            RuntimeWarning, stacklevel=2,
        )
    group_phases, Yg, Zg = _group(phases, Ys, Zs, phase_tol=phase_tol)
    masses = _group_masses(Yg, Zg, K)
    omega = closed_row_mass(Yg, Zg, G, K, register="source").sum(axis=0)
    omega_target = closed_row_mass(Yg, Zg, G, K, register="target").sum(axis=0)

    for name, total in (
        ("group masses", masses.sum()),
        ("source limit", omega.sum()),
        ("target limit", omega_target.sum()),
    ):
        if abs(float(total) - 1.0) > mass_tol:
            raise AssertionError(
                f"the {name} sum to {float(total)!r}, not one within {mass_tol}; the eigenvalue "
                "grouping or the discriminant is wrong"
            )
    if omega.min() < -mass_tol or omega_target.min() < -mass_tol:
        raise AssertionError("the limiting distribution has a negative entry")
    constant = convergence_constant(group_phases, masses, active_tol=active_tol)
    return LimitResult(
        omega=np.maximum(omega, 0.0),
        omega_target=np.maximum(omega_target, 0.0),
        phases=group_phases,
        group_masses=masses,
        n_groups_all=int(group_phases.size),
        min_angle_gap_all=_min_circular_gap(group_phases),
        d_tol=float(d_tol),
        phase_tol=float(phase_tol),
        active_tol=float(active_tol),
        d=d,
        **constant,
        **spectrum,
    )


def limiting_scores(nodes, routed: RoutedOperator, config, *, initial_amplitudes=None) -> LimitingScores:
    """The limiting QPPR on one routed lumped operator, unfolded to borrowers.

    ``routed`` carries ``(G, v, K)`` and the route; ``nodes`` is the lumped
    frame the operator was built on, which the unfolding reads.
    """
    limit = limiting_distribution(
        routed.G, routed.v, K=routed.K, d_tol=config.d_tol, phase_tol=config.phase_tol,
        mass_tol=config.mass_tol, initial_amplitudes=initial_amplitudes,
    )
    ids, source = unfold_class_scores(nodes, limit.omega)
    _ids, target = unfold_class_scores(nodes, limit.omega_target)
    return LimitingScores(
        G=routed.G, v=routed.v, K=routed.K, route=routed.route, limit=limit, borrower_ids=ids,
        omega_source=source, omega_target=target, layer_positions=borrower_layer_positions(nodes),
    )
