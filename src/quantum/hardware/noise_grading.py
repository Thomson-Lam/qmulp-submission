"""Grading a simulated output against the exact lumped average, the pure-noise reference, and what the circuit tier's tables mean.

What the tables mean
--------------------
The status of a cell, the three simulated arms, the calibration convention and
every column, in one place, so the instance record, the manifest and the column
definitions of the three result tables of stage 06 say the same thing. Nothing
here describes an execution on a device: every arm is an Aer simulation.

Grading
-------
One implementation of every graded quantity: the supra-node Spearman, the
total variation, its value for the uniform distribution and the fidelity
ratio ``1 - TV / TV_uniform`` (:func:`supra_metrics`), and the Spearman over
the class probabilities of the masked classes (:func:`masked_class_spearman`).
The same functions grade a measured output and every draw of the pure-noise
band, so the band and the output are comparable by construction.

A masked class is a lumped class whose label is hidden (the test borrowers of
a small instance may all sit in one class, where there is nothing to rank). Its
probability is the sum of the exact or measured mass over its layer copies; a
borrower's share of it (the class probability over the class size) would order
the classes by size, not by what the walk assigned them, and is not used.

The pure-noise band answers "is this output distinguishable from a device that
returns random codes?". A pure-noise output is ``shots`` samples of the uniform
distribution over the ``2^m`` register codes, converted exactly as a measured
output is (padding codes dropped, the retained support renormalised), one
independent draw per circuit, averaged over the ``tau`` circuits. It is graded
against **that instance's own exact average at that tau**; the width of its
band depends strongly on the number of supra-nodes ``R``, so there is one band
per instance and recorded step.

Whether an output carries signal is read from the **fidelity ratio** (continuous,
and its band is narrow) by a rule that is one-sided and strict: a raw cell carries
signal exactly when its fidelity ratio is above the upper edge of its own band
(:func:`signal_above_band`). A value on the edge is not signal, and neither is a
value below the band. The Spearman over a handful of supra-nodes takes few values
(at ``k = 2`` a measured value can equal the band edge), so it stays a secondary
column. An instance is judged at its longest simulated walk, the largest recorded
``tau >= hardware_min_tau`` (``tau = 1`` is the state preparation, ``p_0 = v``, never
a walk): :func:`noise_entry_points` reads where the raw output stops carrying signal,
over instances (``noise_k_band``) and over the steps of the selected instance
(``noise_tau_band``). A band is a 5th-95th percentile range, so a pure-noise output
leaves it about one time in ten per cell; no rule here asks for "inside the band at
every step", which would fail by chance.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from evaluation.reporting import spearman, total_variation
from quantum.lumping import UNKNOWN_LABEL


# ---------------------------------------------------------------------------
# What the tables mean
# ---------------------------------------------------------------------------


# The status of a cell that was not simulated, for any resource reason: a register wider than
# ``hardware_max_register_width``, a state larger than ``hardware_density_matrix_max_bytes``, a
# simulation estimated past ``hardware_instance_budget_seconds``, or a task that ended before the cell.
# The reason column says which, with the estimate.
NOT_SIMULATED = "not simulated: resource budget"

ARM_DEFINITIONS = {
    "exact": "the exact lumped time average of the walk, by the coefficient recurrence on the retained operator (NumPy; no circuit)",
    "noiseless_shots": "Aer sampling of the ideal ISA circuits: shot noise only (statevector of the touched qubits, config.hardware_shots shots per circuit)",
    "aer_noise_model": "Aer density-matrix simulation of the ISA circuits under a generic model: depolarizing error on every one-qubit gate "
                       "(rz included) at the target's median one-qubit error, depolarizing error on every two-qubit gate at its median "
                       "two-qubit error, and a symmetric readout error at its median readout error; no thermal relaxation",
    "fake_marrakesh": "Aer density-matrix simulation of the ISA circuits under NoiseModel.from_backend on the FakeMarrakesh calibration "
                      "snapshot: per-gate depolarizing and thermal-relaxation errors and per-qubit readout errors",
}

READOUT_CALIBRATION_NOTE = (
    "Readout calibration is per circuit. Routing places the measured (source) register on different physical qubits in "
    "different per-step circuits, so the circuits are grouped by their measured-qubit tuple; each distinct tuple gets one "
    "calibration set (the baseline circuit and one single-flip circuit per measured qubit, transpiled onto exactly those qubits and "
    "run on the arm's own simulator) and every circuit is mitigated with the set of its own tuple. "
    "The shard records the tuples and the index of the set each circuit used (circuit_to_calibration)."
)

RESOURCES_COLUMNS = {
    "dataset": "dataset name", "instance": "full lumped operator or a nested retained instance (label)",
    "classes": "lumping classes", "R": "supra-nodes of the (retained) lumped operator", "m": "register width ceil(log2 R)",
    "logical_qubits": "logical qubits 2 m", "borrowers": "borrowers the instance covers",
    "gbar_density": "share of non-zero entries of G_bar", "max_column_support": "largest column support of G_bar",
    "fraction_columns_full_reference_support": "share of columns with support on every reference copy",
    "state_prep_angles": "multiplexed state-preparation angles 2^m (2^m - 1)",
    "two_qubit_logical_analytic": "analytic two-qubit count of one reflection--swap",
    "two_qubit_logical_transpiled": "two-qubit count of one logical step after a basis transpilation (no routing), where m allows it",
    "n_masked_groups": "distinct masked classes the instance retains",
    "route": "discriminant the router selects for the retained operator against its member-level operator (D_bar or H); "
             "null when the instance was not simulated (not routed)",
    "lumpability_residual": "max |C G - G_bar C| of the retained operator against the member-level operator of the same classes",
    "compatibility_residual": "max |D B - B D_bar| of the same pair",
}

STATEVECTOR_COLUMNS = {
    "dataset": "dataset name", "R": "supra-nodes of the full lumped operator", "m": "register width", "logical_qubits": "logical qubits 2 m",
    "status": "ran, or not attempted", "reason": "why it was not attempted", "estimated_angle_bytes": "bytes of the preparation angles",
    "estimated_statevector_bytes": "bytes of the 2m-qubit statevector", "T": "steps simulated", "wall_seconds": "wall seconds of the run",
    "probe_seconds": "wall seconds of the two-step probe", "max_abs_error": "max |circuit source probability - NumPy lumped walk| over steps",
    "budget_seconds": "wall budget of the rung",
}

NOISE_GRID_COLUMNS = {
    "dataset": "dataset name", "k_classes": "classes of the retained instance", "instance": "instance label", "R": "supra-nodes of the instance",
    "m": "register width", "borrowers": "borrowers covered", "n_masked_groups": "distinct masked classes", "tau": "recorded steps averaged",
    "arm": "noiseless_shots, aer_noise_model or fake_marrakesh; the arms are defined in the instance record "
           "(hardware_candidates.json, 'arms') and the manifest",
    "mitigated": "whether per-circuit readout mitigation was applied to the arm's counts (noisy arms only)",
    "noise": "what the arm simulates; for aer_noise_model: depolarizing 1q/2q errors and a symmetric readout error at the "
             "FakeMarrakesh medians, no thermal relaxation; for fake_marrakesh: its calibration-derived model; for noiseless_shots: no noise",
    "shots": "shots per circuit", "depth_max": "maximal transpiled depth over the circuits t < tau",
    "two_qubit_sum": "two-qubit gates summed over the circuits t < tau",
    "predicted_survival_min": "conservative predicted survival (every one-qubit gate at the one-qubit median), min over the circuits",
    "estimated_qpu_seconds": "ESTIMATED QPU seconds of a hand-run submission of this cell: shots x sum over the walk circuits and "
                             "one calibration set per distinct measured-qubit tuple of (depth x two-qubit duration + readout + repetition delay)",
    "active_qubits": "physical qubits the routed circuits touch (max over t < tau)",
    "n_calibration_sets": "distinct measured-qubit tuples among the circuits t < tau, i.e. calibration sets used (per-circuit calibration)",
    "optimization_level": "preset pass-manager level", "transpiler_seed": "transpiler seed", "basis": "native two-qubit gates of the target",
    "route": "discriminant the router selects for the retained operator (D_bar or H)",
    "lumpability_residual": "max |C G - G_bar C| of the retained operator", "compatibility_residual": "max |D B - B D_bar| of the retained operator",
    "estimated_gates": "ISA gates summed over the circuits t < tau (the work of the cell)",
    "estimated_state_bytes": "bytes of the simulated state at the touched-qubit width (16 * 4^n density matrix, 16 * 2^n statevector)",
    "wall_seconds": "wall seconds the cell took to simulate (transpilation to the simulator basis included)",
    "padding_mass": "mean measured mass on padding codes (removed and the rest renormalised)",
    "spearman_supra": "Spearman against the exact time average over the R supra-nodes",
    "tv": "total variation to the exact time average", "tv_uniform": "total variation between uniform and the exact time average",
    "fidelity_ratio": "1 - TV / TV_uniform",
    "spearman_masked_classes": "Spearman over the class probabilities (layer copies summed) of the masked classes; null with fewer than two",
    "pure_noise_spearman_lo": "lower quantile (config.hardware_noise_band_quantiles[0] percent) of the Spearman of a pure-noise output against this "
                              "instance's own exact average at this tau, over config.hardware_noise_band_draws draws; a pure-noise output is "
                              "config.hardware_shots shots of uniform random codes per circuit, converted and averaged like a measured output",
    "pure_noise_spearman_hi": "upper quantile of the same distribution",
    "pure_noise_fidelity_lo": "lower quantile of the fidelity ratio of a pure-noise output against this instance's exact average",
    "pure_noise_fidelity_hi": "upper quantile of the same distribution; a raw cell carries signal when its fidelity_ratio is above this",
    "signal_above_band": "whether the raw cell carries signal: fidelity_ratio strictly above pure_noise_fidelity_hi, the upper edge of its "
                         "own pure-noise band (one-sided; a value on the edge or below the band is not signal); null for mitigated and "
                         "not simulated cells",
    "status": "'" + NOT_SIMULATED + "' for a cell that was not simulated; empty otherwise",
    "reason": "why the cell was not simulated, with the estimate",
    "estimated_simulation_seconds": "estimated wall seconds of the cell, decided before it ran: ISA gates x state bytes x the arm's measured seconds "
                                    "per gate and byte (that of its kind, or config.hardware_prior_bytes_per_second, before the arm has run); "
                                    "wall_seconds is what it took",
    "budget_seconds": "config.hardware_instance_budget_seconds",
    "spent_seconds": "simulation seconds already spent on the instance when the cell was declined",
    "estimated_two_qubit_logical": "analytic logical two-qubit gates of one reflection--swap, for an instance wider than the register cap",
}


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


SUPRA_DECIMALS = 15
BAND_ARM = "fake_marrakesh"


def supra_metrics(omega: np.ndarray, exact: np.ndarray) -> dict[str, float]:
    """Spearman, TV, TV to uniform and the fidelity ratio of ``omega`` against ``exact`` over the supra-nodes."""
    omega, exact = np.asarray(omega, dtype=float), np.asarray(exact, dtype=float)
    uniform = np.full(exact.size, 1.0 / exact.size)
    tv = total_variation(omega, exact, decimals=SUPRA_DECIMALS)
    tv_uniform = total_variation(uniform, exact, decimals=SUPRA_DECIMALS)
    return {"spearman_supra": spearman(omega, exact, decimals=SUPRA_DECIMALS), "tv": tv, "tv_uniform": tv_uniform,
            "fidelity_ratio": 1.0 - tv / tv_uniform if tv_uniform > 0 else float("nan")}


def masked_class_spearman(omega: np.ndarray, exact: np.ndarray, nodes: pd.DataFrame, decimals: int) -> float:
    """Spearman over the class probabilities of the masked classes; NaN with fewer than two of them."""
    totals_o: dict[str, float] = {}
    totals_e: dict[str, float] = {}
    masked: dict[str, bool] = {}
    for position, row in enumerate(nodes.itertuples(index=False)):
        if str(row.node_type) != "borrower_copy":
            continue
        key = str(row.member_borrower_ids)
        totals_o[key] = totals_o.get(key, 0.0) + float(omega[position])
        totals_e[key] = totals_e.get(key, 0.0) + float(exact[position])
        masked[key] = int(row.BAD_masked) == UNKNOWN_LABEL
    keys = [key for key in totals_o if masked[key]]
    if len(keys) < 2:
        return float("nan")
    return spearman(np.array([totals_o[k] for k in keys]), np.array([totals_e[k] for k in keys]), decimals=decimals)


def grade(omega: np.ndarray, exact: np.ndarray, nodes: pd.DataFrame, decimals: int) -> dict[str, float]:
    """The supra-node metrics plus the masked-class Spearman (``spearman_masked_classes``)."""
    return {**supra_metrics(omega, exact), "spearman_masked_classes": masked_class_spearman(omega, exact, nodes, decimals)}


def pure_noise_outputs(R: int, m: int, shots: int, tau_max: int, draws: int, rng: np.random.Generator) -> np.ndarray:
    """``draws`` pure-noise outputs, shape ``(draws, tau_max, R)``: entry ``[d, tau - 1]`` is the output averaged over ``tau`` circuits.

    One circuit of a device that returns uniform random codes is ``shots`` samples of the uniform distribution over the
    ``2^m`` register codes; it is converted as a measured output is (codes ``>= R`` are padding: dropped, the retained
    support renormalised) and the ``tau`` circuits, independent draws, are averaged.
    """
    dim = 2 ** int(m)
    if R > dim:
        raise ValueError(f"{R} supra-nodes do not fit a register of {m} qubits")
    counts = rng.multinomial(int(shots), np.full(dim, 1.0 / dim), size=(int(draws), int(tau_max))).astype(float)
    retained = counts[..., :R]
    total = retained.sum(axis=-1, keepdims=True)
    steps = np.where(total > 0.0, retained / np.maximum(total, 1.0), 1.0 / R)            # one converted output per circuit
    return np.cumsum(steps, axis=1) / np.arange(1, int(tau_max) + 1, dtype=float)[None, :, None]


def pure_noise_band(
    exact_by_tau: Mapping[int, np.ndarray], *, m: int, shots: int, draws: int, quantiles: Sequence[float], seed: int
) -> dict[int, dict[str, float]]:
    """Per tau, the quantiles over ``draws`` pure-noise outputs of the Spearman and the fidelity ratio against ``exact_by_tau[tau]``.

    ``exact_by_tau`` maps each recorded tau to the exact average over ``t < tau`` of the instance; the
    instance has ``R`` supra-nodes (the vector length) in a register of ``m`` qubits. Returns, per tau,
    ``spearman_lo/median/hi`` and ``fidelity_lo/median/hi`` (``lo``/``hi`` the lower and upper quantile).
    """
    taus = sorted(int(t) for t in exact_by_tau)
    R = int(np.asarray(exact_by_tau[taus[0]]).size)
    averages = pure_noise_outputs(R, m, shots, taus[-1], draws, np.random.default_rng(int(seed)))
    lo_q, hi_q = (float(q) for q in quantiles)
    band: dict[int, dict[str, float]] = {}
    for tau in taus:
        exact = np.asarray(exact_by_tau[tau], dtype=float)
        graded = [supra_metrics(averages[d, tau - 1], exact) for d in range(int(draws))]
        rho = np.array([g["spearman_supra"] for g in graded], dtype=float)
        fid = np.array([g["fidelity_ratio"] for g in graded], dtype=float)

        def edges(values: np.ndarray) -> tuple[float, float, float]:
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                return float("nan"), float("nan"), float("nan")
            lo, med, hi = np.percentile(finite, [lo_q, 50.0, hi_q])
            return float(lo), float(med), float(hi)

        s_lo, s_med, s_hi = edges(rho)
        f_lo, f_med, f_hi = edges(fid)
        band[tau] = {"spearman_lo": s_lo, "spearman_median": s_med, "spearman_hi": s_hi,
                     "fidelity_lo": f_lo, "fidelity_median": f_med, "fidelity_hi": f_hi,
                     "n_spearman_defined": int(np.isfinite(rho).sum()), "draws": int(draws)}
    return band


def inside_band(value: Any, lo: Any, hi: Any) -> bool:
    """Whether ``value`` lies in ``[lo, hi]``; an undefined value or band is not inside."""
    try:
        value, lo, hi = float(value), float(lo), float(hi)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(value) and np.isfinite(lo) and np.isfinite(hi) and lo <= value <= hi)


def is_simulated(row: Mapping[str, Any]) -> bool:
    """A row of a simulated cell: its status is not a ``not simulated`` one."""
    return not str(row.get("status") or "").startswith("not simulated")


NO_ENTRY = "none"
NOT_EVALUATED = "not evaluated"


def _finite(value: Any) -> float | None:
    """``value`` as a finite float, or ``None`` when it is missing, not a number or not finite."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def signal_above_band(value: Any, hi: Any) -> bool:
    """Whether ``value`` is strictly above ``hi``, the upper edge of its own pure-noise band.

    One-sided and strict: a value on the edge is not signal and neither is a value below the band; an
    undefined value or edge is not signal.
    """
    value, hi = _finite(value), _finite(hi)
    return value is not None and hi is not None and value > hi


def carries_signal(row: Mapping[str, Any]) -> bool:
    """Whether a raw cell carries signal: its ``fidelity_ratio`` is strictly above its own ``pure_noise_fidelity_hi``."""
    return signal_above_band(row.get("fidelity_ratio"), row.get("pure_noise_fidelity_hi"))


def is_raw_cell(row: Mapping[str, Any]) -> bool:
    """A simulated cell of the raw output (not the readout-mitigated one); ``mitigated`` may be a bool or the text of a CSV."""
    return str(row.get("mitigated")).lower() != "true" and is_simulated(row)


def _judgeable(row: Mapping[str, Any]) -> bool:
    """A cell whose fidelity ratio and band edge are both defined."""
    return _finite(row.get("fidelity_ratio")) is not None and _finite(row.get("pure_noise_fidelity_hi")) is not None


def judged_cells(rows: Sequence[Mapping[str, Any]], *, arm: str, min_tau: int) -> dict[int, Mapping[str, Any]]:
    """Per instance ``k``, the raw simulated cell of ``arm`` at its largest recorded ``tau >= min_tau``: the longest walk.

    ``tau = 1`` is the state preparation alone and is never judged. An instance with no such cell, or whose
    longest-walk cell has an undefined fidelity ratio or band, is not evaluated and is absent from the result.
    """
    longest: dict[int, Mapping[str, Any]] = {}
    for r in rows:
        if r.get("arm") != arm or not is_raw_cell(r) or r.get("tau") is None or int(r["tau"]) < int(min_tau):
            continue
        k = int(r["k_classes"])
        if k not in longest or int(r["tau"]) > int(longest[k]["tau"]):
            longest[k] = r
    return {k: r for k, r in sorted(longest.items()) if _judgeable(r)}


def first_tau_without_signal(cells: Sequence[Mapping[str, Any]], *, min_tau: int) -> int | str:
    """For one instance and arm: the first recorded ``tau >= min_tau`` at which the raw cell carries no signal.

    ``"> <tau>"``, naming the largest simulated ``tau`` among the cells, when every evaluated cell carries
    signal; ``"not evaluated"`` when no cell qualifies (none simulated at ``tau >= min_tau``, or the fidelity
    ratio or band undefined). ``cells`` are the rows of one instance and one arm.
    """
    evaluated = [c for c in cells if is_raw_cell(c) and c.get("tau") is not None and int(c["tau"]) >= int(min_tau) and _judgeable(c)]
    if not evaluated:
        return NOT_EVALUATED
    quiet = [int(c["tau"]) for c in evaluated if not carries_signal(c)]
    return min(quiet) if quiet else f"> {max(int(c['tau']) for c in evaluated)}"


def noise_entry_points(rows: Sequence[Mapping[str, Any]], *, arm: str = BAND_ARM, min_tau: int, selected_k: int | None = None) -> dict[str, Any]:
    """Where the raw output of ``arm`` stops carrying signal, over instances and over steps.

    ``noise_k_band``: the smallest evaluated ``k`` such that neither it nor any larger evaluated ``k`` carries
    signal (each judged at its longest walk, :func:`judged_cells`); ``"none"`` when no evaluated ``k`` satisfies
    that, in particular when every evaluated ``k`` carries signal. ``noise_tau_band``: for the selected instance
    ``selected_k``, the first ``tau >= min_tau`` without signal, or ``"> <tau>"`` (:func:`first_tau_without_signal`);
    ``"not evaluated"`` without a selected instance. The record also lists the evaluated instances, those that
    carry signal and the ``tau`` each was judged at.
    """
    judged = judged_cells(rows, arm=arm, min_tau=min_tau)
    signal = {k: carries_signal(r) for k, r in judged.items()}
    band: int | str = NO_ENTRY
    for k in sorted(judged, reverse=True):
        if signal[k]:
            break
        band = k
    if selected_k is None:
        tau_band: int | str = NOT_EVALUATED
    else:
        tau_band = first_tau_without_signal([r for r in rows if r.get("arm") == arm and int(r["k_classes"]) == int(selected_k)], min_tau=min_tau)
    return {"arm": arm, "noise_k_band": band, "noise_tau_band": tau_band, "selected_k": selected_k,
            "evaluated_k": sorted(judged), "k_with_signal": sorted(k for k, s in signal.items() if s),
            "judged_tau": {str(k): int(r["tau"]) for k, r in judged.items()}}


def first_rankable_instance(rows: Sequence[Mapping[str, Any]], *, arm: str = BAND_ARM, min_tau: int) -> dict[str, Any]:
    """The first instance able to rank test individuals: the smallest ``k`` that retains two or more masked classes.

    Returns ``hw_rankable_k``, ``hw_rankable_m`` and ``hw_rankable_borrowers`` of that instance, ``hw_rankable_fr``,
    the largest raw fidelity ratio of ``arm`` over its cells at ``tau >= min_tau`` (the ties go to the longer walk),
    and ``hw_rankable_signal``, whether that cell carries signal. Every value is ``None`` when no instance has two
    masked classes; the last two are ``None`` when the instance has no such simulated cell.
    """
    ks = sorted({int(r["k_classes"]) for r in rows if (_finite(r.get("n_masked_groups")) or 0) >= 2})
    out: dict[str, Any] = {"hw_rankable_k": None, "hw_rankable_m": None, "hw_rankable_borrowers": None, "hw_rankable_fr": None,
                           "hw_rankable_signal": None}
    if not ks:
        return out
    mine = [r for r in rows if int(r["k_classes"]) == ks[0]]
    out.update(hw_rankable_k=ks[0], hw_rankable_m=mine[0].get("m"), hw_rankable_borrowers=mine[0].get("borrowers"))
    cells = [r for r in mine if r.get("arm") == arm and is_raw_cell(r) and r.get("tau") is not None and int(r["tau"]) >= int(min_tau)
             and _finite(r.get("fidelity_ratio")) is not None]
    if cells:
        best = max(cells, key=lambda r: (float(r["fidelity_ratio"]), int(r["tau"])))
        out.update(hw_rankable_fr=float(best["fidelity_ratio"]), hw_rankable_signal=carries_signal(best))
    return out


def annotate_signal(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The rows of the assembled noise grid: ``signal_above_band`` on every raw simulated cell, ``None`` on the others.

    The shard rows carry the Spearman-based ``inside_pure_noise_band`` of the run that wrote them; it is dropped here.
    The new column is computed from the fidelity ratio and the band edge of the row itself, so shards written
    before this definition need no re-simulation.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        annotated = {key: value for key, value in row.items() if key != "inside_pure_noise_band"}
        annotated["signal_above_band"] = carries_signal(row) if is_raw_cell(row) else None
        out.append(annotated)
    return out
