"""The resumable Szegedy evolution of stage 04 and its warm cache: checkpoints, planned walks, fingerprints.

The resumable evolution
-----------------------
Checkpoints, the completed record, and the timing probe. The state matrix is
``16 N^2`` bytes -- about 2.2 GB at ``N = 11,722`` -- so a job that dies at step
80 of 100 has thrown away most of a day. Every ``every`` recorded steps the
state and the per-step marginals are written and the run resumes from there.
This is the evolution every walk of notebook 04 and every array task runs
(``quantum.inspired.stage.walk``).

The warm cache
--------------
Every planned walk, its checkpoint directory, its fingerprint. One list
enumerates the walks (:func:`planned_walks`); the array worker
(``scripts/warm.py``) reads it. A checkpoint written under another
configuration is refused, never silently recomputed.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from config import ExperimentConfig
from datasets import DatasetConfig
from fingerprint import TEST_OPERATOR_TAG, config_fingerprint


# ---------------------------------------------------------------------------
# The resumable evolution
# ---------------------------------------------------------------------------


_logger = logging.getLogger("quantum.inspired.szegedy")


@dataclass
class WalkRun:
    """What one checkpointed evolution produced.

    ``supra_history`` is present only when the caller asked for it. It is the
    exact per-step source-register marginal over all ``N`` supra-nodes, which
    the finite-shot estimator samples from; the borrower ``history`` is that
    same quantity summed over each borrower's layer copies.
    """

    omega: np.ndarray
    history: np.ndarray
    supra_history: np.ndarray | None
    step_seconds: list[float]


def time_steps(walk, n_steps: int) -> list[float]:
    """Wall seconds of ``n_steps`` recorded steps (marginal + one W^2), nothing kept.

    A calibration probe for the resource plan: it runs the same
    ``_marginal_supra_node_probs`` + ``_advance`` pair the checkpointed walk
    runs, on a fresh initial state, and returns only the per-step timings.
    """
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive; got {n_steps}")
    Psi = walk._build_initial_state()
    seconds: list[float] = []
    for _ in range(int(n_steps)):
        began = perf_counter()
        walk._marginal_supra_node_probs(Psi, register="source")
        Psi = walk._advance(Psi)
        seconds.append(perf_counter() - began)
    return seconds


def run_checkpointed(
    walk,
    tau: int,
    checkpoint_dir,
    *,
    every: int = 5,
    record_supra_history: bool = False,
    validate_probabilities: bool = False,
) -> WalkRun:
    """Exact evolution that survives being killed.

    Only the latest checkpoint is kept, and it is written to a temporary name
    and renamed, so an interrupted write cannot leave a half-file that the
    next run would load as valid state. On completion the state file is
    replaced by ``completed.npz``, which holds only the marginals and the
    per-step timings: keeping the state would cost gigabytes per finished walk
    for nothing, while the marginals make a re-run of the whole notebook free.

    ``record_supra_history`` additionally keeps the full ``(tau, N)``
    source-register marginals, which the finite-shot sweep samples from.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive; got {tau}")
    every = max(1, int(every))
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    latest = directory / "latest.npz"
    completed = directory / "completed.npz"

    def restored(saved) -> np.ndarray | None:
        """The saved supra history, if this run wants one and it is there."""
        if not record_supra_history:
            return None
        return saved["supra_history"] if "supra_history" in saved.files else None

    if completed.exists():
        with np.load(completed) as saved:
            supra = restored(saved)
            reusable = int(saved["tau"]) == tau and int(saved["n"]) == walk.n
            if reusable and (supra is not None or not record_supra_history):
                history = saved["history"]
                # The per-step timings travel with the artifact, so a run that
                # finds a warm cache still reports how long the walk took.
                timings = (
                    [float(x) for x in saved["step_seconds"]]
                    if "step_seconds" in saved.files
                    else []
                )
                _logger.info("reusing completed walk at tau=%d from %s", tau, directory)
                return WalkRun(history.mean(axis=0), history, supra, timings)

    start = 0
    step_seconds: list[float] = []
    history = np.zeros((tau, walk.n_borrowers), dtype=float)
    supra_history = np.zeros((tau, walk.n), dtype=float) if record_supra_history else None
    Psi = None
    if latest.exists():
        with np.load(latest) as saved:
            saved_tau = int(saved["tau"])
            if saved_tau != tau or int(saved["n"]) != walk.n:
                raise ValueError(
                    f"checkpoint in {directory} was written for tau={saved_tau}, "
                    f"n={int(saved['n'])}; delete it or run the same configuration"
                )
            saved_supra = saved["supra_history"] if "supra_history" in saved.files else None
            if record_supra_history and saved_supra is None:
                raise ValueError(
                    f"the checkpoint in {directory} was written without the "
                    "supra-node marginals this run needs; delete it and re-run"
                )
            start = int(saved["step"])
            history[:start] = saved["history"][:start]
            if supra_history is not None:
                supra_history[:start] = saved_supra[:start]
            Psi = saved["Psi"]
            if "step_seconds" in saved.files:
                step_seconds = [float(x) for x in saved["step_seconds"]]
        _logger.info("resuming walk from checkpoint at step %d/%d", start, tau)
    if Psi is None:
        Psi = walk._build_initial_state()

    def snapshot(path, **arrays) -> None:
        # Written through a file object: np.savez appends ".npz" to a path
        # that lacks it, which would defeat the atomic rename.
        temporary = path.with_suffix(path.suffix + ".partial")
        with open(temporary, "wb") as handle:
            np.savez(handle, tau=tau, n=walk.n, **arrays)
        os.replace(temporary, path)

    for t in range(start, tau):
        began = perf_counter()
        if validate_probabilities:
            walk._assert_probability_normalized(Psi, context=f"step {t}")
        # One |Psi|^2 pass serves both: the borrower marginal is the
        # supra-node marginal summed over each borrower's layer copies.
        probabilities = walk._marginal_supra_node_probs(Psi, register="source")
        if supra_history is not None:
            supra_history[t] = probabilities
        borrower = probabilities[walk.borrower_layer_indices[0]].copy()
        for layer_indices in walk.borrower_layer_indices[1:]:
            borrower += probabilities[layer_indices]
        history[t] = borrower
        Psi = walk._advance(Psi)
        step_seconds.append(perf_counter() - began)

        if (t + 1) % every == 0 or t + 1 == tau:
            extra = {} if supra_history is None else {"supra_history": supra_history}
            snapshot(
                latest, Psi=Psi, history=history, step=t + 1,
                step_seconds=np.asarray(step_seconds, dtype=float), **extra,
            )
            _logger.info("checkpoint at step %d/%d (%.2fs/step)", t + 1, tau, step_seconds[-1])

    extra = {} if supra_history is None else {"supra_history": supra_history}
    snapshot(completed, history=history, step_seconds=np.asarray(step_seconds, dtype=float), **extra)
    latest.unlink(missing_ok=True)

    return WalkRun(history.mean(axis=0), history, supra_history, step_seconds)


# ---------------------------------------------------------------------------
# The warm cache
# ---------------------------------------------------------------------------


FINGERPRINT_FILE = "fingerprint.json"


def planned_walks(
    config: ExperimentConfig, *, ladder: tuple[str, ...], primary: str
) -> list[dict[str, Any]]:
    """Every full-operator walk stage 04 runs, in notebook order: one record per (network, tag).

    One test-operator walk per ladder network, to ``reduction_check_steps``
    recorded steps: the full-operator step marginals that the lumped walk is
    checked against. The reported score is the exact limit on the lumped
    operator and needs no full walk; the fold and placebo operators are scored
    by their lumped limits through the router, so no fold or placebo walk is
    planned here. ``primary`` is accepted for call-site symmetry and unused.
    """
    del primary
    return [
        {"kind": "test_operator", "network": name, "tag": TEST_OPERATOR_TAG,
         "tau": int(config.reduction_check_steps)}
        for name in ladder
    ]


def dataset_networks(dataset: DatasetConfig, *, resolve_specs: bool = False) -> dict[str, Any]:
    """One dataset's network set and stage directories, for :func:`planned_walks`.

    ``specs`` is the registry a network name resolves through. Where a
    dataset's specifications are discovered from its processed frame they are
    resolved only when ``resolve_specs`` is set, and are ``None`` for a plan
    that runs nothing.
    """
    specs = None
    if resolve_specs:
        from fingerprint import load_frozen_partitions

        specs = dataset.network_specs(load_frozen_partitions(dataset).frame)
    return {"ladder": dataset.ladder, "primary": dataset.primary, "specs": specs,
            "classical": dataset.stage("classical"), "quantum": dataset.stage("quantum")}


def checkpoint_dir(output: Path, spec_name: str, tag: str) -> Path:
    return Path(output) / "checkpoints" / f"{spec_name}_{tag}"


def walk_fingerprint(config: ExperimentConfig, spec_name: str, tag: str, tau: int, *, extra: dict[str, Any] | None = None) -> str:
    """Identity of one cached walk: the stage fingerprint, the walk's tag and horizon, and the operator.

    ``extra`` is ``fingerprint.operator_cache_extra`` (the operator digests and
    the masked frame's checksum); the stage always passes it, so a checkpoint
    belongs to exactly one operator. The plan listing omits it because no
    operator exists before the walk is built.
    """
    parts = {"tag": tag, "walk_tau": int(tau), **(extra or {})}
    return config_fingerprint(config, "04_quantum", network_spec=spec_name, extra=parts)


def assert_checkpoint_fingerprint(directory: Path, fingerprint: str, *, smoke: bool) -> None:
    """A checkpoint written under another configuration is an error, never a silent recompute.

    Called before every checkpointed run. If the directory holds walk data (a
    completed or partial checkpoint) its recorded fingerprint must equal the
    current one; a checkpoint with data but no record is refused too, since
    nothing says which configuration produced it. On a fresh directory the
    record is written so a later run can be checked against it.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    record_path = directory / FINGERPRINT_FILE
    has_data = (directory / "completed.npz").exists() or (directory / "latest.npz").exists()
    if record_path.exists():
        recorded = json.loads(record_path.read_text())
        if recorded.get("config_fingerprint") != fingerprint or bool(recorded.get("smoke")) != bool(smoke):
            raise AssertionError(
                f"{directory} was written under fingerprint {recorded.get('config_fingerprint')!r} "
                f"(smoke={recorded.get('smoke')}) but this run is {fingerprint!r} (smoke={smoke}). "
                "The warm cache does not belong to this configuration; delete the directory and "
                "re-warm it rather than letting the notebook recompute silently."
            )
        return
    if has_data:
        raise AssertionError(
            f"{directory} holds walk data but no {FINGERPRINT_FILE}; nothing says which configuration "
            "produced it. Delete the directory and re-run."
        )
    record_path.write_text(json.dumps({"config_fingerprint": fingerprint, "smoke": bool(smoke)}, indent=2))
