"""The full-operator walks of stage 04: the reduction check against the production operator.

The reported QPPR score is the exact limit on the lumped operator
(``quantum.mechanism.scores_stage``). What remains of the full-operator walk
is a check: the matrix-free Szegedy walk on the production ``MultiLayerRanker``
operator -- built on the *same* masked frame stage 03 wrote and asserted,
digest by digest, to be the classical arm's operator -- is run for
``reduction_check_steps`` recorded steps, and the scoring stage compares its
per-step borrower marginals with the lumped walk's. The checkpointed
evolution and the warm cache (both ``quantum.inspired.walk_cache``) are kept, keyed on the configuration *and*
the operator digests, so a cache can never be reused for another operator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from config import ExperimentConfig, NetworkSpec
from fingerprint import (
    TEST_OPERATOR_TAG,
    assert_operator_identity,
    build_ranker,
    operator_cache_extra,
    operator_digests,
    sha256_path,
    smoke_enabled,
)
from paths import masked_frame_path, quantum_history_path, stage_dir
from quantum.inspired.mulp import QuantumWalkRanker
from quantum.inspired.walk_cache import (
    WalkRun,
    assert_checkpoint_fingerprint,
    checkpoint_dir,
    run_checkpointed,
    walk_fingerprint,
)

CHECKPOINT_EVERY = 5


# ---------------------------------------------------------------------------
# Operators and the identity check
# ---------------------------------------------------------------------------


def build_walker(masked_path: Path, spec: NetworkSpec, config: ExperimentConfig) -> QuantumWalkRanker:
    """A quantum ranker on a masked frame, configured from the frozen config; implicit mode."""
    ranker = build_ranker(masked_path, spec, config)
    walker = QuantumWalkRanker(ranker=ranker, alpha=config.resolved_alpha())
    walker.buildAdjMatrix()
    walker.buildPersonalizationMatrix()
    walker.buildTransitionMatrix()
    walker.buildPersonalizationVector()
    return walker


def digests_for(walker: QuantumWalkRanker, masked_path: Path) -> dict[str, str]:
    """The same digests stage 03 recorded, taken from the quantum arm's ranker."""
    ranker = walker.ranker
    if ranker.restart_matrix is None:
        ranker.buildGoogleOperator(walker.alpha)
    return {**operator_digests(ranker), "masked_frame": sha256_path(masked_path)}


def load_classical_digests(*, stage: str) -> dict[str, dict[str, str]]:
    path = stage_dir(stage) / "operator_digests.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; the classical stage must run first")
    return json.loads(path.read_text())


def load_quantum_digests(output: Path) -> dict[str, dict[str, str]]:
    path = output / "operator_digests.json"
    return json.loads(path.read_text()) if path.exists() else {}


def write_quantum_digests(output: Path, quantum: dict[str, dict[str, str]]) -> Path:
    path = output / "operator_digests.json"
    path.write_text(json.dumps(quantum, indent=2, sort_keys=True))
    return path


def verified_walker(spec: NetworkSpec, tag: str, config: ExperimentConfig, classical: dict, quantum: dict) -> QuantumWalkRanker:
    """Build the walker on stage 03's masked frame and prove it is the same operator."""
    key = f"{spec.name}/{tag}"
    path = masked_frame_path(spec.name, tag)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; the classical stage writes the masked frames both arms read")
    if key not in classical:
        raise KeyError(f"the classical stage recorded no operator digest for {key}")
    walker = build_walker(path, spec, config)
    digests = digests_for(walker, path)
    assert_operator_identity(classical[key], digests, f"operator for {key}")
    quantum[key] = digests
    walker.buildSzegedyWalk()
    return walker


# ---------------------------------------------------------------------------
# The walks this stage runs
# ---------------------------------------------------------------------------


def walk(walker: QuantumWalkRanker, tau: int, directory: Path, *, fingerprint: str) -> WalkRun:
    """Run (or resume, or reuse) one checkpointed walk after checking the cache belongs to this run."""
    assert_checkpoint_fingerprint(directory, fingerprint, smoke=smoke_enabled())
    return run_checkpointed(walker.szegedy_walk, int(tau), directory, every=CHECKPOINT_EVERY)


def save_history(spec_name: str, tag: str, run: WalkRun, *, stage: str) -> Path:
    path = quantum_history_path(spec_name, tag, stage=stage)
    np.save(path, np.asarray(run.history, dtype=float))
    return path


def run_walk(spec_name: str, tag: str, tau: int, config: ExperimentConfig, classical: dict, quantum: dict,
             output: Path, *, specs, stage: str) -> WalkRun:
    """Verify the operator against stage 03, walk it to ``tau``, save the borrower history.

    The one call behind every full-operator walk and the one call an array
    task makes. The cache fingerprint carries the operator digests and the
    masked frame's checksum, so a completed checkpoint is reused only for the
    operator that produced it.
    """
    spec = specs[spec_name]
    walker = verified_walker(spec, tag, config, classical, quantum)
    digests = quantum[f"{spec.name}/{tag}"]
    fingerprint = walk_fingerprint(config, spec_name, tag, tau,
                                   extra=operator_cache_extra(digests, masked_frame_path(spec.name, tag)))
    run = walk(walker, tau, checkpoint_dir(output, spec_name, tag), fingerprint=fingerprint)
    save_history(spec_name, tag, run, stage=stage)
    return run


def reduction_check_walk(spec_name: str, config: ExperimentConfig, classical: dict, quantum: dict, output: Path,
                         *, specs, stage: str) -> dict[str, Any]:
    """Walk the test operator for ``reduction_check_steps`` recorded steps; save and describe the history."""
    run = run_walk(spec_name, TEST_OPERATOR_TAG, int(config.reduction_check_steps), config, classical, quantum, output,
                   specs=specs, stage=stage)
    history = np.asarray(run.history, dtype=float)
    mean_step = float(np.mean(run.step_seconds)) if run.step_seconds else float("nan")
    timing = {"mean_step_seconds": mean_step, "n_steps_timed": len(run.step_seconds), "n_steps": int(history.shape[0])}
    (output / f"walk_timing_{spec_name}_{TEST_OPERATOR_TAG}.json").write_text(json.dumps(timing))
    return {"history_shape": list(history.shape), "row_mass": [float(history[0].sum()), float(history[-1].sum())],
            **timing}
