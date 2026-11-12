"""Noise-grid shards: one JSON per retained instance, with an identity that is checked, never assumed.

An array task writes ``grid_<k>.json`` under the stage directory; the stage
assembles whatever shards exist. A shard records the identity of the run that
wrote it -- ``{config_fingerprint, smoke, shard_schema_version}`` -- where the
fingerprint covers the configuration (:func:`fingerprint.config_fingerprint`),
the hardware parameters that change a cell, the shard schema version and a
digest of the retained operator. A shard whose identity differs from the
current run raises :class:`AssertionError` with a "delete the directory"
message: it is never reused silently and never recomputed silently. A shard
from before this schema carries no identity and is refused the same way.

Every cell of the grid has a row. A shard written before its task finished
(``complete`` false) or no shard at all leaves cells that no row covers; the
assembly records each such cell as ``not simulated: resource budget`` with the
reason, so a task that ran out of allocation never leaves a hole in the table.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from fingerprint import config_fingerprint
from quantum.hardware.noise_grading import NOT_SIMULATED

SHARD_SCHEMA_VERSION = 2
SHARD_DIR = "grid_shards"
MISSING_CELL_REASON = ("no complete shard: the array task of this instance ended before the cell was simulated "
                       "(allocation exhausted, or the task failed; see its log)")


def shard_path(output: Path, k: int) -> Path:
    return Path(output) / SHARD_DIR / f"grid_{int(k)}.json"


def operator_digest(G: np.ndarray, v: np.ndarray) -> str:
    """SHA-256 of the retained operator's bytes: a shard belongs to exactly one operator."""
    digest = hashlib.sha256()
    for array in (np.asarray(G, dtype=np.float64), np.asarray(v, dtype=np.float64)):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def hardware_parameters(config) -> dict[str, Any]:
    """The configuration values that change what a cell contains, beyond the stage fingerprint's own."""
    return {
        "steps": [int(t) for t in config.hardware_steps], "shots": int(config.hardware_shots),
        "arms": list(config.hardware_noise_arms), "optimization_level": int(config.hardware_optimization_level),
        "max_register_width": int(config.hardware_max_register_width),
        "density_matrix_max_bytes": int(config.hardware_density_matrix_max_bytes),
        "instance_budget_seconds": float(config.hardware_instance_budget_seconds),
        "prior_bytes_per_second": float(config.hardware_prior_bytes_per_second),
        "noise_band_draws": int(config.hardware_noise_band_draws),
        "noise_band_quantiles": [float(q) for q in config.hardware_noise_band_quantiles],
        "backend": str(config.ibm_backend), "router_tol": float(config.router_tol), "tie_tolerance": int(config.tie_tolerance),
    }


def shard_identity(config, dataset_name: str, stage: str, instance: dict[str, Any], smoke: bool) -> dict[str, Any]:
    """The identity a shard of ``instance`` must carry under this configuration."""
    extra = {"shard_schema_version": SHARD_SCHEMA_VERSION, "dataset": dataset_name, "k": int(instance["k"]),
             "operator_sha256": operator_digest(instance["G"], instance["v"]), "hardware": hardware_parameters(config)}
    return {"config_fingerprint": config_fingerprint(config, stage, network_spec=f"{dataset_name}/{instance['label']}", extra=extra),
            "smoke": bool(smoke), "shard_schema_version": SHARD_SCHEMA_VERSION}


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable in a shard")


def write_shard(path: Path, identity: dict[str, Any], record: dict[str, Any], *, complete: bool) -> Path:
    """Write ``record`` with its identity; ``complete`` is false for a shard written part-way through the task."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"identity": identity, "complete": bool(complete), **record}
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(payload, default=_json_default))
    temporary.replace(path)
    return path


def read_shard(path: Path, identity: dict[str, Any]) -> dict[str, Any]:
    """The shard at ``path``, refused unless it was written under exactly ``identity``."""
    path = Path(path)
    record = json.loads(path.read_text())
    found = record.get("identity")
    if found != identity:
        raise AssertionError(
            f"{path} was written under another configuration or shard schema "
            f"(found {json.dumps(found, sort_keys=True)}; this run is {json.dumps(identity, sort_keys=True)}). "
            f"Delete the directory {path.parent} and re-run the array tasks; a stale shard is never reused "
            "and never recomputed silently."
        )
    return record


def expected_cells(config) -> list[tuple[str, int]]:
    """Every ``(arm, tau)`` simulation cell of an instance, in the order the grid visits them."""
    return [(arm, int(tau)) for tau in config.hardware_steps for arm in config.hardware_noise_arms]


def cells_of(record: dict[str, Any] | None, instance: dict[str, Any], dataset: str, config, *,
             reason: str = MISSING_CELL_REASON) -> tuple[list[dict[str, Any]], int]:
    """Every row of one instance, with each cell no row covers recorded as not simulated.

    Returns ``(rows, n_filled)``. ``record`` is a shard (complete or not) or ``None``.
    """
    rows = list(record["rows"]) if record else []
    present = {(r.get("arm"), int(r["tau"])) for r in rows if r.get("tau") is not None}
    filled = 0
    for arm, tau in expected_cells(config):
        if (arm, tau) in present:
            continue
        rows.append({"dataset": dataset, "k_classes": instance["k"], "instance": instance["label"], "R": instance["R"], "m": instance["m"],
                     "borrowers": instance["borrowers"], "n_masked_groups": instance["n_masked_groups"], "tau": tau, "arm": arm,
                     "mitigated": False, "status": NOT_SIMULATED, "reason": reason})
        filled += 1
    return rows, filled

