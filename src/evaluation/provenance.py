"""Provenance: the stage manifests, and the one result writer whose tables they list.

Manifests
---------
An arm that writes artifacts without a manifest is incomplete. The INFORMS
Journal on Computing software policy requires that a reported number be
traceable to the code, inputs, and environment that produced it. A manifest
records all three next to the artifacts themselves, so a result does not
depend on anyone remembering which commit or which split produced it.

Recorded per arm: the git commit, branch and whether the tree was dirty, and
``source_sha256``, a digest of the content of every file git lists for the
tree (tracked or not, ignored files left out), which identifies the exact code
that produced the results whether or not it is committed; the versions of every
numerical and quantum dependency; the SHA-256 of every input and output file;
the frozen configuration and ``config_hash``, the digest of it together with the
dataset configuration; every RNG seed; the class of machine (CPU model, cores,
RAM, OS); the wall clock and the peak memory of the run.

Never recorded: a host name, a user name, a scheduler job id or an absolute
path. A dirty working tree is recorded, never refused: a run from an
uncommitted tree is legitimate while the code is still being finished, and
``source_sha256`` says exactly which code it was.

The result writer
-----------------
``outputs/results/<scope>/<name>.csv`` at full precision. Every result table of
the paper is written by :func:`write_result` exactly once, by the stage that
computes it, under the scope it belongs to (a dataset name, ``synthetic`` or
``runtime``). The CSV is the only data artifact of a table: no ``.tex`` and no
per-table metadata file are written here. The ``.tex`` of a table the
manuscript prints is produced once, from the CSVs, by
``scripts/make_manuscript_tables.py`` through ``evaluation.table_specs``.

Column definitions and provenance go into the **stage manifest**: each call
registers ``{name: {source_stage, scope, path, columns}}`` in a process-local
registry that :func:`write_manifest` copies into the ``tables`` block of the
manifest of the stage named by ``stage``.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from fingerprint import sha256_path
from paths import RESULT_SCOPES, relative_to_repo, repo_root, result_path, results_dir


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------


MANIFEST_NAME = "manifest.json"

# Packages whose version changes can move a reported number, and the ones that run the stages.
TRACKED_PACKAGES = (
    "numpy",
    "scipy",
    "polars",
    "pandas",
    "scikit-learn",
    "xgboost",
    "qiskit",
    "qiskit-aer",
    "qiskit-ibm-runtime",
    "pyarrow",
    "matplotlib",
    "nbclient",
    "nbformat",
    "threadpoolctl",
)

# Import names where they differ from the distribution name.
_IMPORT_NAME = {
    "scikit-learn": "sklearn",
    "qiskit-aer": "qiskit_aer",
    "qiskit-ibm-runtime": "qiskit_ibm_runtime",
}


def _run_git(*args: str, timeout: int = 30) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def git_state() -> dict[str, Any]:
    """Commit, branch, and whether the working tree had uncommitted changes.

    A dirty tree is recorded rather than refused: a run mid-development is
    legitimate, but a result from one must not be presented as reproducible
    from the commit alone (``source_sha256`` identifies the code it ran).
    """
    status = _run_git("status", "--porcelain")
    status = status.strip("\n") if status is not None else None
    commit = _run_git("rev-parse", "HEAD")
    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    return {
        "commit": commit.strip() if commit else None,
        "branch": branch.strip() if branch else None,
        "dirty": bool(status) if status is not None else None,
        "dirty_files": status.splitlines()[:50] if status else [],
    }


@lru_cache(maxsize=1)
def source_sha256() -> str | None:
    """SHA-256 over the sorted ``(relative path, file SHA-256)`` pairs of every file ``git ls-files -co --exclude-standard`` lists.

    The files of the tree as it is: tracked or not, uncommitted edits included,
    ignored files (data, outputs, local job scripts) left out, a listed file that
    no longer exists skipped. ``None`` when git cannot list the tree.
    """
    listing = _run_git("ls-files", "-z", "-co", "--exclude-standard", timeout=300)
    if listing is None:
        return None
    root = repo_root()
    digest = hashlib.sha256()
    for name in sorted(set(n for n in listing.split("\0") if n)):
        path = root / name
        if path.is_file():
            digest.update(f"{name}\0{sha256_path(path)}\n".encode())
    return digest.hexdigest()


def package_versions(packages: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str | None]:
    from importlib import metadata

    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except Exception:
            module = _IMPORT_NAME.get(name, name)
            try:
                versions[name] = __import__(module).__version__
            except Exception:
                versions[name] = None
    return versions


def recorded_path(path: str | Path) -> str:
    """A path as a manifest records it: relative to the repository root, never absolute.

    A file outside the repository (a unit test's temporary directory) is recorded by its name only.
    """
    p = Path(path)
    text = relative_to_repo(p)
    return f"<outside the repository>/{p.name}" if os.path.isabs(text) else text


def file_digests(paths) -> dict[str, dict[str, Any]]:
    """SHA-256, size, and repo-relative name for each existing path."""
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        p = Path(path)
        key = recorded_path(p)
        if not p.exists():
            out[key] = {"sha256": None, "bytes": None, "status": "missing"}
            continue
        out[key] = {"sha256": sha256_path(p), "bytes": p.stat().st_size, "status": "ok"}
    return out


def config_hash(config: Any, dataset: Any = None) -> str:
    """SHA-256 of the frozen configuration as a dictionary, together with the dataset configuration when one is given."""
    payload: dict[str, Any] = {"config": config.as_dict() if hasattr(config, "as_dict") else dict(config)}
    if dataset is not None:
        payload["dataset"] = dataset.as_dict() if hasattr(dataset, "as_dict") else dict(dataset)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS bytes.
    return int(usage) * (1024 if platform.system() == "Linux" else 1)


def machine() -> dict[str, Any]:
    """CPU model, physical and logical cores, RAM and OS of this node: no host name, user name, job id or path."""
    cpu, physical_id, cores_per_package = "unknown", None, {}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            key, _, value = line.partition(":")
            key, value = key.strip().lower(), value.strip()
            if key == "model name" and cpu == "unknown":
                cpu = value
            elif key == "physical id":
                physical_id = value
            elif key == "cpu cores" and physical_id is not None and value.isdigit():
                cores_per_package[physical_id] = int(value)
    except OSError:
        pass
    ram_gb = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                ram_gb = round(int(line.split()[1]) / 1024 / 1024, 1)
    except OSError:
        pass
    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = None
    return {"cpu": cpu, "physical_cores": sum(cores_per_package.values()) or None, "logical_cores": os.cpu_count(),
            "cores_available": available, "ram_gb": ram_gb, "os": platform.platform()}


def environment() -> dict[str, Any]:
    """The class of machine and the two switches that change results; nothing that names a host, a user or a job."""
    from fingerprint import smoke_enabled  # the one definition of "smoke" (QMULP_SMOKE=0 is not smoke)

    return {
        **machine(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "smoke": smoke_enabled(),
    }


@dataclass
class RunTimer:
    """Wall-clock bracket for an arm, reported into its manifest."""

    started_at: float = field(default_factory=time.time)
    _perf_start: float = field(default_factory=time.perf_counter)

    def elapsed(self) -> dict[str, Any]:
        return {
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.started_at)),
            "wall_seconds": round(time.perf_counter() - self._perf_start, 3),
        }


def write_manifest(
    output_dir: str | Path,
    arm_name: str,
    config: Any,
    extra: dict[str, Any] | None = None,
    *,
    inputs=(),
    outputs=(),
    seeds: dict[str, Any] | None = None,
    timer: RunTimer | None = None,
    execution_mode: str = "simulation",
    dataset: Any = None,
) -> Path:
    """Write ``manifest.json`` beside an arm's artifacts and return its path.

    ``execution_mode`` describes what actually produced the numbers. It is
    never ``"hardware"`` unless a circuit ran on physical quantum hardware;
    fake-backend and Aer runs are ``"fake_backend"`` and ``"simulation"``.
    ``dataset`` is the dataset configuration of a dataset-bound stage; it enters ``config_hash``.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    config_dict = config.as_dict() if hasattr(config, "as_dict") else dict(config)
    manifest: dict[str, Any] = {
        "arm": arm_name,
        "dataset": getattr(dataset, "name", None),
        "execution_mode": execution_mode,
        "git": git_state(),
        "source_sha256": source_sha256(),
        "versions": package_versions(),
        "config_hash": config_hash(config, dataset),
        "config": config_dict,
        # Provenance of every result table this stage wrote in this process:
        # scope, path, row count and the definition of every column.
        "tables": registered_tables(arm_name),
        "seeds": dict(seeds or {}),
        "inputs": file_digests(inputs),
        "outputs": file_digests(outputs),
        "environment": environment(),
        "peak_rss_bytes": peak_rss_bytes(),
        "output_dir": recorded_path(directory),
    }
    manifest.update(timer.elapsed() if timer is not None else {"wall_seconds": None})
    if extra:
        manifest["extra"] = extra

    path = directory / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return path


def write_stage_manifest(
    dataset: Any,
    stage_key: str,
    config: Any,
    *,
    outputs,
    timer: RunTimer | None,
    extra: dict[str, Any],
    seeds: dict[str, Any] | None = None,
    execution_mode: str = "simulation",
    output_dir: str | Path | None = None,
) -> Path:
    """The manifest of a dataset-bound stage that reads the processed frame and the frozen split.

    The arm is the dataset's directory name for ``stage_key`` and the manifest goes to that stage directory unless
    ``output_dir`` names another. ``seeds`` adds to ``master_seed``; ``extra`` adds the stage's own entries to the
    dataset name and its networks.
    """
    return write_manifest(
        dataset.stage_dir(stage_key) if output_dir is None else output_dir,
        arm_name=dataset.stage(stage_key),
        config=config,
        inputs=[dataset.processed_path(), dataset.frozen_split_path()],
        outputs=outputs,
        seeds={"master_seed": config.master_seed, **(seeds or {})},
        timer=timer,
        execution_mode=execution_mode,
        dataset=dataset,
        extra={"dataset": dataset.name, "networks": list(dataset.ladder), **extra},
    )


# ---------------------------------------------------------------------------
# The result writer
# ---------------------------------------------------------------------------


# stage -> table name -> provenance record.
_REGISTRY: dict[str, dict[str, dict[str, Any]]] = {}

# Column-name conventions shared by the writers and the readers (assembler, number map): the two
# Freddie Mac variability summaries never share a column name.
BOOT_PREFIX = "boot_"          # the bootstrap over test loans (the shared resample matrix)
RESAMPLE_PREFIX = "resample_"  # the summary over the 80% reference resamples


def ci_columns(metric: str, definition: str, *, lo: str | None = None, hi: str | None = None) -> dict[str, str]:
    """Definitions of a point column and its 95% interval bounds (``<metric>_ci_lo`` / ``_ci_hi`` by default)."""
    lo = lo or f"{metric}_ci_lo"
    hi = hi or f"{metric}_ci_hi"
    return {metric: definition, lo: f"lower end of the 95% bootstrap interval of {metric}",
            hi: f"upper end of the 95% bootstrap interval of {metric}"}


def write_result(
    scope: str,
    name: str,
    frame: pl.DataFrame,
    *,
    stage: str,
    columns: Mapping[str, str],
) -> Path:
    """Write ``outputs/results/<scope>/<name>.csv`` and register its provenance for the stage manifest.

    ``columns`` maps a CSV column to its definition. A column of ``frame``
    without a definition is still written (nothing is dropped) and listed
    under ``undocumented`` in the provenance record, so a reader of the
    manifest can see the gap; a definition for a column the frame does not
    carry is recorded under ``absent`` for the same reason.
    """
    if scope not in RESULT_SCOPES:
        raise ValueError(f"result scope must be one of {RESULT_SCOPES}; got {scope!r}")
    if not stage:
        raise ValueError("a result table must name the stage that writes it")
    results_dir(scope)
    path = result_path(scope, name)
    frame.write_csv(path, float_precision=None)
    present = list(frame.columns)
    record = {
        "source_stage": str(stage),
        "scope": scope,
        "path": relative_to_repo(path),
        "n_rows": int(frame.height),
        "columns": {col: str(columns[col]) for col in present if col in columns},
        "undocumented": [col for col in present if col not in columns],
        "absent": [col for col in columns if col not in present],
    }
    _REGISTRY.setdefault(str(stage), {})[name] = record
    return path


def registered_tables(stage: str | None = None) -> dict[str, dict[str, Any]]:
    """The provenance records written so far in this process, for one stage or for all."""
    if stage is None:
        return {name: dict(rec) for records in _REGISTRY.values() for name, rec in records.items()}
    return {name: dict(rec) for name, rec in _REGISTRY.get(str(stage), {}).items()}


def read_result(scope: str, name: str) -> pl.DataFrame | None:
    """One result CSV as a frame with the schema inferred over the whole file, or ``None`` when absent."""
    path = result_path(scope, name)
    if not path.exists():
        return None
    return pl.read_csv(path, infer_schema_length=None)
