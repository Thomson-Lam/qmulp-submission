"""Repository-anchored paths. Nothing here depends on the process working directory.

The root is the directory holding this file. ``nbconvert`` executes notebooks
with ``notebooks/`` as the working directory, so a path resolved against the
working directory would land in the wrong place; every path in the repository
is therefore derived from :data:`ROOT` and every stage, notebook, script and
test goes through the helpers below.

A recorded absolute path is useless outside the machine that produced it, so
:func:`relative_to_repo` is what artifacts record.

Dataset-specific locations (a raw file, a processed frame, a frozen split, a
stage directory name) are not defined here: they belong to the dataset's own
``DatasetConfig`` (``hmeq/config.py``, ``fmac/config.py``). The file names of the
stage artifacts that another stage or script reads (the scores, the convergence
record, the classical scores, the reduction-check history) are defined here once,
under the stage directory name a dataset's ``stage`` gives.

Output contract
---------------
``outputs/results/<scope>/<name>.csv`` is the one data artifact of every
result table (``scope`` is a dataset name, ``synthetic`` or ``runtime``);
``outputs/manuscript/`` holds what the manuscript includes (``tables/*.tex``,
``figures/*.pdf``, ``numbers.tex``, ``number_map.csv``) and is written only by
the assembler scripts; ``outputs/notebook_pipeline/<stage>/`` holds the
intermediates and the stage manifests.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "results" / "outputs"
PIPELINE_DIR = OUTPUTS_DIR / "notebook_pipeline"
RESULTS_DIR = OUTPUTS_DIR / "results"
MANUSCRIPT_DIR = OUTPUTS_DIR / "manuscript"
MASKED_DIR = PIPELINE_DIR / "masked"
LOGS_DIR = OUTPUTS_DIR / "logs"

# The scopes a result table may be written under: one per dataset, plus the
# dataset-free experiments. Two datasets never write to the same file.
RESULT_SCOPES: tuple[str, ...] = ("hmeq", "fmac", "synthetic", "runtime")


def repo_root() -> Path:
    """Absolute path to the repository root (the directory holding this file).

    The root is a property of the checkout, not of where the caller is running.
    """
    return ROOT


def repo_path(path: str | Path) -> Path:
    """Resolve a repo-relative configuration path; absolute paths pass through."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def relative_to_repo(path: str | Path, *, root: str | Path | None = None) -> str:
    """POSIX path relative to the repo root, for recording in artifacts.

    A path outside the repository is returned resolved and absolute: silently
    rewriting it would be worse than reporting it truthfully.
    """
    base = Path(root).resolve() if root is not None else ROOT
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(base).as_posix()
    except ValueError:
        return resolved.as_posix()


def assert_no_absolute_paths(values, *, context: str = "artifact") -> None:
    """Fail if any recorded value looks like an absolute filesystem path."""
    offenders = [
        str(value)
        for value in values
        if isinstance(value, (str, Path)) and str(value).startswith("/")
    ]
    if offenders:
        raise AssertionError(
            f"{context} records absolute path(s) {offenders[:5]}; use "
            "paths.relative_to_repo so the artifact is portable"
        )


def stage_path(name: str) -> Path:
    """Output directory of one stage under the pipeline directory, **not** created.

    For code that only computes or lists paths (``scripts.warm --list`` must leave
    ``outputs/`` untouched). ``name`` is the directory name a dataset's
    ``DatasetConfig.stage`` maps a stage key to; the registry of valid names lives
    there, not here.
    """
    if not name or "/" in name or name in (".", ".."):
        raise ValueError(f"stage name must be one plain directory name; got {name!r}")
    return PIPELINE_DIR / name


def stage_dir(name: str) -> Path:
    """Output directory of one stage under the pipeline directory, created on demand."""
    directory = stage_path(name)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def results_dir(scope: str) -> Path:
    """``outputs/results/<scope>/``, created on demand; ``scope`` must be one of :data:`RESULT_SCOPES`."""
    if scope not in RESULT_SCOPES:
        raise ValueError(f"result scope must be one of {RESULT_SCOPES}; got {scope!r}")
    directory = RESULTS_DIR / scope
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def result_path(scope: str, name: str) -> Path:
    """The CSV of result table ``name`` under ``scope`` (the directory is not created)."""
    if scope not in RESULT_SCOPES:
        raise ValueError(f"result scope must be one of {RESULT_SCOPES}; got {scope!r}")
    if not name or "/" in name or name.endswith(".csv"):
        raise ValueError(f"result name must be a bare table name; got {name!r}")
    return RESULTS_DIR / scope / f"{name}.csv"


def manuscript_dir(kind: str | None = None) -> Path:
    """``outputs/manuscript/`` or one of its ``tables`` / ``figures`` subdirectories, created on demand."""
    directory = MANUSCRIPT_DIR if kind is None else MANUSCRIPT_DIR / kind
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def masked_dir(spec_name: str) -> Path:
    """Where the label-masked frames of one network live, shared by every arm.

    Inputs to notebooks 03 and 04 alike: writing them once is what lets both
    arms provably read the same bytes.
    """
    directory = MASKED_DIR / spec_name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def masked_frame_path(spec_name: str, tag: str) -> Path:
    return masked_dir(spec_name) / f"masked_{tag}.parquet"


def classical_scores_path(spec_name: str, *, stage: str) -> Path:
    """Stage 03's classical scores of one network (``omega_C_test`` and the cross-fitted column)."""
    return stage_dir(stage) / f"classical_scores_{spec_name}.parquet"


def quantum_history_path(spec_name: str, tag: str, *, stage: str, create: bool = True) -> Path:
    """The full-operator step history of the reduction check (``quantum_history_<network>_<tag>.npy``).

    ``create=False`` leaves the stage directory uncreated, for code that only lists the path.
    """
    return (stage_dir(stage) if create else stage_path(stage)) / f"quantum_history_{spec_name}_{tag}.npy"


def scores_path(spec_name: str, *, stage: str) -> Path:
    return stage_dir(stage) / f"scores_{spec_name}.parquet"


def convergence_path(spec_name: str, *, stage: str) -> Path:
    return stage_dir(stage) / f"convergence_{spec_name}.json"
