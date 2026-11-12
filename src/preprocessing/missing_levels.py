"""The repo-wide rule for which one-hot levels mean "absent" rather than a category.

Every cleaner (``hmeq.cleaning``, the Freddie Mac cleaning notebook) fills a
null categorical value with the literal string :data:`MISSING_SENTINEL` before
one-hot encoding, so ``JOB_Missing`` or ``PostalCode_Missing`` emerge from the
encoding looking exactly like genuine categories; and every imputed numeric
column keeps a retained ``<col>`` + :data:`MISSING_INDICATOR_SUFFIX` flag, a
downstream covariate that is never a graph edge.

An absence marker must never become a category node. A category node is a
"borrowers who share this attribute" relation, and a blank field is not an
attribute: linking every borrower whose field was blank into one clique would
put a hub in the middle of the graph that propagates default risk between
borrowers with nothing in common. On HMEQ that hub's membership is comparable
in size to several genuine categories, so it is not a small artifact that can
be waved away. The counts are deliberately not written down here; call
:func:`absence_hub_sizes` on the frame in hand and report them as data.

This module is the single definition of both the fill value and the suffix
rule, so the two cannot drift apart, and it depends on nothing else in the repo
and on no third-party package. Borrowers with *no* recorded categorical
attribute in any layer are removed once, by the dataset's cleaner.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

MISSING_SENTINEL = "Missing"
MISSING_SUFFIX = "_" + MISSING_SENTINEL
# The retained flag an imputed numeric column leaves behind.
MISSING_INDICATOR_SUFFIX = "_was_missing"


def is_missing_column(column: str, missing_suffix: str = MISSING_SUFFIX) -> bool:
    """Does this one-hot column encode an absent value rather than a category?

    Tested on the *suffix*, not containment: a genuine category whose name
    merely contains the word (``"JOB_MissingPersonsBureau"``) keeps its hub.
    """
    return str(column).endswith(missing_suffix)


def split_missing_columns(
    cols: Sequence[str], missing_suffix: str = MISSING_SUFFIX
) -> tuple[list[str], list[str]]:
    """Partition column names into (genuine categories, absence markers).

    Order within each list follows the input.
    """
    real: list[str] = []
    missing: list[str] = []
    for col in cols:
        (missing if is_missing_column(col, missing_suffix) else real).append(str(col))
    return real, missing


def missing_category_indices(cols: Sequence[str], missing_suffix: str = MISSING_SUFFIX) -> set[int]:
    """Positions, within ``cols`` exactly as given, of the absence markers.

    ``cols`` must be a :class:`Sequence`, not a bare iterable: the return value
    indexes into it, so the caller's ordering is part of the contract and the
    same ``cols`` has to be re-iterable to interpret the result.
    """
    return {i for i, col in enumerate(cols) if is_missing_column(col, missing_suffix)}


def _active_flags(frame: Any, column: str) -> list[int]:
    """Per-row 0/1 activity for one column, for pandas and polars alike."""
    flags: list[int] = []
    for value in frame[column]:
        if value is None:
            flags.append(0)
            continue
        try:
            flags.append(1 if int(value) == 1 else 0)
        except (TypeError, ValueError):
            flags.append(1 if bool(value) else 0)
    return flags


def absence_hub_sizes(
    frame: Any,
    *column_groups: Sequence[str],
    markers: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Borrower counts per absence marker and per genuine category.

    Accepts one column group per layer, so it generalizes from the two-layer
    JOB/REASON graph to any number of layers. Emitting the counts as data keeps
    the absence-hub argument recomputed on the frame actually in use.

    Returns ``{"marker_sizes", "genuine_sizes", "n_rows", "largest_genuine"}``.
    """
    all_cols = [str(col) for group in column_groups for col in group]
    genuine, found_markers = split_missing_columns(all_cols)
    marker_cols = [str(c) for c in markers] if markers is not None else found_markers

    marker_sizes = {col: sum(_active_flags(frame, col)) for col in marker_cols}
    genuine_sizes = {col: sum(_active_flags(frame, col)) for col in genuine}

    n_rows = len(_active_flags(frame, all_cols[0])) if all_cols else 0
    largest_genuine: tuple[str, int] | None = None
    if genuine_sizes:
        col = max(genuine_sizes, key=lambda name: (genuine_sizes[name], name))
        largest_genuine = (col, genuine_sizes[col])

    return {
        "marker_sizes": marker_sizes,
        "genuine_sizes": genuine_sizes,
        "n_rows": n_rows,
        "largest_genuine": largest_genuine,
    }


def assert_absence_consistent(
    frame: Any,
    genuine_cols: Sequence[str],
    marker_cols: Sequence[str],
    group_name: str,
) -> None:
    """Every row must have exactly one active column across genuine + markers.

    ``check_at_most_one_hot`` deliberately accepts *zero* active genuine
    columns, because "never recorded" is a real state costing the borrower a
    membership edge and nothing more. That tolerance leaves one corruption
    invisible: a row with neither a genuine category nor a marker. Such a row
    is a broken encoding, not an unrecorded field, and would become an isolated
    node.
    """
    cols = [str(col) for col in (*genuine_cols, *marker_cols)]
    if not cols:
        raise ValueError(f"{group_name}: no columns given to check")

    columns = [_active_flags(frame, col) for col in cols]
    totals = [sum(col[row] for col in columns) for row in range(len(columns[0]))]
    empty = [row for row, total in enumerate(totals) if total == 0]
    over = [row for row, total in enumerate(totals) if total > 1]
    if empty or over:
        raise ValueError(
            f"{group_name} absence consistency failed: expected exactly one active "
            f"column per row across genuine {list(genuine_cols)} + markers "
            f"{list(marker_cols)}; {len(empty)} row(s) have none "
            f"(sample {empty[:10]}) and {len(over)} row(s) have more than one "
            f"(sample {over[:10]})"
        )
