"""Assemble the manuscript's ``.tex`` tables and the number map from the result CSVs.

    python -m scripts.make_manuscript_tables

The only writer of ``.tex`` in the repository. For every table of
``evaluation.table_specs`` it reads the result CSVs under
``outputs/results/<scope>/`` (both datasets' scopes for a dataset table),
builds the frame, formats every cell through ``evaluation.formatting`` and
writes the **bare** ``tabular`` (booktabs, no vertical rules, no ``table``
environment, caption, label or notes: those live in the manuscript inside
``\\TABLE{}{}{}``) to ``outputs/manuscript/tables/<name>.tex``: the panels of a
table stacked in one ``tabular`` on a shared grid of columns, the rows of each
group (dataset, or dataset and network) under a ``\\midrule`` and an italic label.
It also writes ``outputs/manuscript/table_columns.csv``, one row per printed
column with its header and its definition (the text the manuscript's note
under the table reuses), then ``outputs/manuscript/number_map.csv`` and
``numbers.tex`` (``evaluation.number_map``). A table whose required inputs are
missing is reported and skipped; no number is typed here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import DATASET_NAMES  # noqa: E402
from evaluation.formatting import RULES, Block, cell_span, human_label, tabular  # noqa: E402
from evaluation.number_map import write_number_map  # noqa: E402
from evaluation.provenance import read_result  # noqa: E402
from evaluation.table_specs import RAW, TABLE_SPECS, TableSpec, stacked  # noqa: E402
from paths import MANUSCRIPT_DIR, PIPELINE_DIR, RESULTS_DIR, manuscript_dir  # noqa: E402


def load_frames(spec: TableSpec) -> dict[tuple[str, str], pl.DataFrame | None]:
    """Every source of ``spec``, read from ``outputs/results``; a missing file maps to ``None``."""
    return {(scope, name): read_result(scope, name) for scope, name in spec.expanded_sources()}


def missing_required(spec: TableSpec, frames: Mapping[tuple[str, str], pl.DataFrame | None]) -> list[str]:
    """Required source names that no scope provides."""
    required = [name for _scope, name in spec.sources if name not in spec.optional]
    return [name for name in required if all(frames.get((scope, n)) is None for scope, n in spec.expanded_sources() if n == name)]


def _sort_key(spec: TableSpec, row: Mapping[str, Any]):
    key = []
    for column in spec.sort:
        value = row.get(column)
        if column == "dataset":
            value = DATASET_NAMES.index(value) if value in DATASET_NAMES else len(DATASET_NAMES)
        key.append((value is None, value if value is not None else 0))
    return tuple(key)


def grid_width(spec: TableSpec) -> int:
    """The number of grid columns of a table: the widest panel, a spanned cell counting for its span."""
    return max(sum(column.span for column in panel.columns) for panel in spec.panels)


def _padded(cells: list[str], width: int) -> list[str]:
    """``cells`` followed by empty cells up to ``width`` grid columns."""
    return cells + [""] * (width - sum(cell_span(cell) for cell in cells))


def _groups(spec: TableSpec, rows: list[dict[str, Any]]) -> list[tuple[str, int]]:
    """Consecutive rows with the same group value as ``(label, n_rows)``, the label through the one mapping of ``human_label``."""
    groups: list[tuple[str, int]] = []
    for row in rows:
        value = row.get(spec.group_by)
        if value is None:
            raise ValueError(f"{spec.name}: a row has no {spec.group_by!r} to group by")
        label = human_label(value)
        if groups and groups[-1][0] == label:
            groups[-1] = (label, groups[-1][1] + 1)
        else:
            groups.append((label, 1))
    return groups


def render(spec: TableSpec, frames: Mapping[tuple[str, str], pl.DataFrame | None]) -> str | None:
    """The bare tabular of ``spec`` from its source frames, or ``None`` when the frame cannot be built."""
    for column in spec.columns:
        if column.rule != RAW and column.rule not in RULES:
            raise ValueError(f"{spec.name}: column {column.source!r} has unknown rule {column.rule!r}")
        if not column.definition.strip():
            raise ValueError(f"{spec.name}: column {column.source!r} has no definition")
    frame = spec.build(frames) if spec.build is not None else stacked(frames, spec.sources[0][1])
    if frame is None or frame.height == 0:
        return None
    rows = [r for r in frame.to_dicts() if spec.where is None or spec.where(r)]
    if spec.sort:
        rows.sort(key=lambda r: _sort_key(spec, r))
    width = grid_width(spec)
    blocks = []
    for panel in spec.panels:
        mine = [r for r in rows if panel.where is None or panel.where(r)]
        if not mine:
            continue
        headers = _padded([column.header_cell() for column in panel.columns], width)
        cells = [_padded([column.cell(row) for column in panel.columns], width) for row in mine]
        grouped = _groups(spec, mine) if spec.group_by is not None and panel.grouped else None
        blocks.append(Block(panel.title, headers, cells, grouped))
    return tabular(align=spec.align, panels=blocks) if blocks else None


def column_rows(spec: TableSpec) -> list[dict[str, str]]:
    """One row per printed column of ``spec``: its panel, header, source column, rule and definition."""
    return [{"table": spec.name, "panel": panel.title or "", "header": column.header, "source": column.source, "rule": column.rule,
             "definition": column.definition} for panel in spec.panels for column in panel.columns]


def assemble(specs: Iterable[TableSpec] = TABLE_SPECS) -> tuple[dict[str, str], list[str]]:
    """Every table of ``specs`` whose required inputs exist, rendered from the result CSVs under ``outputs/results``.

    Returns ``({table name: tabular}, [one message per table that was skipped, with the reason])``. Nothing is written:
    :func:`main` writes the tables, and the tests render them through this function so that they measure what the
    current code builds from the results rather than a file an earlier run left behind.
    """
    tables: dict[str, str] = {}
    skipped: list[str] = []
    for spec in specs:
        frames = load_frames(spec)
        missing = missing_required(spec, frames)
        if missing:
            skipped.append(f"{spec.name}: missing {', '.join(missing)}; skipped")
            continue
        tex = render(spec, frames)
        if tex is None:
            skipped.append(f"{spec.name}: no rows; skipped")
            continue
        tables[spec.name] = tex
    return tables, skipped


def main() -> int:
    out_dir = manuscript_dir("tables")
    tables, skipped = assemble()
    for message in skipped:
        print(message)
    written: list[Path] = []
    for name, tex in tables.items():
        path = out_dir / f"{name}.tex"
        path.write_text(tex)
        written.append(path)
    defined = [row for spec in TABLE_SPECS if spec.name in tables for row in column_rows(spec)]
    if defined:
        pl.DataFrame(defined).write_csv(MANUSCRIPT_DIR / "table_columns.csv")
    report = write_number_map(RESULTS_DIR, PIPELINE_DIR, MANUSCRIPT_DIR)
    print(f"number map: {report['n_resolved']} of {report['n_keys']} keys resolved -> {MANUSCRIPT_DIR / 'numbers.tex'}")
    for r in report["unresolved"]:
        print(f"  unresolved {r['key']}: {r['status']} ({r['source']})")
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
