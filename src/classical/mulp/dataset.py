"""Loading a processed frame and discovering its one-hot layer groups.

A network has one or two layers, named in block order; the names are whatever
the dataset's ``DatasetConfig`` calls them (``JOB``/``REASON`` on HMEQ,
``PostalCode``/``SellerName`` on Freddie Mac) and this loader does not know the
dataset. Every pipeline caller passes ``layer_names`` explicitly from its
network specification.

``layer_columns`` holds only *genuine* categories. The ``"Missing"`` level a
cleaner produces for a null field is split out and never given a category
node, because "the field was blank" is not an attribute borrowers can share.
See ``preprocessing.missing_levels``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from preprocessing.missing_levels import split_missing_columns

from .config import validate_dataset_path

# The two-layer fixture the operator-contract tests build their rankers on
# without naming a network. Pipeline code never relies on this default.
DEFAULT_LAYER_NAMES = ("JOB", "REASON")
MAX_LAYERS = 2


def validate_layer_names(layer_names) -> tuple[str, ...]:
    """One or two distinct, non-empty layer names, in the order given."""
    names = tuple(str(n) for n in layer_names)
    if not names:
        raise ValueError("at least one layer is required")
    if any(not n for n in names):
        raise ValueError(f"layer names must be non-empty; got {names}")
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate layer names in {names}")
    if len(names) > MAX_LAYERS:
        raise ValueError(f"at most {MAX_LAYERS} layers; got {names}")
    return names


@dataclass(slots=True)
class MuLPDataset:
    path: Path
    df: pl.DataFrame
    # (layer_name, genuine_columns) in layer order.
    layer_columns: list[tuple[str, list[str]]]
    # (layer_name, absence_marker_columns) in the same order.
    layer_missing_columns: list[tuple[str, list[str]]] = field(default_factory=list)

    @property
    def layer_names(self) -> list[str]:
        return [name for name, _ in self.layer_columns]

    def columns_for(self, layer_name: str) -> list[str]:
        for name, cols in self.layer_columns:
            if name == layer_name:
                return cols
        return []

    def missing_columns_for(self, layer_name: str) -> list[str]:
        for name, cols in self.layer_missing_columns:
            if name == layer_name:
                return cols
        return []

    @property
    def job_cols(self) -> list[str]:
        return self.columns_for("JOB")

    @property
    def reason_cols(self) -> list[str]:
        return self.columns_for("REASON")

    @property
    def job_missing_cols(self) -> list[str]:
        return self.missing_columns_for("JOB")

    @property
    def reason_missing_cols(self) -> list[str]:
        return self.missing_columns_for("REASON")


class MuLPDatasetLoader:

    @staticmethod
    def discover_layer_columns(
        df: pl.DataFrame, layer_names=DEFAULT_LAYER_NAMES
    ) -> tuple[list[tuple[str, list[str]]], list[tuple[str, list[str]]]]:
        """Split ``<LAYER>_*`` columns into genuine and marker groups per layer.

        Columns are sorted so the state ordering is deterministic, and the
        ``_was_missing`` numeric indicators a cleaner writes for an imputed
        column are excluded: they are downstream covariates, never graph edges.
        """
        names = validate_layer_names(layer_names)
        # Block order is the frame's own: the first column of each layer's
        # one-hot block fixes where that layer sits, and the names must follow
        # it. A frame carries no layer the dataset did not encode.
        first_column: dict[str, int] = {}
        for name in names:
            positions = [
                i
                for i, c in enumerate(df.columns)
                if c.startswith(f"{name}_") and not c.endswith("_was_missing")
            ]
            if not positions:
                raise ValueError(
                    f"No columns found with {name}_ prefix; layer {name!r} cannot be built"
                )
            first_column[name] = min(positions)
        frame_order = tuple(sorted(names, key=first_column.__getitem__))
        if names != frame_order:
            raise ValueError(
                f"layers must be given in the frame's block order {list(frame_order)}; got {list(names)}"
            )
        genuine: list[tuple[str, list[str]]] = []
        markers: list[tuple[str, list[str]]] = []
        for name in names:
            prefix = f"{name}_"
            columns = sorted(
                c for c in df.columns if c.startswith(prefix) and not c.endswith("_was_missing")
            )
            real, missing = split_missing_columns(columns)
            if not real:
                raise ValueError(
                    f"No columns found with {prefix} prefix; layer {name!r} cannot be built"
                )
            genuine.append((name, real))
            markers.append((name, missing))
        return genuine, markers

    @staticmethod
    def load(dataset: str | Path, layer_names=DEFAULT_LAYER_NAMES) -> MuLPDataset:
        parquet_path = validate_dataset_path(dataset)
        df = pl.read_parquet(parquet_path)
        genuine, markers = MuLPDatasetLoader.discover_layer_columns(df, layer_names)
        return MuLPDataset(
            path=parquet_path,
            df=df,
            layer_columns=genuine,
            layer_missing_columns=markers,
        )

    @staticmethod
    def check_at_most_one_hot(frame: pl.DataFrame, cols: list[str], group_name: str) -> int:
        """Validate at most one active category per row; return how many have none.

        Two or more active values is a hard error -- a broken encoding. *Zero*
        is legitimate: ``cols`` excludes the ``"Missing"`` level, so a borrower
        whose field was never recorded has no active genuine category and
        simply earns no membership edge in that layer.
        """
        sum_expr = pl.sum_horizontal([pl.col(c).cast(pl.Int64) for c in cols]).alias("ones")
        ones = frame.select(sum_expr).get_column("ones").to_list()
        over = [i for i, val in enumerate(ones) if val > 1]
        if over:
            raise ValueError(
                f"{group_name} one-hot validation failed for {len(over)} rows; "
                f"expected at most one active value per row. Sample: {over[:10]}"
            )
        return sum(1 for val in ones if val == 0)

    @staticmethod
    def active_pairs(frame: pl.DataFrame, cols: list[str]) -> list[tuple[int, str]]:
        """Active (borrower_id, category) pairs, in long form.

        Filtering on ``active == 1`` is what makes an unrecorded field cost a
        membership edge and nothing more: an all-zero row produces no pair.
        """
        long_df = (
            frame.with_row_index("borrower_id")
            .select(["borrower_id", *cols])
            .unpivot(index="borrower_id", variable_name="category", value_name="active")
            .filter(pl.col("active").cast(pl.Int64) == 1)
            .select(["borrower_id", "category"])
        )
        return list(long_df.iter_rows())
