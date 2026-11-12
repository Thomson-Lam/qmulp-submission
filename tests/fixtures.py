"""Shared fixture of the lumping and mechanism tests: a 40-borrower two-layer frame and its operators.

40 borrowers over 3 genuine JOB categories, 2 genuine REASON categories, with
unrecorded fields on both sides and uneven group sizes -- an exactness claim
that only holds for equal-sized groups is not an exactness claim. "-" means
the field was never recorded; no borrower is missing both. ``LumpingFixture``
builds the production ``MultiLayerRanker`` on the frame, materialises its
operator, and lumps the frame both ways; ``assert_aggregates`` is the check
that a reduced vector sums its members and matches every category node.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from classical.mulp import MultiLayerRanker
from quantum.inspired.szegedy_walk import SzegedyWalk
from quantum.lumping import build_topk_lumped_subgraph

ALPHA = 0.85
STICKINESS = 1.0
TAU = 10
ATOL = 1e-10

JOB_COLS = ["JOB_A", "JOB_B", "JOB_C", "JOB_Missing"]
REASON_COLS = ["REASON_Missing", "REASON_X", "REASON_Y"]

SPEC: list[tuple[str, str, int]] = (
    [("A", "X", 0)] * 7 + [("A", "X", 1)] * 3 + [("A", "Y", 0)] * 4 + [("B", "X", 0)] * 6
    + [("B", "X", 1)] * 2 + [("B", "Y", 0)] * 3 + [("B", "Y", 1)] * 1 + [("C", "X", 0)] * 5
    + [("C", "Y", 0)] * 2 + [("-", "X", 0)] * 3 + [("-", "Y", 1)] * 1 + [("A", "-", 0)] * 2
    + [("C", "-", 1)] * 1
)


def fixture_frame() -> pd.DataFrame:
    assert len(SPEC) == 40, "the fixture is specified as 40 borrowers"
    rows: list[dict[str, int]] = []
    for job, reason, bad in SPEC:
        row = {col: 0 for col in (*JOB_COLS, *REASON_COLS)}
        row["JOB_Missing" if job == "-" else f"JOB_{job}"] = 1
        row["REASON_Missing" if reason == "-" else f"REASON_{reason}"] = 1
        row["BAD"] = bad
        rows.append(row)
    return pd.DataFrame(rows, columns=[*JOB_COLS, *REASON_COLS, "BAD"])


def stationary_eig(G: np.ndarray) -> np.ndarray:
    """Leading eigenvector of a column-stochastic G, normalized to sum 1."""
    values, vectors = np.linalg.eig(G)
    leading = int(np.argmin(np.abs(values - 1.0)))
    pi = np.real(vectors[:, leading])
    pi = pi / pi.sum()
    assert float(np.linalg.norm(G @ pi - pi, 1)) < 1e-10
    return pi


def tau_average(G: np.ndarray, v: np.ndarray, bl1, bl2, tau: int = TAU) -> np.ndarray:
    return SzegedyWalk(G, v, bl1, bl2).supra_node_history(tau).mean(axis=0)


def member_lists(nodes: pd.DataFrame) -> dict[int, list[int]]:
    return {position: [int(m) for m in str(row["member_borrower_ids"]).split(";")]
            for position, (_, row) in enumerate(nodes.iterrows()) if row["node_type"] == "borrower_copy"}


def category_positions(nodes: pd.DataFrame) -> dict[str, int]:
    return {str(row["label"]): position for position, (_, row) in enumerate(nodes.iterrows()) if row["node_type"] == "category"}


class LumpingFixture:
    """Everything both the classical and the quantum assertions need."""

    def __init__(self, tmpdir: str) -> None:
        self.df = fixture_frame()
        self.borrower_ids = list(range(len(self.df)))
        path = Path(tmpdir) / "lumpability_fixture.parquet"
        pl.DataFrame(self.df.to_dict(orient="list")).write_parquet(path)

        self.ranker = MultiLayerRanker(str(path), sparse=False, stickiness=STICKINESS)
        self.ranker.buildAdjMatrix()
        self.ranker.buildPersonalizationMatrix()
        self.pi_full = np.asarray(self.ranker.pageRank(alpha=ALPHA), dtype=float)
        # The operator is matrix-free; materialise it for this 40-borrower fixture.
        self.G_full = np.asarray(self.ranker.pagerank_operator @ np.eye(self.ranker.N), dtype=float)
        self.v_full = np.asarray(self.ranker._dangling_patch_vector(), dtype=float)
        self.bl1_full = np.arange(self.ranker.n_rows, dtype=int)
        self.bl2_full = self.ranker.reason_offset + self.bl1_full

        self.nodes_none, self.edges_none = build_topk_lumped_subgraph(
            self.df, JOB_COLS, REASON_COLS, self.borrower_ids, lumping_level="none", stickiness=STICKINESS)
        self.nodes_lump, self.edges_lump = build_topk_lumped_subgraph(
            self.df, JOB_COLS, REASON_COLS, self.borrower_ids, lumping_level="attr_bad", stickiness=STICKINESS)

    def node_of(self, layer: str):
        offset = self.ranker.job_offset if layer == "JOB" else self.ranker.reason_offset
        return lambda borrower: offset + int(borrower)

    def category_node(self, label: str) -> int:
        layer, column = label.split(":", 1)
        if layer == "JOB":
            return self.ranker.index.job_category_node(self.ranker.job_cols.index(column))
        return self.ranker.index.reason_category_node(self.ranker.reason_cols.index(column))

    def assert_aggregates(self, case: unittest.TestCase, nodes: pd.DataFrame, reduced: np.ndarray, full: np.ndarray) -> None:
        """Every class value equals the sum over its members; every category equals its full node."""
        for position, members in member_lists(nodes).items():
            node_of = self.node_of(nodes.iloc[position]["layer"])
            case.assertAlmostEqual(reduced[position], float(sum(full[node_of(m)] for m in members)), delta=ATOL,
                                   msg=f"position {position} ({len(members)} members)")
        for label, position in category_positions(nodes).items():
            case.assertAlmostEqual(reduced[position], float(full[self.category_node(label)]), delta=ATOL, msg=label)

