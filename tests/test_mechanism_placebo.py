"""Stage 05a needs stage 05's placebo summary, and says so when it is missing.

Table 4 prints, for each arm, the mean top-k Jaccard with the placebo rankings and the placebo p-value of
precision@k. Both come from ``EC_placebo``, which stage 05 writes. In the first clean run stage 05a started
three seconds before stage 05, found no table, and wrote both columns empty. The stage now raises before it
does any work, and the two columns are looked up strictly: the row of the network and arm must exist and both
values must be finite. The local job script makes 05a wait for 05 (``afterok``).
"""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import polars as pl

from evaluation import provenance
from evaluation.provenance import write_result
from hmeq.config import DATASET as HMEQ
from paths import ROOT
from quantum.mechanism.studies import placebo_columns
from stages import mechanism

K = 100


def summary_rows(network: str = "NET_2L") -> list[dict]:
    """The rows stage 05 writes to ``EC_placebo``: one per arm, with the two columns Table 4 reads."""
    return [{"dataset": "hmeq", "network": network, "arm": arm, "n_perm": 99, f"mean_top_{K}_jaccard_with_true": jaccard,
             f"pval_precision_at_{K}": p_value, "routes": "D_bar"}
            for arm, jaccard, p_value in (("qppr", 0.5459, 0.05), ("classical", 0.1693, 0.20))]


class PlaceboSummaryTests(unittest.TestCase):
    def test_the_stage_refuses_to_start_when_stage_05_has_not_written_the_summary(self):
        with tempfile.TemporaryDirectory() as td, mock.patch("paths.RESULTS_DIR", Path(td) / "results"):
            with self.assertRaises(FileNotFoundError) as caught:
                mechanism.run("hmeq")
            self.assertEqual(list(Path(td).rglob("*")), [], "it refuses before it writes anything")
        self.assertIn("stage 05 (evaluation) has not written EC_placebo for hmeq; run stage 05 first", str(caught.exception))

    def test_the_summary_stage_05_wrote_gives_finite_placebo_columns(self):
        with tempfile.TemporaryDirectory() as td, mock.patch("paths.RESULTS_DIR", Path(td) / "results"):
            write_result("hmeq", "EC_placebo", pl.DataFrame(summary_rows()), stage="99_unit", columns={})
            summary = mechanism._placebo_summary(HMEQ)
        provenance._REGISTRY.pop("99_unit", None)
        qppr, classical = (placebo_columns(summary, "NET_2L", arm, K) for arm in ("qppr", "classical"))
        self.assertEqual(set(qppr), {"jacc_placebo", "pval_p100"})
        self.assertAlmostEqual(qppr["jacc_placebo"], 0.5459, places=12)
        self.assertAlmostEqual(qppr["pval_p100"], 0.05, places=12)
        self.assertAlmostEqual(classical["jacc_placebo"], 0.1693, places=12)
        self.assertAlmostEqual(classical["pval_p100"], 0.20, places=12)

    def test_a_missing_row_an_absent_column_an_empty_value_or_a_nan_raises(self):
        rows = summary_rows()
        for network, arm, summary, k in (
            ("NET_2L", "qppr", [rows[1]], K),                                                          # the arm's row is absent
            ("NET_JOB", "qppr", rows, K),                                                             # another network
            ("NET_2L", "qppr", rows + rows[:1], K),                                                   # two rows for one arm
            ("NET_2L", "qppr", rows, 25),                                                             # the summary holds k = 100 only
            ("NET_2L", "qppr", [{**rows[0], f"mean_top_{K}_jaccard_with_true": float("nan")}, rows[1]], K),
            ("NET_2L", "qppr", [{**rows[0], f"pval_precision_at_{K}": None}, rows[1]], K),
        ):
            with self.subTest(network=network, arm=arm, k=k, rows=len(summary)):
                with self.assertRaises(ValueError):
                    placebo_columns(summary, network, arm, k)


LOCAL_CHAIN = ROOT / "scripts" / "run_pipeline.sh"


@unittest.skipUnless(LOCAL_CHAIN.exists(), "scripts/run_pipeline.sh is local and untracked; a clone without it skips this check")
class ChainOrderTests(unittest.TestCase):
    def test_05a_waits_for_05_and_not_for_04_alone(self):
        text = LOCAL_CHAIN.read_text()
        submitted = re.search(r'^\s*job05a=\$\(submit "\$\(script_for "\$dataset" 05a\)" "(\$job0[45])"\)', text, re.MULTILINE)
        self.assertIsNotNone(submitted, "run_pipeline.sh no longer submits 05a in the expected form")
        self.assertEqual(submitted.group(1), "$job05")
        self.assertIn("-> 05 -> 05a", text, "the header diagram shows the order")


if __name__ == "__main__":
    unittest.main()
