"""The variability study behind Table 2: distinct seeds, the frozen split equal to Table 2, and unambiguous Freddie Mac columns.

(1) The HMEQ seeds are the set ``{42} U {1..n_splits_hmeq}``: seed 42, the frozen split, lies inside ``1..50``
and is counted once. (2) The two Freddie Mac summaries never share a column name: ``resample_*`` over the
reference resamples, ``boot_*`` the bootstrap over test loans, and every written column is documented.
(3) The summary records how many splits did not reach ``tau*`` even when none did. (4) On real outputs, when
they exist: no ``(network, seed)`` appears twice, ``n_splits`` equals the number of distinct seeds, and the
frozen-seed row equals Table 2 to 1e-12 -- and the Freddie Mac file names its two families as above.
"""

from __future__ import annotations

import csv
import math
import unittest

from config import DEFAULT_CONFIG
from evaluation.multisplit import HMEQ_FROZEN_SEED, resample_summary, split_seeds, summary_columns, variability_summary
from evaluation.provenance import BOOT_PREFIX, RESAMPLE_PREFIX
from fingerprint import smoke_config
from paths import RESULTS_DIR

HMEQ, FMAC = RESULTS_DIR / "hmeq", RESULTS_DIR / "fmac"
AGREEMENT_COLUMNS = ("rho_qc", "kendall_qc", "ov_qc", "selfov_q", "selfov_c")


def _read(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


class SeedTests(unittest.TestCase):
    def test_the_seeds_are_a_set_and_the_frozen_split_is_counted_once(self):
        seeds = split_seeds(DEFAULT_CONFIG)
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertIn(HMEQ_FROZEN_SEED, seeds)
        self.assertEqual(seeds, list(range(1, 51)), "42 lies inside 1..50, so the study has 50 distinct splits")
        self.assertEqual(len(split_seeds(DEFAULT_CONFIG.with_overrides(n_splits_hmeq=51))), 51)
        self.assertEqual(split_seeds(smoke_config(DEFAULT_CONFIG)), [1, 2, HMEQ_FROZEN_SEED])


class SummaryTests(unittest.TestCase):
    def test_freddie_mac_summaries_never_share_a_column_name_and_every_column_is_documented(self):
        summary = [{"dataset": "fmac", "network": "NET_FMAC_2L", "n_splits": 50, "rho_qc_med": 0.4, "rho_qc_lo": 0.3, "rho_qc_hi": 0.5,
                    "n_single_identical_NET_FMAC_POSTAL": 3}]
        intervals = [{"dataset": "fmac", "network": "NET_FMAC_2L", "rho_qc": 0.39, "rho_qc_lo": 0.36, "rho_qc_hi": 0.42}]
        row = resample_summary(summary, intervals)[0]
        self.assertEqual(row["n_resamples"], 50)
        self.assertEqual((row[f"{RESAMPLE_PREFIX}rho_qc_lo"], row[f"{BOOT_PREFIX}rho_qc_lo"]), (0.3, 0.36))
        for bare in ("rho_qc_lo", "rho_qc_hi", "rho_qc_med", "n_splits", "n_single_identical_NET_FMAC_POSTAL"):
            self.assertNotIn(bare, row)
        documented = summary_columns("resample", ["NET_FMAC_POSTAL"])
        self.assertEqual(set(row) - set(documented), set(), "a written column without a definition")
        self.assertEqual(set(summary_columns("bootstrap")) - {"dataset", "network"}, {"rho_qc", "rho_qc_lo", "rho_qc_hi", "kendall_qc",
                                                                                    "kendall_qc_lo", "kendall_qc_hi", "ov_qc", "ov_qc_lo", "ov_qc_hi"})

    def test_the_count_of_splits_that_missed_tau_star_is_recorded_even_when_none_reached_it(self):
        rows = [{"dataset": "hmeq", "network": "NET_2L", "seed": seed, "rho_qc": 0.1 * seed, "tau_star": "> tau_max", "tau_star_reached": False}
                for seed in (1, 2, 3)]
        rows += [{"dataset": "hmeq", "network": "NET_JOB", "seed": seed, "rho_qc": 1.0, "tau_star": None, "tau_star_reached": None}
                 for seed in (1, 2, 3)]
        by_network = {r["network"]: r for r in variability_summary(rows, frozen_seed=None)}
        self.assertEqual((by_network["NET_2L"]["n_tau_star_evaluated"], by_network["NET_2L"]["n_tau_star_not_reached"]), (3, 3))
        self.assertNotIn("tau_star_med", by_network["NET_2L"])
        self.assertNotIn("n_tau_star_evaluated", by_network["NET_JOB"])
        self.assertEqual(by_network["NET_2L"]["n_splits"], 3)


@unittest.skipUnless((HMEQ / "EC_multisplit_long.csv").exists() and (HMEQ / "T2_agreement.csv").exists(),
                     "no HMEQ variability outputs yet")
class RealOutputTests(unittest.TestCase):
    def test_distinct_seeds_and_the_frozen_row_equals_table_2(self):
        long = _read(HMEQ / "EC_multisplit_long.csv")
        table2 = {r["network"]: r for r in _read(HMEQ / "T2_agreement.csv")}
        pairs = [(r["network"], r["seed"]) for r in long]
        self.assertEqual(len(pairs), len(set(pairs)), "a (network, seed) pair appears twice")
        frozen = [r for r in long if int(r["seed"]) == HMEQ_FROZEN_SEED]
        self.assertEqual(sorted(r["network"] for r in frozen), sorted(table2))
        for row in frozen:
            for column in AGREEMENT_COLUMNS:
                a, b = float(row[column]), float(table2[row["network"]][column])
                if math.isnan(a) and math.isnan(b):
                    continue
                self.assertAlmostEqual(a, b, delta=1e-12, msg=f"{row['network']}.{column}: variability {a!r} vs Table 2 {b!r}")
        for record in _read(HMEQ / "T2_variability.csv"):
            distinct = len({r["seed"] for r in long if r["network"] == record["network"]})
            self.assertEqual(int(float(record["n_splits"])), distinct, record["network"])

    @unittest.skipUnless((FMAC / "T2_variability.csv").exists(), "no Freddie Mac variability outputs yet")
    def test_the_freddie_mac_file_names_its_two_families(self):
        with open(FMAC / "T2_variability.csv", newline="") as handle:
            header = next(csv.reader(handle))
        self.assertIn("n_resamples", header)
        self.assertIn(f"{BOOT_PREFIX}rho_qc_lo", header)
        self.assertIn(f"{RESAMPLE_PREFIX}rho_qc_med", header)
        for bare in ("rho_qc_lo", "rho_qc_hi", "rho_qc_med", "n_splits"):
            self.assertNotIn(bare, header)


if __name__ == "__main__":
    unittest.main()
