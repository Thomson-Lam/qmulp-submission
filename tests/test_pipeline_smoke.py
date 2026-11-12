"""One end-to-end run over a small synthetic frame: clean, freeze, build, walk, lump, score, table.

The chain is the pipeline's, on module functions with temporary paths: the
cleaner writes the processed frame and the frozen split; every later step
loads that split rather than recomputing one; the classical and quantum arms
are built on the same masked frame and asserted to be the same operator (digest
for digest); the exact walk on the lumped operator unfolds to the unlumped walk
with zero reduction error; the shared bootstrap gives marginal intervals and a
paired difference; and the result writer produces its CSV under the scope.
Also here: the configuration carries no fixed horizon, every seed derives from
the master seed, and a cache key changes with the operator it was built on.
"""

from __future__ import annotations

import ast
import inspect
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from config import DEFAULT_CONFIG, derive_seed
from evaluation.masked_eval import load_partitions
from evaluation.provenance import write_result
from evaluation.reporting import agreement_with, bootstrap_replicates, metrics_with_ci, paired_difference
from evaluation.ties import canonicalize
from fingerprint import (
    assert_operator_identity,
    bootstrap_indices,
    config_fingerprint,
    operator_cache_extra,
    operator_digests,
    smoke_config,
    solve_classical,
)
from hmeq.config import DATASET as HMEQ, NET_2L, NETWORK_SPECS
from quantum.inspired.mulp import QuantumWalkRanker
from quantum.inspired.szegedy_walk import SzegedyWalk
from quantum.inspired.walk_cache import run_checkpointed
from quantum.lumped_operator import borrower_layer_positions, build_lumped_matrices, unfold_class_scores
from quantum.lumping import full_lumped_subgraph, lumped_facts
from tests.test_leakage import clean_to

TAU = 6
K = 10


@pytest.mark.slow
class PipelineSmokeTest(unittest.TestCase):
    def test_clean_through_tables_on_a_synthetic_frame(self):
        config = DEFAULT_CONFIG.with_overrides(reduction_check_steps=TAU, bootstrap_reps=50, primary_k=K, top_k=(5, K))
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            processed, split, result = clean_to(root)
            self.assertTrue(processed.exists() and split.exists())

            # Every stage loads the frozen split; none recomputes it.
            partitions = load_partitions(processed, split)
            np.testing.assert_array_equal(partitions.test_idx, result.test_idx)
            for function in (load_partitions, solve_classical, full_lumped_subgraph):
                tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
                called = {getattr(n.func, "id", getattr(n.func, "attr", "")) for n in ast.walk(tree) if isinstance(n, ast.Call)}
                self.assertNotIn("train_test_split", called, function.__name__)

            # Test labels masked before the operator exists; both arms on the same bytes.
            masked_path = root / "masked_test_operator.parquet"
            partitions.mask_labels(partitions.test_idx).write_parquet(masked_path)
            spec = NETWORK_SPECS[NET_2L.name]
            bundle = solve_classical(masked_path, spec, config)
            self.assertAlmostEqual(float(bundle.pi.sum()), 1.0, places=10)
            walker = QuantumWalkRanker(ranker=solve_classical(masked_path, spec, config).ranker, alpha=config.resolved_alpha())
            walker.buildAdjMatrix()
            walker.buildPersonalizationMatrix()
            walker.buildTransitionMatrix()
            walker.buildPersonalizationVector()
            walker.buildSzegedyWalk()
            assert_operator_identity(bundle.digests, operator_digests(walker.ranker), "smoke operator")

            # The checkpointed walk is what the stage runs; the plain run must agree with it.
            run = run_checkpointed(walker.szegedy_walk, TAU, root / "checkpoint", every=2)
            self.assertEqual(run.history.shape, (TAU, partitions.frame.height))
            np.testing.assert_allclose(run.omega, walker.quantumPageRank(tau=TAU), atol=1e-12)
            resumed = run_checkpointed(walker.szegedy_walk, TAU, root / "checkpoint", every=2)
            np.testing.assert_array_equal(resumed.history, run.history)

            # The exact walk on the lumped operator unfolds to the unlumped walk.
            nodes, edges = full_lumped_subgraph(partitions, spec, config, partitions.test_idx)
            facts = lumped_facts(nodes, spec)
            self.assertLess(facts["n_supra_nodes_lumped"], walker.N)
            G_red, v_red = build_lumped_matrices(nodes, edges, alpha=config.resolved_alpha(), influence_type=config.influence_type)
            layers = borrower_layer_positions(nodes)
            lumped_history = SzegedyWalk(G_red, v_red, layers[0], layers[1]).supra_node_history(TAU)
            ids, unfolded = unfold_class_scores(nodes, lumped_history.mean(axis=0))
            np.testing.assert_array_equal(ids, np.arange(partitions.frame.height))
            np.testing.assert_allclose(unfolded, run.omega, atol=1e-8)

            # Scores on the holdout partition, one shared bootstrap, a paired difference, a table.
            test_idx, y_test = partitions.test_idx, partitions.y_true[partitions.test_idx]
            classical = canonicalize(bundle.omega[test_idx], config.tie_tolerance)
            quantum = canonicalize(run.omega[test_idx], config.tie_tolerance)
            boot = bootstrap_indices(y_test, config.bootstrap_reps, config.master_seed)
            rows = []
            for name, scores in (("classical stationary", classical), ("quantum-inspired", quantum)):
                row = {"mechanism": name, **metrics_with_ci(scores, y_test, boot, k=K, decimals=config.tie_tolerance)}
                row.update(agreement_with(scores, classical, K, decimals=config.tie_tolerance, suffix="vs_classical"))
                rows.append(row)
            rep_q = bootstrap_replicates(quantum, y_test, boot, k=K, decimals=config.tie_tolerance)
            rep_c = bootstrap_replicates(classical, y_test, boot, k=K, decimals=config.tie_tolerance)
            diff = paired_difference(rep_q["pr_auc"], rep_c["pr_auc"], rows[1]["pr_auc"], rows[0]["pr_auc"])
            rows.append({"mechanism": "quantum minus classical (paired)", "pr_auc": diff["observed"],
                         "pr_auc_ci_lo": diff["ci_lo"], "pr_auc_ci_hi": diff["ci_hi"], "pr_auc_p_gt_0": diff["p_delta_pos"]})
            self.assertTrue(0.0 <= diff["p_delta_pos"] <= 1.0)
            self.assertAlmostEqual(rows[0]["spearman_vs_classical"], 1.0, places=12)
            self.assertEqual(rows[0]["normalized_l1_vs_classical"], 0.0)
            from unittest import mock

            results = root / "results"
            with mock.patch("paths.RESULTS_DIR", results):
                path = write_result("synthetic", "smoke_ladder", pl.DataFrame(rows, strict=False), stage="smoke",
                                    columns={"mechanism": "score", "pr_auc": "PR-AUC or its paired difference",
                                             "pr_auc_p_gt_0": "share of replicates with a positive difference"})
            self.assertEqual(path, results / "synthetic" / "smoke_ladder.csv")
            self.assertTrue(path.exists())
            self.assertEqual(pl.read_csv(path).height, 3)
            self.assertEqual(sorted(root.glob("results/synthetic/*")), [path])


class ConfigurationFingerprintTests(unittest.TestCase):
    def test_another_operator_or_the_smoke_configuration_never_shares_a_fingerprint(self):
        primary = HMEQ.primary
        a = config_fingerprint(DEFAULT_CONFIG, "04_quantum", network_spec=primary)
        self.assertNotEqual(a, config_fingerprint(DEFAULT_CONFIG.with_overrides(reduction_check_steps=5), "04_quantum", network_spec=primary))
        self.assertNotEqual(a, config_fingerprint(smoke_config(DEFAULT_CONFIG), "04_quantum", network_spec=primary))
        self.assertEqual(a, config_fingerprint(DEFAULT_CONFIG, "04_quantum", network_spec=primary))
        # Two datasets' caches never share a fingerprint: the network name keeps them apart.
        self.assertNotEqual(a, config_fingerprint(DEFAULT_CONFIG, "04_quantum", network_spec="NET_FMAC_2L"))
        # The operator digests and the masked frame's checksum enter the key through ``extra``.
        digests = {"transition_matrix": "t" * 64, "restart_matrix": "r" * 64, "personalization_vector": "v" * 64}
        with tempfile.TemporaryDirectory() as td:
            masked = Path(td) / "masked.parquet"
            masked.write_bytes(b"frame-a")
            keyed = config_fingerprint(DEFAULT_CONFIG, "04_quantum", network_spec=primary,
                                       extra=operator_cache_extra(digests, masked))
            self.assertNotEqual(keyed, a)
            masked.write_bytes(b"frame-b")
            self.assertNotEqual(keyed, config_fingerprint(DEFAULT_CONFIG, "04_quantum", network_spec=primary,
                                                          extra=operator_cache_extra(digests, masked)))
        with self.assertRaises(KeyError):
            operator_cache_extra({"transition_matrix": "t"}, masked)
        self.assertEqual(sorted(NETWORK_SPECS), ["NET_2L", "NET_JOB", "NET_REASON"])
        self.assertEqual(HMEQ.ladder, ("NET_JOB", "NET_REASON", "NET_2L"))

    def test_there_is_no_fixed_horizon_and_the_score_is_the_limit(self):
        for retired in ("tau", "tau_band", "tau_band_step", "tau_reference", "band_taus", "shots"):
            self.assertFalse(hasattr(DEFAULT_CONFIG, retired), retired)
        self.assertEqual(DEFAULT_CONFIG.quantum_score, "limit")
        self.assertEqual((DEFAULT_CONFIG.alpha, DEFAULT_CONFIG.stickiness, DEFAULT_CONFIG.split_seed), (0.85, 1.0, 42))
        self.assertEqual(DEFAULT_CONFIG.tau_max_for("hmeq"), 50_000)
        self.assertEqual(DEFAULT_CONFIG.tau_max_for("fmac"), 200_000)
        with self.assertRaises(KeyError):
            DEFAULT_CONFIG.tau_max_for("other")
        with self.assertRaises(ValueError):
            DEFAULT_CONFIG.with_overrides(quantum_score="tau_100")
        # Every seed derives from the master seed; distinct purposes never collide.
        seeds = {DEFAULT_CONFIG.seed_bootstrap(), DEFAULT_CONFIG.seed_folds(), DEFAULT_CONFIG.seed_placebo(0),
                 DEFAULT_CONFIG.seed_placebo(1), DEFAULT_CONFIG.seed_synthetic("S1", 0), DEFAULT_CONFIG.seed_aer("hmeq", 3, 2, "fake", 0)}
        self.assertEqual(len(seeds), 6)
        self.assertEqual(DEFAULT_CONFIG.seed_folds(), derive_seed(42, "folds"))
        self.assertNotEqual(DEFAULT_CONFIG.with_overrides(master_seed=7).seed_folds(), DEFAULT_CONFIG.seed_folds())
        self.assertNotEqual(DEFAULT_CONFIG.seed_noise_band("hmeq", 4), DEFAULT_CONFIG.seed_noise_band("hmeq", 6))
        # Nothing in the configuration names a device to submit to: the calibration snapshot is a fake backend.
        self.assertEqual(DEFAULT_CONFIG.ibm_backend, "FakeMarrakesh")
        self.assertFalse(any("live" in name for name in DEFAULT_CONFIG.as_dict()))


if __name__ == "__main__":
    unittest.main()
