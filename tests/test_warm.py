"""The array worker: one task list, derived not hard-coded, indexed by the sbatch ``--array`` bounds.

``scripts/warm.py`` is a dispatcher over functions that already exist. These
tests pin what makes it safe to run concurrently: a deterministic task order,
one distinct output path per task, ``--array`` upper bounds that equal the
task count minus one, a ``--list`` mode that computes nothing, and a smoke
task list that can never be mistaken for the full one.

Stage 04 plans one full-operator walk per ladder network, to the
reduction-check horizon only: the reported score is the exact limit on the
lumped operator, and the fold and placebo operators are scored by their lumped
limits, so no fold or placebo walk exists any more.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

from config import DEFAULT_CONFIG
from fingerprint import TEST_OPERATOR_TAG, smoke_config
from hmeq.config import DATASET as HMEQ
from quantum.hardware.class_candidates import sweep_sizes
from quantum.inspired.walk_cache import planned_walks as _planned_walks, walk_fingerprint
from scripts.warm import tasks_for_stage

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
LADDER_NETWORKS, PRIMARY_NETWORK = HMEQ.ladder, HMEQ.primary
N_WALKS = len(LADDER_NETWORKS)


def planned_walks(config):
    """The HMEQ plan, which is what the un-suffixed array scripts index."""
    return _planned_walks(config, ladder=LADDER_NETWORKS, primary=PRIMARY_NETWORK)


def _array_upper_bound(sbatch: Path) -> int:
    match = re.search(r"^#SBATCH --array=0-(\d+)$", sbatch.read_text(), re.MULTILINE)
    assert match, f"{sbatch} has no '#SBATCH --array=0-N' line"
    return int(match.group(1))


def _cpus(sbatch: Path) -> int:
    return int(re.search(r"^#SBATCH --cpus-per-task=(\d+)$", sbatch.read_text(), re.MULTILINE).group(1))


class Stage04TaskListTests(unittest.TestCase):
    def test_one_reduction_check_walk_per_network_in_ladder_order(self):
        tasks = tasks_for_stage("04", DEFAULT_CONFIG)
        self.assertEqual(len(tasks), N_WALKS)
        self.assertEqual([t.index for t in tasks], list(range(N_WALKS)))
        steps = int(DEFAULT_CONFIG.reduction_check_steps)
        expected = [("test_operator", n, TEST_OPERATOR_TAG, steps) for n in LADDER_NETWORKS]
        self.assertEqual([(t.kind, t.network, t.tag, t.tau) for t in tasks], expected)
        self.assertEqual(tasks[-1].network, "NET_2L")
        # No fold or placebo walk is planned: those operators are scored by their lumped limits.
        self.assertEqual({t.kind for t in tasks}, {"test_operator"})

    def test_the_list_is_the_stage_modules_plan(self):
        tasks = tasks_for_stage("04", DEFAULT_CONFIG)
        plan = planned_walks(DEFAULT_CONFIG)
        self.assertEqual([(t.network, t.tag, t.tau) for t in tasks], [(w["network"], w["tag"], w["tau"]) for w in plan])

    def test_no_two_tasks_write_the_same_path(self):
        for stage in ("04", "06"):
            paths = [t.path for t in tasks_for_stage(stage, DEFAULT_CONFIG, n_full=54)]
            self.assertEqual(len(paths), len(set(paths)), stage)
            for path in paths:
                self.assertIn("outputs/notebook_pipeline", path.as_posix())
                self.assertNotIn("manifest", path.name)

    def test_smoke_task_list_never_shares_a_fingerprint_with_the_full_one(self):
        smoke = smoke_config(DEFAULT_CONFIG)
        self.assertEqual(len(tasks_for_stage("04", smoke)), N_WALKS)
        for w in planned_walks(smoke):
            self.assertNotEqual(
                walk_fingerprint(smoke, w["network"], w["tag"], w["tau"]),
                walk_fingerprint(DEFAULT_CONFIG, w["network"], w["tag"], DEFAULT_CONFIG.reduction_check_steps),
            )

    def test_each_walk_has_its_own_fingerprint(self):
        keys = {walk_fingerprint(DEFAULT_CONFIG, w["network"], w["tag"], w["tau"]) for w in planned_walks(DEFAULT_CONFIG)}
        self.assertEqual(len(keys), N_WALKS)


class Stage06TaskListTests(unittest.TestCase):
    def test_one_task_per_sweep_size(self):
        tasks = tasks_for_stage("06", DEFAULT_CONFIG, n_full=54)
        sizes = sweep_sizes(DEFAULT_CONFIG, 54)
        self.assertEqual(len(tasks), len(sizes))
        self.assertEqual(len(tasks), 23)
        self.assertEqual(tasks[-1].n_classes, 2)   # the ladder ends at the smallest instance with a reference class
        self.assertEqual([t.n_classes for t in tasks], sizes)
        self.assertEqual(tasks[0].tag, "full")
        self.assertTrue(all(t.path.name == f"grid_{t.n_classes}.json" for t in tasks))
        self.assertTrue(all(t.path.parent.name == "grid_shards" for t in tasks))

    def test_a_smaller_operator_gives_a_shorter_list(self):
        self.assertEqual(len(tasks_for_stage("06", DEFAULT_CONFIG, n_full=40)), len(sweep_sizes(DEFAULT_CONFIG, 40)))
        self.assertEqual(len(tasks_for_stage("06", smoke_config(DEFAULT_CONFIG), n_full=54)), 3)


@unittest.skipUnless((SCRIPTS / "warm04_walks.sbatch").exists(), "scripts/*.sbatch are local and untracked; a clone without them skips these checks")
class SbatchTests(unittest.TestCase):
    def test_array_bounds_match_the_task_lists(self):
        self.assertEqual(_array_upper_bound(SCRIPTS / "warm04_walks.sbatch"), len(tasks_for_stage("04", DEFAULT_CONFIG)) - 1)
        self.assertEqual(_array_upper_bound(SCRIPTS / "warm06_sweep.sbatch"), len(tasks_for_stage("06", DEFAULT_CONFIG)) - 1)

    def test_cpus_per_task_are_2_except_where_aer_runs(self):
        for name in ("nb03_classical", "warm04_walks"):
            self.assertEqual(_cpus(SCRIPTS / f"{name}.sbatch"), 2, name)
        # The grid worker runs three Aer arms per instance.
        self.assertEqual(_cpus(SCRIPTS / "warm06_sweep.sbatch"), 4)
        self.assertEqual(_cpus(SCRIPTS / "warm06_fmac_sweep.sbatch"), 4)
        self.assertEqual(_cpus(SCRIPTS / "nb06_hardware.sbatch"), 8)

    def test_the_array_files_dispatch_to_the_worker(self):
        for stage, name in (("04", "warm04_walks"), ("06", "warm06_sweep")):
            text = (SCRIPTS / f"{name}.sbatch").read_text()
            self.assertIn(f'python -m scripts.warm --stage {stage} --task-id "$SLURM_ARRAY_TASK_ID"', text)
            self.assertIn("set -euo pipefail", text)
            self.assertIn("mkdir -p outputs/logs", text)
            self.assertIn("OMP_NUM_THREADS=1", text)


class ListModeTests(unittest.TestCase):
    def test_the_worker_is_thin(self):
        lines = [l for l in (SCRIPTS / "warm.py").read_text().splitlines() if l.strip() and not l.strip().startswith("#")]
        self.assertLess(len(lines), 150)

    def test_list_prints_the_table_and_writes_nothing(self):
        env = {k: v for k, v in os.environ.items() if k != "QMULP_SMOKE"}
        env["PYTHONPATH"] = os.pathsep.join((str(REPO / "src"), str(REPO)))
        outputs = REPO / "results" / "outputs"

        def snapshot() -> set[Path]:
            """Every file and directory under outputs/ except logs/ (a job's own log grows while the suite runs)."""
            logs = outputs / "logs"
            return {p for p in outputs.rglob("*") if p != logs and logs not in p.parents} if outputs.exists() else set()

        before = snapshot()
        for stage, n in (("04", N_WALKS), ("06", None)):
            result = subprocess.run([sys.executable, "-m", "scripts.warm", "--stage", stage, "--list"],
                                    cwd=REPO, env=env, capture_output=True, text=True, check=True)
            rows = [l for l in result.stdout.splitlines() if re.match(r"^\s*\d+\s+\w+\s+NET_", l)]
            if n is not None:
                self.assertEqual(len(rows), n, result.stdout)
            self.assertIn(f"sbatch --array=0-{len(rows) - 1}", result.stdout)
            print(result.stdout)
        self.assertEqual(before, snapshot(), "--list must not create a file or a directory")

    def test_an_out_of_range_task_id_is_refused(self):
        env = {k: v for k, v in os.environ.items() if k != "QMULP_SMOKE"}
        env["PYTHONPATH"] = os.pathsep.join((str(REPO / "src"), str(REPO)))
        result = subprocess.run([sys.executable, "-m", "scripts.warm", "--stage", "04", "--task-id", "99"],
                                cwd=REPO, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn(f"outside 0..{N_WALKS - 1}", result.stderr)


if __name__ == "__main__":
    unittest.main()
