"""Every path is anchored to the repository root, never to the working directory.

``nbconvert`` executes the notebooks from ``notebooks/``; a path resolved
against the working directory would land there.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

import paths

REPO = Path(__file__).resolve().parents[1]


class PathAnchoringTests(unittest.TestCase):
    def test_root_is_the_directory_holding_paths_py(self):
        self.assertEqual(paths.ROOT, REPO)
        self.assertTrue((paths.ROOT / "src" / "paths.py").exists())
        self.assertEqual(paths.repo_root(), REPO)
        with self.assertRaises(TypeError):
            paths.repo_root("somewhere/else")                    # the root does not depend on where the caller is

    def test_every_derived_path_hangs_off_the_root(self):
        self.assertEqual(paths.DATA_DIR, REPO / "data")
        self.assertEqual(paths.OUTPUTS_DIR, REPO / "results" / "outputs")
        self.assertEqual(paths.PIPELINE_DIR, REPO / "results" / "outputs" / "notebook_pipeline")
        self.assertEqual(paths.RESULTS_DIR, REPO / "results" / "outputs" / "results")
        self.assertEqual(paths.MANUSCRIPT_DIR, REPO / "results" / "outputs" / "manuscript")
        self.assertEqual(paths.result_path("fmac", "T1_data"), REPO / "results" / "outputs" / "results" / "fmac" / "T1_data.csv")
        with self.assertRaises(ValueError):
            paths.result_path("tables", "T1_data")
        self.assertFalse(hasattr(paths, "TABLES_DIR"))
        self.assertEqual(paths.repo_path("data/hmeq.csv"), REPO / "data" / "hmeq.csv")
        self.assertEqual(paths.repo_path("/abs/path.csv"), Path("/abs/path.csv"))
        self.assertEqual(paths.stage_path("04_quantum"), REPO / "results" / "outputs" / "notebook_pipeline" / "04_quantum")
        self.assertFalse(paths.stage_path("a_stage_that_was_never_created").exists(), "stage_path computes a path and never creates it")

    def test_resolution_is_identical_from_the_root_and_from_a_subdirectory(self):
        before = os.getcwd()
        try:
            os.chdir(REPO)
            from_root = (
                paths.DATA_DIR,
                paths.repo_path("data/hmeq.csv"),
                paths.relative_to_repo(REPO / "data" / "hmeq.csv"),
            )
            os.chdir(REPO / "notebooks")
            from_sub = (
                paths.DATA_DIR,
                paths.repo_path("data/hmeq.csv"),
                paths.relative_to_repo(REPO / "data" / "hmeq.csv"),
            )
        finally:
            os.chdir(before)
        self.assertEqual(from_root, from_sub)
        self.assertEqual(from_sub[2], "data/hmeq.csv")

    def test_a_fresh_interpreter_started_in_notebooks_agrees(self):
        """The import-time constants do not depend on the process working directory."""
        env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(REPO / "src"), str(REPO)))}
        command = [
            sys.executable,
            "-c",
            "import paths; print(paths.DATA_DIR); print(paths.repo_path('outputs/frozen_split.json'))",
        ]
        from_root = subprocess.run(
            command, cwd=REPO, env=env, capture_output=True, text=True, check=True
        ).stdout
        from_sub = subprocess.run(
            command, cwd=REPO / "notebooks", env=env, capture_output=True, text=True, check=True
        ).stdout
        self.assertEqual(from_root, from_sub)
        self.assertEqual(from_root.splitlines()[0], str(REPO / "data"))

    def test_relative_to_repo_reports_outside_paths_truthfully(self):
        self.assertEqual(
            paths.relative_to_repo("/definitely/outside.csv"), "/definitely/outside.csv"
        )
        with self.assertRaises(AssertionError):
            paths.assert_no_absolute_paths(["/abs"], context="unit")


if __name__ == "__main__":
    unittest.main()
