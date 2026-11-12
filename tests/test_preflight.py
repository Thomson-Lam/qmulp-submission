"""The preflight of a clean rerun: ``outputs/`` may hold ``logs/`` and empty directories, nothing else.

Walk and cross-fit fingerprints cover the configuration and the operator, not the source code,
so the only way to exclude a stale checkpoint, shard, split or table is to start from an empty
tree. The first link of every batch chain therefore fails, with a listing, when ``outputs/``
holds a file outside ``logs/``. A directory tree with no file below it (what a listing of the
warm tasks or an earlier test run can leave behind) cannot hold a cache and does not count: the
smoke job that followed a test run once failed at its preflight for exactly that reason.
The module is standard library only, because it runs before anything else is importable.
"""

from __future__ import annotations

import ast
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import preflight


class Tree:
    """A throwaway repository root with the module's path constants pointed at it."""

    def __enter__(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        self.outputs = self.root / "outputs"
        self._patch = mock.patch.multiple(preflight, ROOT=self.root, OUTPUTS=self.outputs, LOGS=self.outputs / "logs")
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        self._td.cleanup()

    def touch(self, relative: str) -> None:
        path = self.outputs / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")

    def mkdir(self, relative: str) -> None:
        (self.outputs / relative).mkdir(parents=True, exist_ok=True)


def run(function):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = function()
    return code, out.getvalue(), err.getvalue()


class FreshTests(unittest.TestCase):
    def test_a_missing_outputs_directory_and_one_with_only_logs_are_a_clean_start(self):
        with Tree() as tree:
            self.assertEqual(run(preflight.fresh)[0], 0)
            tree.touch("logs/qmulp-smoke_1.out")
            tree.touch("logs/nested/qmulp-smoke_1.err")
            code, out, _ = run(preflight.fresh)
            self.assertEqual(code, 0)
            self.assertIn("clean start", out)

    def test_empty_directory_trees_do_not_count(self):
        with Tree() as tree:
            for relative in ("notebook_pipeline/hmeq/checkpoints", "notebook_pipeline/fmac", "results/hmeq", "manuscript/figures", "logs"):
                tree.mkdir(relative)
            self.assertEqual(preflight.offending_entries(), [])
            self.assertEqual(run(preflight.fresh)[0], 0)
            self.assertFalse(preflight.holds_a_file(tree.outputs / "notebook_pipeline"))

    def test_any_file_outside_logs_fails_and_is_listed_by_its_top_level_entry(self):
        for relative in ("frozen_split.json", "results/hmeq/t3_variants.csv", "notebook_pipeline/hmeq/checkpoints/a/completed.npz"):
            with Tree() as tree:
                tree.touch("logs/kept.out")
                tree.mkdir("manuscript")
                tree.touch(relative)
                code, out, err = run(preflight.fresh)
                self.assertEqual(code, 1, relative)
                self.assertEqual(out, "")
                top = relative.split("/")[0]
                listed = [line.split()[0] for line in err.splitlines() if line.startswith("  ")]
                self.assertEqual(listed, [f"outputs/{top}"], "logs/ and an empty directory are not offending")
                self.assertEqual([p.name for p in preflight.offending_entries()], [top])


class InventoryTests(unittest.TestCase):
    def test_the_inventory_never_fails_and_counts_what_a_stage_would_reuse(self):
        with Tree() as tree:
            code, out, _ = run(preflight.inventory)
            self.assertEqual(code, 0)
            self.assertIn("outputs/ does not exist", out)
            tree.touch("notebook_pipeline/hmeq/grid_shards/grid_0003.json")
            tree.touch("results/hmeq/t3_variants.csv")
            code, out, _ = run(preflight.inventory)
            self.assertEqual(code, 0)
            self.assertIn("hardware noise-grid shards (stage 06): 1 file(s)", out)
            self.assertIn("result tables: 1 file(s)", out)
            self.assertIn("quantum walk checkpoints (stage 04 reduction check): 0 file(s)", out)

    def test_fresh_and_inventory_exclude_each_other(self):
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                preflight.main(["--fresh", "--inventory"])


class StandardLibraryOnlyTests(unittest.TestCase):
    def test_the_module_imports_nothing_outside_the_standard_library(self):
        tree = ast.parse(Path(preflight.__file__).read_text())
        roots = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        roots |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module and n.level == 0}
        self.assertTrue(roots <= set(sys.stdlib_module_names) | {"__future__"}, roots - set(sys.stdlib_module_names))


if __name__ == "__main__":
    unittest.main()
