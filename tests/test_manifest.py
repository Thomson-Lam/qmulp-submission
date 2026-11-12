"""The manifest: what it records, and what it never records.

A manifest names the commit, the dirty flag, ``source_sha256``, the versions, the digests of
the inputs and outputs, the configuration with its hash (the dataset configuration included),
the seeds and the class of machine. It never names a host, a user, a scheduler job or an
absolute path. The same scan runs over the manifests under ``outputs/`` when a rerun has
written them, and over every text file git lists for the tree for the strings that identify
a person, a machine or an account (the repository goes to double-anonymous review).
"""

from __future__ import annotations

import json
import re
import shutil
import socket
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import polars as pl

from config import DEFAULT_CONFIG
from evaluation import provenance
from evaluation.provenance import write_result
from fmac.config import DATASET as FMAC
from hmeq.config import DATASET as HMEQ
from paths import OUTPUTS_DIR, ROOT

# What identifies a person, a machine, an account or a job, written in pieces so that this file does not match itself.
IDENTIFYING = ("lus" "tre", "tgati" "cag", "/ho" "me/", "/Us" "ers/", "nar" "val", "def" "-cbravo", "6040" "457", "uwo" r"\.ca")
EMAIL = r"[A-Za-z0-9._+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}"
HOST_AND_JOB = ("host" "name", "SLURM" "_", "slurm" "_")
PATH_ROOTS = ("ho" "me", "lus" "tre", "project", "scratch", "Us" "ers", "tmp", "var", "cvmfs", "usr", "opt")
ABSOLUTE_PATH = re.compile("\"/(?:" + "|".join(PATH_ROOTS) + ")/")
GIT = shutil.which("git")


def assert_portable(test: unittest.TestCase, text: str, label: str, *, email: bool = True) -> None:
    """``text`` names no person, machine, account, host, job or absolute path (``email=False`` for LaTeX macro names, which contain ``@``)."""
    for pattern in (*IDENTIFYING, *((EMAIL,) if email else ()), *(re.escape(p) for p in HOST_AND_JOB)):
        found = re.search(pattern, text)
        test.assertIsNone(found, f"{label} carries {found.group(0)!r}" if found else "")
    test.assertIsNone(ABSOLUTE_PATH.search(text), f"{label} carries an absolute path")


class ManifestContentTests(unittest.TestCase):
    def _write(self, dataset, **kwargs):
        with tempfile.TemporaryDirectory() as td:
            path = provenance.write_manifest(Path(td) / "stage", "99_unit", DEFAULT_CONFIG, dataset=dataset,
                                           inputs=[ROOT / "src" / "paths.py"], outputs=[Path(td) / "absent.csv"], seeds={"master_seed": 1}, **kwargs)
            return path.read_text()

    def test_it_records_the_commit_the_source_digest_and_the_hash_of_configuration_and_dataset(self):
        text = self._write(HMEQ)
        data = json.loads(text)
        self.assertEqual(data["dataset"], "hmeq")
        self.assertRegex(data["config_hash"], r"^[0-9a-f]{64}$")
        self.assertEqual(data["config_hash"], provenance.config_hash(DEFAULT_CONFIG, HMEQ))
        self.assertNotEqual(data["config_hash"], provenance.config_hash(DEFAULT_CONFIG, FMAC))
        self.assertNotEqual(data["config_hash"], provenance.config_hash(DEFAULT_CONFIG.with_overrides(bootstrap_reps=7), HMEQ))
        self.assertIn("commit", data["git"])
        self.assertIn("dirty", data["git"])
        self.assertTrue(data["source_sha256"] is None or re.fullmatch(r"[0-9a-f]{64}", data["source_sha256"]))
        self.assertEqual(data["inputs"]["src/paths.py"]["status"], "ok")
        self.assertEqual(data["outputs"]["<outside the repository>/absent.csv"]["status"], "missing")
        self.assertIn("python", data["versions"])
        self.assertIn("qiskit-ibm-runtime", data["versions"])

    def test_it_never_records_a_host_a_user_a_job_or_an_absolute_path(self):
        text = self._write(FMAC)
        data = json.loads(text)
        self.assertEqual(set(data["environment"]), {"cpu", "physical_cores", "logical_cores", "cores_available", "ram_gb", "os",
                                                    "omp_num_threads", "smoke"})
        host = socket.gethostname()
        if len(host) > 3:
            self.assertNotIn(host, text)
        assert_portable(self, text, "a fresh manifest")

    def test_the_machine_is_a_class_of_node_not_a_node(self):
        facts = provenance.machine()
        self.assertEqual(set(facts), {"cpu", "physical_cores", "logical_cores", "cores_available", "ram_gb", "os"})
        self.assertIsInstance(facts["cpu"], str)
        self.assertGreaterEqual(facts["logical_cores"], 1)
        assert_portable(self, json.dumps(facts), "the machine description")

    def test_it_carries_the_tables_the_stage_registered_with_their_column_definitions(self):
        frame = pl.DataFrame({"a": [1.0]})
        with tempfile.TemporaryDirectory() as td:
            with mock.patch("paths.RESULTS_DIR", Path(td) / "results"):
                write_result("synthetic", "unit_manifest_table", frame, stage="99_unit", columns={"a": "a value"})
            data = json.loads(provenance.write_manifest(Path(td) / "stage", "99_unit", DEFAULT_CONFIG).read_text())
        provenance._REGISTRY.pop("99_unit", None)
        self.assertEqual(data["tables"]["unit_manifest_table"]["columns"], {"a": "a value"})
        self.assertEqual(data["tables"]["unit_manifest_table"]["scope"], "synthetic")

    def test_a_path_outside_the_repository_is_recorded_by_name_only(self):
        self.assertEqual(provenance.recorded_path(ROOT / "config.py"), "config.py")
        self.assertEqual(provenance.recorded_path(Path(tempfile.gettempdir()) / "somewhere" / "x.csv"), "<outside the repository>/x.csv")


@unittest.skipUnless(GIT, "git is not available")
class SourceDigestTests(unittest.TestCase):
    def test_it_hashes_the_files_git_lists_and_nothing_else(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            (root / ".gitignore").write_text("ignored.txt\n")
            (root / "a.py").write_text("x = 1\n")
            (root / "ignored.txt").write_text("1")
            digests = []
            with mock.patch("evaluation.provenance.repo_root", return_value=root):
                def digest() -> str:
                    provenance.source_sha256.cache_clear()
                    return provenance.source_sha256()

                digests.append(digest())
                self.assertRegex(digests[0], r"^[0-9a-f]{64}$")
                (root / "ignored.txt").write_text("2")
                self.assertEqual(digest(), digests[0], "an ignored file is not part of the code")
                (root / "a.py").write_text("x = 2\n")
                digests.append(digest())
                (root / "b.py").write_text("y = 3\n")
                digests.append(digest())
                (root / "b.py").unlink()
                digests.append(digest())
            provenance.source_sha256.cache_clear()
            self.assertEqual(len(set(digests)), 3, "an edit, a new file and its removal each change the digest")
            self.assertEqual(digests[3], digests[1], "the digest depends on the content of the tree, not on its history")


class WrittenManifestTests(unittest.TestCase):
    """The manifests of a rerun, when there are any: one run, one code, no identifying string."""

    def setUp(self):
        self.paths = sorted((OUTPUTS_DIR / "notebook_pipeline").glob("*/manifest.json")) if (OUTPUTS_DIR / "notebook_pipeline").exists() else []
        if not self.paths:
            self.skipTest("no stage manifests under outputs/ yet")

    def test_every_manifest_is_portable_and_they_describe_one_source_tree(self):
        sources, commits, hashes = set(), set(), set()
        for path in self.paths:
            text = path.read_text()
            assert_portable(self, text, path.parent.name)
            data = json.loads(text)
            sources.add(data.get("source_sha256"))
            commits.add(data["git"]["commit"])
            hashes.add(bool(re.fullmatch(r"[0-9a-f]{64}", data["config_hash"])))
        self.assertEqual(len(sources), 1, f"the manifests of one rerun disagree on source_sha256: {sources}")
        self.assertEqual(len(commits), 1, f"the manifests of one rerun disagree on the commit: {commits}")
        self.assertEqual(hashes, {True})

    def test_results_and_numbers_carry_no_identifying_string_either(self):
        for path in sorted((OUTPUTS_DIR / "results").glob("*/*.csv")):
            assert_portable(self, path.read_text().splitlines()[0] if path.stat().st_size else "", f"the header of {path.name}")
        numbers = OUTPUTS_DIR / "manuscript" / "numbers.tex"
        if numbers.exists():
            assert_portable(self, numbers.read_text(), "numbers.tex", email=False)


@unittest.skipUnless(GIT, "git is not available")
class AnonymityTests(unittest.TestCase):
    def test_no_text_file_of_the_tree_carries_an_identifying_string(self):
        listing = subprocess.run(["git", "ls-files", "-z", "-co", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True, check=False)
        if listing.returncode != 0:
            self.skipTest("git could not list the tree")
        offenders = []
        for name in sorted(n for n in listing.stdout.split("\0") if n):
            path = ROOT / name
            if not path.is_file():
                continue
            try:
                text = path.read_bytes().decode("utf-8")
            except UnicodeDecodeError:
                continue                                   # a binary file (the reference PDFs): not scanned here
            for number, line in enumerate(text.splitlines(), 1):
                if any(re.search(pattern, line) for pattern in (*IDENTIFYING, EMAIL)):
                    offenders.append(f"{name}:{number}")
        self.assertEqual(offenders, [], "identifying strings in: " + ", ".join(offenders[:10]))


if __name__ == "__main__":
    unittest.main()
