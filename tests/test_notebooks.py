"""One notebook per stage, parameterised on the dataset, and the driver that sets the parameter.

Every stage notebook from stage 02 on carries one cell tagged ``parameters`` that holds
``DATASET``; ``scripts.run_all`` replaces that cell in memory for each dataset and writes the
executed copy to ``outputs/executed_notebooks/<stage>_<dataset>.ipynb``, so the tracked notebooks
stay stripped. The data stage keeps one notebook per dataset (the cleaning differs) and the
Freddie Mac exploration is its own notebook. No notebook carries an output, an execution count,
execution metadata, an absolute path or a string that names a machine or an account.
"""

from __future__ import annotations

import json
import re
import unittest

from datasets import DATASET_NAMES
from paths import ROOT
from scripts import run_all

try:
    import nbformat
except ImportError:  # pragma: no cover - nbformat ships with nbclient, which run_all needs
    nbformat = None

NOTEBOOKS = ROOT / "notebooks"
KEPT_PER_DATASET = {"00_fmac_eda.ipynb", "01_fmac_cleaning.ipynb", "01_hmeq_cleaning.ipynb"}
IDENTIFYING = re.compile("|".join(("lus" "tre", "tgati" "cag", "/ho" "me/", "/Us" "ers/", "def" "-cbravo", "6040" "457")))


def load(name: str) -> dict:
    return json.loads((NOTEBOOKS / name).read_text())


class StageNotebookTests(unittest.TestCase):
    def test_there_is_one_notebook_per_stage_and_none_per_dataset_beyond_the_data_stage(self):
        self.assertEqual({p.name for p in NOTEBOOKS.glob("*.ipynb")}, KEPT_PER_DATASET | set(run_all.STAGE_NOTEBOOKS.values()))
        for stage, name in run_all.STAGE_NOTEBOOKS.items():
            for dataset in DATASET_NAMES:
                self.assertEqual(run_all.notebook_for(stage, dataset), name)
        self.assertEqual(run_all.notebook_for("01", "hmeq"), "01_hmeq_cleaning.ipynb")
        self.assertEqual(run_all.notebook_for("01", "fmac"), "01_fmac_cleaning.ipynb")
        for stage in run_all.STAGES:
            for dataset in DATASET_NAMES:
                if stage in run_all.STAGE_NOTEBOOKS or stage == "01":
                    self.assertTrue((NOTEBOOKS / run_all.notebook_for(stage, dataset)).is_file(), (stage, dataset))

    def test_each_stage_notebook_sets_the_dataset_in_one_tagged_cell_before_anything_reads_it(self):
        for name in run_all.STAGE_NOTEBOOKS.values():
            cells = load(name)["cells"]
            tagged = [i for i, c in enumerate(cells) if run_all.PARAMETERS_TAG in c["metadata"].get("tags", [])]
            self.assertEqual(len(tagged), 1, name)
            self.assertEqual(cells[tagged[0]]["cell_type"], "code", name)
            self.assertIn('DATASET = "hmeq"\n', "".join(cells[tagged[0]]["source"]), name)
            readers = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and i != tagged[0] and "DATASET" in "".join(c["source"])]
            self.assertTrue(readers and min(readers) > tagged[0], f"{name}: DATASET is read before it is set")
            for i in readers:
                self.assertIsNone(re.search(r"^DATASET\s*=", "".join(cells[i]["source"]), re.MULTILINE), f"{name}: DATASET is assigned twice")

    def test_every_notebook_is_stripped_and_names_neither_a_machine_nor_an_account(self):
        for path in sorted(NOTEBOOKS.glob("*.ipynb")):
            nb = json.loads(path.read_text())
            for cell in nb["cells"]:
                self.assertEqual(cell.get("outputs", []), [], path.name)
                self.assertIsNone(cell.get("execution_count"), path.name)
                self.assertNotIn("execution", cell["metadata"], path.name)
            self.assertIsNone(IDENTIFYING.search(path.read_text()), path.name)

    def test_the_stage_notebooks_call_the_stage_drivers_and_hold_no_stage_logic(self):
        drivers = {"02_network.ipynb": "network", "03_classical.ipynb": "classical", "04_quantum.ipynb": "quantum",
                   "05_evaluation.ipynb": "evaluation", "05a_mechanism.ipynb": "mechanism", "06_hardware.ipynb": "hardware"}
        for name, driver in drivers.items():
            code = "\n".join("".join(c["source"]) for c in load(name)["cells"] if c["cell_type"] == "code")
            self.assertIn(f"from stages import {driver}", code, name)
            self.assertIn(f"{driver}.run(DATASET, config)", code, name)
            self.assertLess(code.count("\n"), 60, f"{name} is thin: a call, a display and an artifact list")


@unittest.skipIf(nbformat is None, "nbformat is not installed")
class DriverTests(unittest.TestCase):
    def test_the_parameter_cell_is_replaced_in_memory_and_the_source_notebook_is_untouched(self):
        before = (NOTEBOOKS / "02_network.ipynb").read_text()
        nb = nbformat.read(NOTEBOOKS / "02_network.ipynb", as_version=4)
        self.assertTrue(run_all.set_dataset(nb, "fmac"))
        cell = next(c for c in nb.cells if run_all.PARAMETERS_TAG in c.metadata.get("tags", []))
        self.assertEqual(cell.source, 'DATASET = "fmac"\n')
        self.assertEqual((NOTEBOOKS / "02_network.ipynb").read_text(), before)

    def test_a_notebook_without_the_cell_is_reported_and_a_bad_stage_range_is_refused(self):
        nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("x = 1")])
        self.assertFalse(run_all.set_dataset(nb, "hmeq"))
        with self.assertRaises(SystemExit):
            run_all.main(["--from-stage", "06", "--to-stage", "02"])


if __name__ == "__main__":
    unittest.main()
