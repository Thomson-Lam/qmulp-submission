"""Every table the specs build fits the page: at most 6.5 in wide, set at 10 pt with ``\\tabcolsep`` 4 pt.

When ``pdflatex`` is on the path, every table that ``scripts.make_manuscript_tables.assemble`` builds from the result CSVs
under ``outputs/results`` is set in a box of one small document (10 pt, booktabs, the maths packages the tables use,
``\\tabcolsep`` 4 pt) and the box widths are read back from the log; a table wider than the 6.5 in text width of the
journal class fails. The tables are rendered by the code under test, in memory, from the real results: the test measures
what the current specs and builders make of them, not a ``.tex`` file an earlier run left under ``outputs/manuscript/``
(whose layout may be older than the code), and it needs no hand-off to have run. It skips when ``pdflatex``, one of its
packages or the result CSVs are absent. The static estimate that guards the same limit while the specs are written is in
``docs/internal/revision_log.txt``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from paths import RESULTS_DIR
from scripts.make_manuscript_tables import assemble

TEXT_WIDTH_PT = 6.5 * 72.27  # the text width of the journal class, in TeX points
WIDTH = re.compile(r"QPPR-WIDTH (\d+) ([0-9.]+)pt")
PACKAGES = ("booktabs", "amsmath", "amssymb")


def _have_latex() -> bool:
    if shutil.which("pdflatex") is None or shutil.which("kpsewhich") is None:
        return False
    return all(subprocess.run(["kpsewhich", f"{package}.sty"], capture_output=True, text=True).stdout.strip() for package in PACKAGES)


def table_widths(tables: dict[str, str], work: Path) -> dict[str, float]:
    """The width in points of each table set in its own box at 10 pt with ``\\tabcolsep`` 4 pt, from one ``pdflatex`` run."""
    names = sorted(tables)
    body = []
    for index, name in enumerate(names):
        (work / f"t{index}.tex").write_text(tables[name])
        body.append(rf"\sbox{{\tablebox}}{{\input{{t{index}.tex}}}}\typeout{{QPPR-WIDTH {index} \the\wd\tablebox}}")
    document = "\n".join([
        r"\documentclass[10pt]{article}",
        *[rf"\usepackage{{{package}}}" for package in PACKAGES],
        r"\newsavebox{\tablebox}",
        r"\begin{document}",
        r"\setlength{\tabcolsep}{4pt}",
        *body,
        r"\end{document}",
    ])
    (work / "widths.tex").write_text(document)
    run = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "widths.tex"], cwd=work, capture_output=True, text=True, timeout=600)
    found = {int(i): float(width) for i, width in WIDTH.findall(run.stdout)}
    if len(found) != len(names):
        raise AssertionError(f"pdflatex measured {len(found)} of {len(names)} tables (exit {run.returncode}):\n{run.stdout[-1500:]}")
    return {names[i]: width for i, width in found.items()}


class TableWidthTests(unittest.TestCase):
    @unittest.skipUnless(RESULTS_DIR.exists() and any(RESULTS_DIR.glob("*/*.csv")), "no result CSVs under outputs/results yet")
    @unittest.skipUnless(_have_latex(), "pdflatex (with booktabs and the AMS packages) is not available")
    def test_every_table_is_at_most_6p5_inches_wide(self):
        tables, _skipped = assemble()
        if not tables:
            self.skipTest("no table could be built from the result CSVs")
        with tempfile.TemporaryDirectory() as tmp:
            widths = table_widths(tables, Path(tmp))
        too_wide = {name: round(width / 72.27, 2) for name, width in widths.items() if width > TEXT_WIDTH_PT}
        self.assertEqual(too_wide, {}, f"tables wider than 6.5 in (inches): {too_wide}")


if __name__ == "__main__":
    unittest.main()
