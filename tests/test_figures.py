"""The figure style: every figure is a 6.5 in wide vector PDF with embedded TrueType fonts and a palette that prints.

A trivial two-panel figure is drawn through the style of ``scripts/make_figures.py`` and the saved file is
read back. The page width comes from the PDF itself (``make_figures.save`` raises unless it is 6.5 in, whatever
the contents), the fonts must be embedded as TrueType (Type 42, no Type 3 glyph procedures), and no
palette colour may be as light as the yellow the audit flagged. Every label a figure draws is a word from
``evaluation.formatting`` (no ``NET_*`` or ``*_like`` identifier, no ``S1(a)``), and the per-step noise figures
draw the instance sizes the script names.
"""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from evaluation.formatting import ARM_LABELS, DATASET_LABELS, NETWORK_LABELS, SYNTHETIC_LABELS
from paths import ROOT
from scripts import make_figures


def _luminance(colour: str) -> float:
    r, g, b = (int(colour[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


class StyleTests(unittest.TestCase):
    def test_a_saved_figure_is_a_6p5_inch_vector_pdf_with_embedded_truetype_fonts(self):
        make_figures.apply_style()
        fig, axes = make_figures.figure(2)
        axes[0, 0].plot([1, 10, 100], [0.1, 0.5, 0.9], marker="o")
        axes[0, 0].set_xscale("log")
        axes[0, 0].set_xlabel(r"$\tau$ and $\rho$")
        axes[0, 1].plot([0, 1], [1, 0])
        make_figures.panel_letter(axes[0, 0], "a")
        make_figures.panel_letter(axes[0, 1], "b")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "figure.pdf"
            width = make_figures.save(fig, path)
            data = path.read_bytes()
        self.assertIsNotNone(width, "the page width could not be read back from the PDF")
        self.assertAlmostEqual(width, make_figures.FULL_WIDTH_IN, places=3)
        self.assertTrue(data.startswith(b"%PDF"))
        self.assertIn(b"/FontFile2", data)
        self.assertNotIn(b"/Type3", data)

    def test_the_palette_has_no_colour_too_light_to_print_and_every_series_has_a_marker_and_a_line_style(self):
        self.assertLess(max(_luminance(c) for c in make_figures.PALETTE), 0.8)
        self.assertEqual(len(make_figures.PALETTE), len(make_figures.MARKERS))
        self.assertEqual(len(make_figures.PALETTE), len(make_figures.LINESTYLES))         # zip() would silently drop series otherwise
        self.assertEqual(len(set(make_figures.LINESTYLES)), 4)
        self.assertEqual(len(set(make_figures.PALETTE)), len(make_figures.PALETTE))


class LabelTests(unittest.TestCase):
    def test_every_label_a_figure_draws_is_a_word_from_the_one_mapping(self):
        text = (ROOT / "scripts" / "make_figures.py").read_text()
        self.assertIsNone(re.search(r"NET_|_like\b|S1\(a\)", text), "the figure script holds no code identifier and no 'S1(a)'")
        self.assertEqual(dict(make_figures.DATASETS), DATASET_LABELS)
        codes = [*NETWORK_LABELS, *SYNTHETIC_LABELS, *make_figures.NOISE_ARMS]
        for code in codes:
            label = {**NETWORK_LABELS, **SYNTHETIC_LABELS, **ARM_LABELS}[code]
            self.assertNotRegex(label, r"_|NET", f"{code} would be drawn as a code identifier")
        self.assertEqual(make_figures.K_SUBSET, (2, 3, 4, 5, 6, 8, 12), "the per-step noise figures draw these instance sizes")
        self.assertLessEqual(len(make_figures.K_SUBSET), len(make_figures.PALETTE), "one colour per instance size, in a legend")


if __name__ == "__main__":
    unittest.main()
