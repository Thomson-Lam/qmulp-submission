"""Pytest bootstrapping: repository imports without installation, and the markers.

Replaces what ``[tool.pytest.ini_options]`` in the removed ``pyproject.toml``
provided. Nothing else belongs here.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
sys.path.insert(0, str(Path(ROOT) / "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MARKERS = (
    "slow: takes more than roughly five seconds; typically Aer simulation, transpilation, "
    "or a full-graph walk",
    "qiskit: requires qiskit and qiskit-aer",
    "ibm_runtime: requires qiskit-ibm-runtime (fake backends)",
)


def pytest_configure(config) -> None:
    """Register the markers the suite uses so ``--strict-markers`` accepts them."""
    for marker in MARKERS:
        config.addinivalue_line("markers", marker)
