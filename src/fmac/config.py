"""What Freddie Mac is: two discovered layers, three networks, one temporal split.

The layers are ``PostalCode`` (the three-digit ZIP prefix) and ``SellerName``
(the originator). Both are "borrowers who share this recorded attribute"
relations, which is what a MuLP layer is; neither is a quantile bin of a
continuous risk driver. Their states are data -- 24 postal prefixes and 19
sellers in the development cohort -- so they are discovered from the processed
frame rather than listed here, and :data:`EXPECTED_LAYER_STATES` is the
contract that keeps that discovery honest.

The split is the calendar: the loans whose first payment fell in March 2023 are
the development partition and those with a first payment in March 2024 the
test partition. Nothing draws it and no seed reproduces it. The extract and its
``target`` column are used exactly as supplied to the authors; nothing here
downloads, rebuilds or redefines them.
"""

from __future__ import annotations

from pathlib import Path

from datasets import DatasetConfig
from paths import DATA_DIR

NET_FMAC_POSTAL = "NET_FMAC_POSTAL"
NET_FMAC_SELLER = "NET_FMAC_SELLER"
NET_FMAC_2L = "NET_FMAC_2L"

DEVELOPMENT_COHORT = "2023"
TEST_COHORT = "2024"

# The per-cohort encoded frames: the tabular view of the same rows, kept
# because a borrower-level model reads them directly.
TRAIN_PROCESSED = "data/fmac_train_processed.parquet"
TEST_PROCESSED = "data/fmac_test_processed.parquet"

# The raw cohorts, keyed by the partition each one becomes. Matched by
# first-payment month rather than by an exact file name, because the supplied
# extract carries a suffix that differs between machines.
COHORT_PATTERNS = {
    DEVELOPMENT_COHORT: "Quantum_florida_202303*.csv",
    TEST_COHORT: "Quantum_florida_202403*.csv",
}

# States each layer is expected to carry, fitted on the development cohort.
# Measured, not chosen: 24 three-digit ZIP prefixes and 19 sellers.
EXPECTED_LAYER_STATES: dict[str, int] = {"PostalCode": 24, "SellerName": 19}

# ``SpecialEligibilityProgram`` is null for every loan that is in no special
# program (Freddie Mac's layout codes the programs and leaves the field empty
# otherwise), so its absence level means "none", a recorded state, and never a
# blank field. A loan's record is not incomplete for carrying it.
STRUCTURAL_NONE_COLUMNS: tuple[str, ...] = ("SpecialEligibilityProgram",)

DATASET = DatasetConfig(
    name="fmac",
    layer_order=("PostalCode", "SellerName"),
    networks={
        NET_FMAC_POSTAL: ("PostalCode",),
        NET_FMAC_SELLER: ("SellerName",),
        NET_FMAC_2L: ("PostalCode", "SellerName"),
    },
    primary=NET_FMAC_2L,
    descriptions={
        NET_FMAC_POSTAL: "Single-layer ablation: borrowers who share a three-digit ZIP prefix.",
        NET_FMAC_SELLER: (
            "Single-layer ablation: borrowers whose loan was sold by the same seller."
        ),
        NET_FMAC_2L: (
            "Both Freddie Mac layers: the geographic and the originator relation together."
        ),
    },
    split="temporal",
    processed="data/fmac_processed.parquet",
    frozen_split="results/outputs/fmac_frozen_split.json",
    # Drawn over a different test partition than HMEQ's and never read in its place.
    bootstrap_indices="results/outputs/fmac_bootstrap_indices.npy",
    stages={
        "data": "01_fmac_data",
        "network": "02_fmac_network",
        "classical": "03_fmac_classical",
        "quantum": "04_fmac_quantum",
        "downstream": "05_fmac_downstream",
        "mechanism": "05a_fmac_mechanism",
        "hardware": "06_fmac_hardware",
    },
    expected_layer_states=EXPECTED_LAYER_STATES,
    # Both layer fields are complete in the supplied extract (the cleaning
    # notebook asserts it); an active layer marker on a test loan is a seller
    # or postal prefix the development vocabulary did not contain.
    layer_absence_means="unseen_level",
    structural_none_columns=STRUCTURAL_NONE_COLUMNS,
)


def raw_csvs(data_dir: str | Path | None = None) -> dict[str, Path]:
    """One raw Freddie Mac CSV per cohort, matched by first-payment month.

    Raises if a cohort is missing or ambiguous: silently picking one of several
    candidates could put the wrong quarter in the wrong partition.
    """
    directory = Path(data_dir) if data_dir is not None else DATA_DIR
    resolved: dict[str, Path] = {}
    for cohort, pattern in COHORT_PATTERNS.items():
        matches = sorted(directory.glob(pattern))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected exactly one file matching {pattern!r} in {directory}; "
                f"found {[match.name for match in matches]}"
            )
        resolved[cohort] = matches[0]
    return resolved
