"""What HMEQ is: two written-down layers, three networks, one stratified split.

The registry is exactly three networks and there is no mechanism for adding a
fourth: the two categorical MuLP layers (``JOB``, ``REASON``) singly as
ablations and jointly as the paper's primary network.
"""

from __future__ import annotations

from config import LayerSpec, NetworkSpec
from datasets import DatasetConfig
from paths import DATA_DIR

RAW_CSV = DATA_DIR / "hmeq.csv"

# The one-hot block the cleaner writes per layer, absence marker included.
JOB_COLUMNS: tuple[str, ...] = (
    "JOB_Mgr",
    "JOB_Missing",
    "JOB_Office",
    "JOB_Other",
    "JOB_ProfExe",
    "JOB_Sales",
    "JOB_Self",
)
REASON_COLUMNS: tuple[str, ...] = ("REASON_DebtCon", "REASON_HomeImp", "REASON_Missing")

# The numeric covariates, median-imputed on the development partition.
NUMERIC_COLS: tuple[str, ...] = (
    "LOAN",
    "MORTDUE",
    "VALUE",
    "YOJ",
    "DEROG",
    "DELINQ",
    "CLAGE",
    "NINQ",
    "CLNO",
    "DEBTINC",
)

NET_JOB_NAME = "NET_JOB"
NET_REASON_NAME = "NET_REASON"
NET_2L_NAME = "NET_2L"

DATASET = DatasetConfig(
    name="hmeq",
    layer_order=("JOB", "REASON"),
    networks={
        NET_JOB_NAME: ("JOB",),
        NET_REASON_NAME: ("REASON",),
        NET_2L_NAME: ("JOB", "REASON"),
    },
    primary=NET_2L_NAME,
    descriptions={
        NET_JOB_NAME: "Single-layer ablation: borrowers who share a JOB category.",
        NET_REASON_NAME: "Single-layer ablation: borrowers who share a loan REASON.",
        NET_2L_NAME: (
            "The paper's primary network: the two categorical MuLP layers, "
            "8 category nodes, 20 possible structural levels."
        ),
    },
    layer_columns={"JOB": JOB_COLUMNS, "REASON": REASON_COLUMNS},
    split="stratified",
    processed="data/hmeq_processed.parquet",
    frozen_split="results/outputs/frozen_split.json",
    bootstrap_indices="results/outputs/bootstrap_indices.npy",
    stages={
        "data": "01_data",
        "network": "02_network",
        "classical": "03_classical",
        "quantum": "04_quantum",
        "downstream": "05_downstream",
        "mechanism": "05a_mechanism",
        "hardware": "06_hardware",
    },
    numeric_covariates=NUMERIC_COLS,
)

# The static specifications, for callers (tests) that need one without a frame.
NETWORK_SPECS: dict[str, NetworkSpec] = DATASET.network_specs()
JOB_LAYER: LayerSpec = NETWORK_SPECS[NET_2L_NAME].layers[0]
REASON_LAYER: LayerSpec = NETWORK_SPECS[NET_2L_NAME].layers[1]
NET_JOB = NETWORK_SPECS[NET_JOB_NAME]
NET_REASON = NETWORK_SPECS[NET_REASON_NAME]
NET_2L = NETWORK_SPECS[NET_2L_NAME]
