"""What a dataset is, and the registry of the datasets the pipeline can run on.

:class:`DatasetConfig` holds a dataset's **identity**: its label column, its
layers and networks, how it was split, where its frozen artifacts live and
which stage directories it writes. One instance per dataset, defined in that
dataset's own package (``hmeq/config.py``, ``fmac/config.py``) and nowhere
else; the engine reads it and never names a dataset itself. A dataset may not
carry a run parameter -- those live in :class:`config.ExperimentConfig` alone,
so a difference between two datasets is a difference in the data, never in how
they were run.

A layer whose states are fixed writes its one-hot columns down; one whose
states are data (a postal prefix, a seller name) is discovered from the
processed frame by :meth:`DatasetConfig.network_specs`, under the same
:class:`config.NetworkSpec`.

:func:`get_dataset` is the one place a dataset *name* -- the ``--dataset`` of a
batch script -- is turned into its :class:`DatasetConfig`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from config import LayerSpec, NetworkSpec
from paths import repo_path, stage_dir, stage_path
from preprocessing.missing_levels import MISSING_INDICATOR_SUFFIX

DATASET_NAMES: tuple[str, ...] = ("hmeq", "fmac")
# The stage keys every dataset maps onto its own directory names.
STAGE_KEYS: tuple[str, ...] = (
    "data",
    "network",
    "classical",
    "quantum",
    "downstream",
    "mechanism",
    "hardware",
)
SPLIT_KINDS: tuple[str, ...] = ("stratified", "temporal")
ABSENCE_MEANINGS: tuple[str, ...] = ("blank_field", "unseen_level")


def discover_layer_spec(frame: pl.DataFrame, name: str) -> LayerSpec:
    """A :class:`LayerSpec` from the frame's own ``<name>_*`` one-hot columns.

    For a layer whose states are data. ``source_columns`` carries the absence
    marker alongside the genuine categories, exactly as a written-down layer
    does, while ``n_states`` counts only nodes.
    """
    columns = sorted(
        c
        for c in frame.columns
        if c.startswith(f"{name}_") and not c.endswith(MISSING_INDICATOR_SUFFIX)
    )
    if not columns:
        raise ValueError(f"layer {name!r} has no columns in the frame; run the cleaning stage first")
    return LayerSpec(name=name, source_columns=tuple(columns))


@dataclass(frozen=True)
class DatasetConfig:
    """What one dataset is: label, layers, networks, split, artifacts, stage directories.

    ``networks`` maps each network name to its layer names in block order and
    fixes the ladder order; ``primary`` is the network the folds, the
    controls, the placebo, the downstream models and the hardware sweep run
    on. ``layer_columns`` writes a layer's one-hot columns down when its states
    are fixed; a layer absent from it is discovered from the processed frame.
    Result tables are written under ``outputs/results/<name>/``: the dataset
    name is the scope and no file name carries a dataset prefix.
    """

    name: str
    layer_order: tuple[str, ...]
    networks: dict[str, tuple[str, ...]]
    primary: str
    split: str
    processed: str
    frozen_split: str
    bootstrap_indices: str
    stages: dict[str, str]
    label_col: str = "BAD"
    descriptions: dict[str, str] = field(default_factory=dict)
    layer_columns: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # The numeric covariates named first in the downstream feature matrix;
    # every other non-layer column of the frame follows in frame order.
    numeric_covariates: tuple[str, ...] = ()
    # Layer states the processed frame is expected to carry, where the states
    # are data. Measured once and written down; a mismatch is a finding.
    expected_layer_states: dict[str, int] = field(default_factory=dict)
    # What an active ``<Layer>_Missing`` marker means in this dataset: a
    # genuinely blank raw field (``blank_field``) or a level the development
    # vocabulary did not contain (``unseen_level``). Both cost the membership
    # edge; only the first is missingness of the record.
    layer_absence_means: str = "blank_field"
    # One-hot families whose absence level means "none" rather than "unknown"
    # (a program the loan is not in); never counted as a blank field.
    structural_none_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.layer_order:
            raise ValueError("a dataset needs a name and at least one layer")
        if len(self.layer_order) > 2:
            raise ValueError(f"{self.name}: at most two layers; got {self.layer_order}")
        if self.split not in SPLIT_KINDS:
            raise ValueError(f"{self.name}: split must be one of {SPLIT_KINDS}; got {self.split!r}")
        if self.primary not in self.networks:
            raise ValueError(f"{self.name}: primary {self.primary!r} is not one of its networks")
        for network, layers in self.networks.items():
            ordered = tuple(n for n in self.layer_order if n in layers)
            if not layers or len(set(layers)) != len(layers) or tuple(layers) != ordered:
                raise ValueError(
                    f"{self.name}: network {network!r} must name distinct layers in the block "
                    f"order {self.layer_order}; got {layers}"
                )
        missing = [key for key in STAGE_KEYS if key not in self.stages]
        if missing:
            raise ValueError(f"{self.name}: stage directories missing for {missing}")
        if self.layer_absence_means not in ABSENCE_MEANINGS:
            raise ValueError(f"{self.name}: layer_absence_means must be one of {ABSENCE_MEANINGS}")
        object.__setattr__(self, "structural_none_columns", tuple(self.structural_none_columns))
        object.__setattr__(self, "layer_order", tuple(self.layer_order))
        object.__setattr__(self, "networks", {k: tuple(v) for k, v in self.networks.items()})
        object.__setattr__(self, "numeric_covariates", tuple(self.numeric_covariates))

    # -- networks ----------------------------------------------------------

    @property
    def ladder(self) -> tuple[str, ...]:
        """Every network, in the order Table 1 reports them."""
        return tuple(self.networks)

    def layer_specs(self, frame: pl.DataFrame | None = None) -> dict[str, LayerSpec]:
        """One :class:`LayerSpec` per layer, written down or discovered from ``frame``."""
        specs: dict[str, LayerSpec] = {}
        for name in self.layer_order:
            if name in self.layer_columns:
                specs[name] = LayerSpec(name=name, source_columns=self.layer_columns[name])
            elif frame is not None:
                specs[name] = discover_layer_spec(frame, name)
            else:
                raise ValueError(
                    f"{self.name}: layer {name!r} is discovered from the processed frame; pass it"
                )
        return specs

    def network_specs(self, frame: pl.DataFrame | None = None) -> dict[str, NetworkSpec]:
        """The dataset's networks, keyed by name, in ladder order."""
        layers = self.layer_specs(frame)
        return {
            network: NetworkSpec(
                name=network,
                layers=tuple(layers[n] for n in names),
                description=self.descriptions.get(network, ""),
            )
            for network, names in self.networks.items()
        }

    def assert_expected_states(self, specs: dict[str, NetworkSpec]) -> dict[str, int]:
        """Fail if a discovered layer carries a different number of states than recorded.

        A mismatch means the level set the cleaner fitted has changed. That is
        a finding about the data or the cleaning stage; report it rather than
        editing ``expected_layer_states``.
        """
        observed = {
            layer.name: layer.n_states for spec in specs.values() for layer in spec.layers
        }
        mismatched = {
            name: (expected, observed.get(name))
            for name, expected in self.expected_layer_states.items()
            if observed.get(name) != expected
        }
        if mismatched:
            detail = "; ".join(f"{k}: expected {a}, got {b}" for k, (a, b) in mismatched.items())
            raise AssertionError(
                f"{self.name} layer states do not match the recorded counts ({detail})"
            )
        return observed

    # -- artifacts and directories ------------------------------------------

    def processed_path(self) -> Path:
        return repo_path(self.processed)

    def frozen_split_path(self) -> Path:
        return repo_path(self.frozen_split)

    def bootstrap_path(self) -> Path:
        return repo_path(self.bootstrap_indices)

    def stage(self, key: str) -> str:
        """The directory name of one stage under ``outputs/notebook_pipeline/``."""
        if key not in self.stages:
            raise ValueError(f"{self.name}: unknown stage {key!r}; expected one of {STAGE_KEYS}")
        return self.stages[key]

    def stage_dir(self, key: str) -> Path:
        return stage_dir(self.stage(key))

    def stage_path(self, key: str) -> Path:
        """The stage directory without creating it (for listing, never for writing)."""
        return stage_path(self.stage(key))

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ladder"] = list(self.ladder)
        return data


def get_dataset(name: str) -> DatasetConfig:
    """The :class:`DatasetConfig` registered under ``name``."""
    if name == "hmeq":
        from hmeq.config import DATASET
    elif name == "fmac":
        from fmac.config import DATASET
    else:
        raise ValueError(f"unknown dataset {name!r}; expected one of {DATASET_NAMES}")
    return DATASET
