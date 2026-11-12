"""The one frozen run configuration every arm reads, and the network types a dataset describes itself with.

:class:`ExperimentConfig` holds every **run parameter**: ``alpha``,
``stickiness``, the score definition (the exact Cesàro limit), the convergence
criterion that locates ``tau*``, the folds, the bootstrap, the placebo, the
variability study, the top-k grid, every seed, the hardware steps, shots, noise
arms and simulation budget, the calibration snapshot the simulated circuits
read, and the numerical tolerances of the lumping router and the spectral
limit. Nothing else in the repository may define one of these
locally, and a dataset may not override one: a difference between two datasets
must be a difference in the data, never in how they were run. A dataset's
**identity** -- its layers, networks, split and artifact paths -- is a
:class:`datasets.DatasetConfig`, defined in that dataset's own package.

There is no fixed horizon. The reported QPPR score is the exact limit of the
time-averaged measured distribution, computed spectrally on the lumped
operator; ``tau*`` is a per-network convergence point that is computed and
reported, never chosen. The running average converges; the per-step
distributions of a unitary walk oscillate and are never described as
converging.

Seeds
-----
Every random draw in the repository takes its seed from
:func:`derive_seed` applied to ``master_seed`` and a purpose string (plus the
indices that distinguish repeated draws), so that two draws never share a seed
by accident and every seed is reproducible from the one master value. The
stratified split of a dataset without dates is the one exception: it is drawn
from ``split_seed`` (42) and frozen once.

Network specifications
Network specifications
----------------------
:class:`LayerSpec` and :class:`NetworkSpec` are the network types a dataset
describes itself with: a layer and a one- or two-layer network. A layer is a
"borrowers who share this recorded attribute" relation; quantile bins of
continuous risk drivers are not layers and are not expressible here. A network
has at most two layers, in the dataset's block order. Every dataset package and
test imports them from ``config``.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from preprocessing.missing_levels import is_missing_column


__all__ = [
    "DEFAULT_CONFIG", "DEFAULT_MECHANISM", "ExperimentConfig", "LayerSpec", "MechanismConfig",
    "NetworkSpec", "derive_seed",
]


# --- seeds -------------------------------------------------------------------


def derive_seed(master_seed: int, purpose: str, *parts: Any) -> int:
    """A 31-bit seed from the master seed, a purpose string and distinguishing indices.

    ``SHA-256`` of ``"<master>|<purpose>|<part>|..."`` truncated to 31 bits, so
    every draw has its own documented seed and no two purposes collide. The
    parts are rendered with ``str``; pass dataset names, step indices or arm
    names, never floats whose printing could differ between platforms.
    """
    text = "|".join([str(int(master_seed)), str(purpose), *(str(p) for p in parts)])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


# --- the calibration snapshot ------------------------------------------------

# The local calibration snapshot every simulated arm reads. FakeMarrakesh is a
# Heron r2 snapshot (156 qubits, native CZ); its calibration medians drive the
# noise model of the two noisy arms, the per-circuit readout mitigation and the
# QPU-time estimate. Nothing in the repository submits to a device: the real
# run, if any, is made by hand from the export of scripts/export_hardware_instance.py.
IBM_FAKE_BACKEND = "FakeMarrakesh"


# --- the frozen run configuration -------------------------------------------


@dataclass(frozen=True)
class ExperimentConfig:
    stickiness: float = 1.0
    # The paper's alpha. Fixed at 0.85 and never refit.
    alpha: float = 0.85
    influence_type: str = "combined"
    # The reported QPPR score. "limit" is the exact Cesàro limit of the
    # source-register time average (Prop. 1), computed spectrally on the lumped
    # operator through the router. It is the only admissible value; the field
    # exists so every manifest states which score definition produced its numbers.
    quantum_score: str = "limit"
    # Recorded steps t = 0 .. reduction_check_steps - 1 of the full-operator
    # walk kept for the full-versus-reduced check (max |unfolded lumped step
    # marginal - full step marginal|). Not a horizon of any reported score.
    reduction_check_steps: int = 21
    # The convergence point tau*: the smallest grid horizon from which the
    # running average stays within conv_tol_l1 (L1 over supra-nodes) of the
    # exact limit, keeps Spearman >= conv_min_spearman with it over the test
    # individuals and keeps the expected top-k overlap within conv_overlap_slack
    # of the self-overlap ceiling, continuously up to tau_max. "> tau_max" is a
    # reported outcome, never a silent fallback.
    conv_tol_l1: float = 0.01
    conv_min_spearman: float = 0.99
    conv_overlap_slack: float = 0.01
    # tau* is accepted only if the criterion holds on a non-trivial tail: at least this many grid
    # points at or above tau* (a criterion that holds only at the cap, after failing at every earlier
    # grid point, is not a convergence point and is reported as "> tau_max" with its last violation).
    conv_min_tail_points: int = 20
    # The largest horizon the running average is followed to, per dataset.
    conv_tau_max: dict[str, int] = field(
        default_factory=lambda: {"hmeq": 50_000, "fmac": 200_000}
    )
    # tau* is re-evaluated over these tolerance grids (the sensitivity table).
    conv_sensitivity_l1: tuple[float, ...] = (0.005, 0.01, 0.02)
    conv_sensitivity_spearman: tuple[float, ...] = (0.98, 0.99, 0.995)
    # Top-k grid; primary_k is the k every headline metric reports.
    top_k: tuple[int, ...] = (10, 25, 50, 100)
    primary_k: int = 100
    # Decimals of rounding applied before every metric. Scores that differ only
    # below this are ties, not an ordering.
    tie_tolerance: int = 12
    # One outcome-stratified bootstrap index matrix per dataset, shared by every arm.
    bootstrap_reps: int = 2_000
    crossfit_folds: int = 5
    # Seed-permutation placebo: B permutations, p = (1 + #(placebo >= true)) / (1 + B).
    n_placebo_permutations: int = 99
    # Variability study: HMEQ is re-split with the distinct seeds {frozen split seed 42} U {1..n}
    # (as a set: 42 lies inside 1..50, so n = 50 gives 50 distinct splits, the frozen one among them;
    # n = 51 gives 51); Freddie Mac's development reference set is resampled at 80%, this many draws.
    n_splits_hmeq: int = 50
    n_reference_resamples_fmac: int = 50
    master_seed: int = 42
    # The stratified split a dataset without dates draws once (stage 01); a
    # temporally split dataset ignores both.
    test_frac: float = 0.2
    split_seed: int = 42
    # Hardware tier: the recorded steps tau = 1..5 of the noise grid, the shots
    # per circuit, and the simulated arms graded against the exact lumped walk.
    hardware_steps: tuple[int, ...] = (1, 2, 3, 4, 5)
    hardware_shots: int = 4_096
    hardware_noise_arms: tuple[str, ...] = ("noiseless_shots", "aer_noise_model", "fake_marrakesh")
    # The retained candidate must reach this fidelity ratio (1 - TV/TV_uniform)
    # under the fake-backend arm to count as able to rank test borrowers.
    hardware_min_fidelity_ratio: float = 0.5
    # The selected instance must be read at tau >= this: at tau = 1 the circuit holds
    # the state preparation alone (p_0 = v, no walk step), which is not a walk result.
    hardware_min_tau: int = 2
    # Largest register width m the noise grid transpiles and simulates: one
    # reflection--swap costs about 2^m (2^m - 1) two-qubit gates. Wider instances
    # are not attempted and every one of their cells is recorded as "not simulated:
    # resource budget" with the estimate. At m = 6 (12 logical qubits) the ideal arm
    # runs, while the two density-matrix arms exceed the simulation budget beyond the
    # first recorded steps (or entirely, when the routed circuit touches more qubits
    # than the memory cap allows); those cells are recorded the same way.
    hardware_max_register_width: int = 6
    # Simulation budget of one retained instance, shared by its arms and recorded steps.
    # A noisy arm runs as a density matrix of 16 * 4^n bytes over the n physical qubits
    # the routed circuit touches; a cell whose state needs more than this is not run.
    hardware_density_matrix_max_bytes: int = 4 * 2**30
    # Wall-clock seconds one instance may spend simulating. Cells run in recorded-step
    # order, all arms at each step; before a cell the time it needs is estimated from
    # its gates and state size (at the arm's measured rate, or at the assumed throughput
    # below until the arm has run), and a cell that would end past this budget is not
    # run. Below the 4 h allocation of an array task, which also pays for the transpilation.
    hardware_instance_budget_seconds: float = 10_800.0
    # A gate of a simulated circuit sweeps the whole state, so a cell costs about gates x state
    # bytes / throughput. This is the throughput assumed for the first cell of an arm, before any
    # rate has been measured; it is deliberately pessimistic, so that a first cell which could not
    # end inside the instance budget is declined with its estimate instead of being started.
    hardware_prior_bytes_per_second: float = 2.0e9
    # The pure-noise reference of Figure 4: a device that returns uniform random
    # codes, sampled with the same shots, converted and graded like a real output.
    # The band is these quantiles (percent) over this many independent draws, per
    # instance and recorded step, against that instance's own exact time average.
    hardware_noise_band_draws: int = 1_000
    hardware_noise_band_quantiles: tuple[float, float] = (5.0, 95.0)
    # Largest m the noiseless statevector rung builds the logical circuit for
    # (no routing, no noise; m = 8 is 65,280 preparation angles per step).
    hardware_statevector_max_register_width: int = 8
    # Preset pass-manager level for the ISA circuits of the noise grid.
    hardware_optimization_level: int = 1
    # Estimated QPU seconds (shots x circuit durations, walk and calibration circuits) the
    # hand-run submission of the selected instance may consume: half of an Open Plan
    # allocation. The candidate selection keeps an instance only within this estimate.
    hardware_max_runtime_seconds: float = 300.0
    # Class counts of the nested retained instances. The ladder always begins at
    # the lumped operator's actual class count (entries above it are clamped to
    # it, the full count is inserted if absent) and descends without large gaps.
    # Two classes is a legitimate rung (six supra-nodes, three qubits per
    # register); on HMEQ it retains 315 test borrowers in one masked class.
    hardware_class_sweep: tuple[int, ...] = (
        54, 48, 42, 36, 30, 26, 24, 22, 20, 18, 16, 14, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2,
    )
    # The calibration snapshot (a fake backend shipped with qiskit-ibm-runtime) every arm reads.
    ibm_backend: str = IBM_FAKE_BACKEND
    # Numerical tolerances. router_tol decides between the lumped discriminant
    # D-bar and the compressed discriminant H (max |D B - B D-bar| <= router_tol
    # routes to D-bar); d_tol treats a discriminant eigenvalue within it of +-1
    # as a one-dimensional block; phase_tol groups U-eigenphases into one
    # eigenspace; mass_tol is the unit-sum check on every limiting distribution.
    router_tol: float = 1e-12
    d_tol: float = 1e-9
    phase_tol: float = 1e-9
    mass_tol: float = 1e-9
    overrides: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.influence_type not in ("combined", "intra", "inter"):
            raise ValueError("influence_type must be 'combined', 'intra', or 'inter'")
        if not 0.0 < float(self.alpha) < 1.0:
            raise ValueError("alpha must be strictly between zero and one")
        if float(self.stickiness) <= 0.0:
            raise ValueError("stickiness must be positive")
        if self.quantum_score != "limit":
            raise ValueError("quantum_score must be 'limit': the reported score is the exact Cesàro limit")
        for name in (
            "reduction_check_steps", "hardware_shots", "bootstrap_reps", "crossfit_folds",
            "n_placebo_permutations", "n_splits_hmeq", "n_reference_resamples_fmac",
            "hardware_max_register_width", "conv_min_tail_points",
            "hardware_noise_band_draws", "hardware_density_matrix_max_bytes",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.crossfit_folds < 2:
            raise ValueError("crossfit_folds must be at least 2")
        if self.primary_k not in self.top_k:
            raise ValueError(f"primary_k {self.primary_k} must appear in top_k {self.top_k}")
        if not (0.0 < float(self.conv_tol_l1) and 0.0 < float(self.conv_min_spearman) <= 1.0):
            raise ValueError("conv_tol_l1 must be positive and conv_min_spearman in (0, 1]")
        if not 0.0 <= float(self.conv_overlap_slack) < 1.0:
            raise ValueError("conv_overlap_slack must lie in [0, 1)")
        if not self.conv_tau_max or any(int(t) < 1 for t in self.conv_tau_max.values()):
            raise ValueError("conv_tau_max must give a positive horizon per dataset")
        if not self.hardware_steps or any(int(t) < 1 for t in self.hardware_steps):
            raise ValueError("hardware_steps must be positive recorded-step counts")
        if not self.hardware_noise_arms:
            raise ValueError("hardware_noise_arms must name at least one arm")
        if not 0.0 < float(self.hardware_min_fidelity_ratio) <= 1.0:
            raise ValueError("hardware_min_fidelity_ratio must lie in (0, 1]")
        if float(self.hardware_max_runtime_seconds) <= 0.0:
            raise ValueError("hardware_max_runtime_seconds must be positive")
        if float(self.hardware_instance_budget_seconds) <= 0.0:
            raise ValueError("hardware_instance_budget_seconds must be positive")
        if float(self.hardware_prior_bytes_per_second) <= 0.0:
            raise ValueError("hardware_prior_bytes_per_second must be positive")
        if not 1 <= int(self.hardware_min_tau) <= max(int(t) for t in self.hardware_steps):
            raise ValueError("hardware_min_tau must be one of the recorded steps (between 1 and the largest hardware step)")
        lo, hi = (float(q) for q in self.hardware_noise_band_quantiles)
        if not 0.0 < lo < hi < 100.0:
            raise ValueError("hardware_noise_band_quantiles must be two percentiles 0 < lo < hi < 100")
        if int(self.hardware_optimization_level) not in (0, 1, 2, 3):
            raise ValueError("hardware_optimization_level must be a preset level 0..3")
        if int(self.hardware_statevector_max_register_width) < int(self.hardware_max_register_width):
            raise ValueError("hardware_statevector_max_register_width must not be below hardware_max_register_width")
        for name in ("router_tol", "d_tol", "phase_tol", "mass_tol"):
            if not float(getattr(self, name)) > 0.0:
                raise ValueError(f"{name} must be positive")
        object.__setattr__(self, "conv_tau_max", {str(k): int(v) for k, v in self.conv_tau_max.items()})
        object.__setattr__(self, "hardware_steps", tuple(int(t) for t in self.hardware_steps))
        object.__setattr__(self, "hardware_noise_band_quantiles", tuple(float(q) for q in self.hardware_noise_band_quantiles))

    # -- derived values -------------------------------------------------------

    def resolved_alpha(self) -> float:
        """The damping factor of the paper: fixed, never refit."""
        return float(self.alpha)

    def tau_max_for(self, dataset_name: str) -> int:
        """The convergence cap of one dataset; a dataset without one is an error, not a default."""
        if dataset_name not in self.conv_tau_max:
            raise KeyError(f"conv_tau_max carries no cap for dataset {dataset_name!r}")
        return int(self.conv_tau_max[dataset_name])

    # -- seeds, all derived from master_seed ----------------------------------

    def seed_bootstrap(self) -> int:
        """The shared outcome-stratified bootstrap matrix of every dataset."""
        return derive_seed(self.master_seed, "bootstrap")

    def seed_folds(self) -> int:
        """The stratified K-fold assignment both arms cross-fit on."""
        return derive_seed(self.master_seed, "folds")

    def seed_placebo(self, permutation: int) -> int:
        """Placebo seed set number ``permutation`` (0-based)."""
        return derive_seed(self.master_seed, "placebo", int(permutation))

    def seed_synthetic(self, *parts: Any) -> int:
        """A synthetic-experiment draw, keyed by its experiment tag and replicate."""
        return derive_seed(self.master_seed, "synthetic", *parts)

    def seed_transpiler(self, *parts: Any) -> int:
        """The transpiler seed of one (dataset, instance) compilation."""
        return derive_seed(self.master_seed, "transpiler", *parts)

    def seed_aer(self, *parts: Any) -> int:
        """The Aer simulator seed of one (dataset, instance, tau, arm, step) run."""
        return derive_seed(self.master_seed, "aer", *parts)

    def seed_reference_resample(self, draw: int) -> int:
        """One 80% resample of the Freddie Mac development reference set."""
        return derive_seed(self.master_seed, "reference_resample", int(draw))

    def seed_downstream(self, model: str) -> int:
        return derive_seed(self.master_seed, "downstream", str(model))

    def seed_noise_band(self, *parts: Any) -> int:
        """The draws of the pure-noise band of one (dataset, instance), as one stream."""
        return derive_seed(self.master_seed, "noise_band", *parts)

    # -- serialisation ---------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in (
            "top_k", "conv_sensitivity_l1", "conv_sensitivity_spearman", "hardware_steps",
            "hardware_noise_arms", "hardware_class_sweep", "hardware_noise_band_quantiles", "overrides",
        ):
            data[key] = list(getattr(self, key))
        data["conv_tau_max"] = dict(self.conv_tau_max)
        return data

    def with_overrides(self, **changes: Any) -> "ExperimentConfig":
        applied = tuple(sorted(changes)) + self.overrides
        return replace(self, overrides=applied, **changes)


DEFAULT_CONFIG = ExperimentConfig()


# --- the mechanism grids ------------------------------------------------------


@dataclass(frozen=True)
class MechanismConfig:
    """The control grids of the mechanism stage (notebook 05a).

    Hashed into that stage's manifests and table metadata only; never into
    :class:`ExperimentConfig`, whose hash every stage manifest and every warm
    checkpoint fingerprint carries. Nothing here is a run parameter of a
    reported score: every grid point is a control on the lumped operator. The
    numerical tolerances of the limit live in :class:`ExperimentConfig`.
    """

    # Interlayer weights swept on the two-layer lumped operator. 0.0 is a
    # legitimate value: the copies stay coupled through the combined restart
    # (a reference individual restarts to both copies) and through dangling
    # patches to v. The router chooses the discriminant at every weight and
    # T_eta records the route it chose.
    eta_grid: tuple[float, ...] = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
    # Damping values of the classical (alpha, eta) grid; the frozen alpha must be a member.
    alpha_grid: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99)
    # The limiting score must match the long-horizon time average within this
    # normalized L1 on the test fixture.
    fixture_l1_tol: float = 5e-4
    # A smoke run trims the grids; it is recorded in the manifest and its
    # outputs are never reportable.
    smoke: bool = False

    def __post_init__(self) -> None:
        etas = tuple(float(eta) for eta in self.eta_grid)
        if not etas or any(eta < 0.0 for eta in etas):
            raise ValueError("eta_grid must be non-empty and non-negative")
        if float(ExperimentConfig.stickiness) not in etas:
            raise ValueError(
                f"eta_grid must contain the frozen stickiness {ExperimentConfig.stickiness}"
            )
        alphas = tuple(float(a) for a in self.alpha_grid)
        if not alphas or any(not 0.0 < a < 1.0 for a in alphas):
            raise ValueError("alpha_grid values must lie strictly in (0, 1)")
        if float(ExperimentConfig.alpha) not in alphas:
            raise ValueError(f"alpha_grid must contain the frozen alpha {ExperimentConfig.alpha}")
        if not float(self.fixture_l1_tol) > 0.0:
            raise ValueError("fixture_l1_tol must be positive")
        object.__setattr__(self, "eta_grid", etas)
        object.__setattr__(self, "alpha_grid", alphas)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("eta_grid", "alpha_grid"):
            data[key] = list(getattr(self, key))
        return data


DEFAULT_MECHANISM = MechanismConfig()


# ---------------------------------------------------------------------------
# Network specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerSpec:
    """One categorical layer: its name and the one-hot columns the cleaner emits.

    ``source_columns`` includes the ``<LAYER>_Missing`` absence marker so the
    schema check sees every column the cleaner writes; the marker is never a
    category node (see ``preprocessing.missing_levels``), and ``n_states`` excludes it.
    """

    name: str
    source_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a layer needs a name")
        if not self.source_columns:
            raise ValueError(f"layer {self.name}: source_columns must not be empty")
        object.__setattr__(self, "source_columns", tuple(self.source_columns))

    @property
    def genuine_columns(self) -> tuple[str, ...]:
        return tuple(c for c in self.source_columns if not is_missing_column(c))

    @property
    def marker_column(self) -> str:
        return f"{self.name}_Missing"

    @property
    def n_states(self) -> int:
        """Category states, excluding the unrecorded bucket."""
        return len(self.genuine_columns)

    @property
    def prefix(self) -> str:
        return f"{self.name}_"


@dataclass(frozen=True)
class NetworkSpec:
    """A named set of one or two layers, in block order."""

    name: str
    layers: tuple[LayerSpec, ...]
    description: str = ""

    def __post_init__(self) -> None:
        names = [layer.name for layer in self.layers]
        if not names or len(set(names)) != len(names):
            raise ValueError(f"network {self.name}: layers must be non-empty and distinct")
        if len(names) > 2:
            raise ValueError(f"network {self.name}: at most two layers; got {names}")

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def layer_names(self) -> tuple[str, ...]:
        return tuple(layer.name for layer in self.layers)

    @property
    def total_states(self) -> int:
        return sum(layer.n_states for layer in self.layers)

    def layer_size(self, n_borrowers: int) -> int:
        """Positions one layer block reserves: borrowers + every layer's states.

        The MuLP block convention: each layer reserves a full copy of the shared
        identity space, only its own category segment carrying edges.
        """
        return int(n_borrowers) + self.total_states

    def n_supra_nodes(self, n_borrowers: int) -> int:
        return self.n_layers * self.layer_size(n_borrowers)

    def max_distinct_levels(self) -> int:
        """Ceiling on how many different scores this network can produce.

        Borrowers touching the same category nodes are graph automorphs and get
        identical structural scores, so the ceiling is ``prod(states_l + 1) - 1``
        (the ``+1`` is the unrecorded state; the ``-1`` removes the borrower
        with nothing recorded anywhere, who is dropped upstream).

        The ``-1`` therefore assumes this network spans every layer of its
        dataset, which is what the upstream drop is judged on. A single-layer
        ablation can retain a borrower unrecorded in its one layer, and for
        that population the ceiling is one higher; the applicable value is
        measured per population as ``max_distinct_levels_in_population`` in
        :func:`classical.network_diagnostics.describe_network`.
        """
        product = 1
        for layer in self.layers:
            product *= layer.n_states + 1
        return product - 1
