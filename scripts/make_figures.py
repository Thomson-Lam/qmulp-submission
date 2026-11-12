"""Every figure of the manuscript from the CSV tables alone, in one style.

    python -m scripts.make_figures             # all figures whose CSVs exist
    python -m scripts.make_figures --only fig2

Reads only ``outputs/results/<scope>/*.csv`` and writes only ``outputs/manuscript/figures/*.pdf``. Each
figure prints a few spot-check values read back off its CSV and the page width read back off the saved
PDF (6.5 in), and a figure whose input is missing is skipped, saying so, rather than drawn from anything else.

The figure style
----------------
One figure style for every figure: INFORMS sizing, embedded fonts, vector PDF, a grayscale-safe palette.
Full width is exactly 6.5 in: a figure is laid out by the constrained layout engine and saved without a
cropping bounding box, so the PDF page is the figure, and :func:`save` reads the page width back from the
file. A panel is about 3 in tall; type is 9 pt with nothing below 7 pt; fonts are embedded (Type 42) so the
PDF is vector. Every series differs in marker and line style as well as colour, and the palette has no light
yellow, so a figure survives grayscale; panels carry a letter and no in-figure title.
"""

from __future__ import annotations

import argparse
import re
import sys
from itertools import cycle
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_CONFIG  # noqa: E402
from evaluation.formatting import ARM_LABELS, DATASET_LABELS, NETWORK_LABELS, SYNTHETIC_LABELS  # noqa: E402
from evaluation.provenance import read_result  # noqa: E402
from paths import manuscript_dir  # noqa: E402


# ---------------------------------------------------------------------------
# The figure style
# ---------------------------------------------------------------------------


FULL_WIDTH_IN = 6.5
PANEL_HEIGHT_IN = 3.0
FONT_PT = 9
SMALL_PT = 7
# Okabe--Ito without its yellow (invisible on white and in grayscale), plus a dark wine.
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#882255", "#000000")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
LINESTYLES = ("-", "--", "-.", ":", "-", "--", "-.", ":")
GREY = "#7f7f7f"


def apply_style() -> None:
    """Set the rcParams every figure script uses."""
    plt.rcParams.update({
        "font.size": FONT_PT, "axes.labelsize": FONT_PT, "axes.titlesize": FONT_PT, "legend.fontsize": SMALL_PT + 1,
        "xtick.labelsize": SMALL_PT + 1, "ytick.labelsize": SMALL_PT + 1, "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "lines.linewidth": 1.2, "lines.markersize": 4, "figure.dpi": 150, "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
    })


def series_styles():
    """An endless iterator of ``(colour, marker, linestyle)`` triples, redundant on purpose."""
    return cycle(zip(PALETTE, MARKERS, LINESTYLES))


def colour_markers():
    """An endless iterator of ``(colour, marker)`` pairs, for figures where the line style carries a meaning."""
    return cycle(zip(PALETTE, MARKERS))


def panel_letter(ax, letter: str) -> None:
    """``(a)``, ``(b)``, ... above the top-left corner of a panel (the layout engine reserves its room); no in-figure titles."""
    ax.set_title(f"({letter})", loc="left", fontsize=FONT_PT, fontweight="bold", pad=3)


def figure(n_cols: int, n_rows: int = 1, *, height: float = PANEL_HEIGHT_IN, width: float = FULL_WIDTH_IN, **kwargs):
    """A full-width constrained-layout figure with ``n_rows x n_cols`` panels; ``kwargs`` go to ``plt.subplots``."""
    return plt.subplots(n_rows, n_cols, figsize=(width, height * n_rows), layout="constrained", squeeze=False, **kwargs)


def break_marks(left_ax, right_ax, size: float = 7.0) -> None:
    """Slashes on the x axes where ``left_ax`` and ``right_ax`` meet: the axis is broken between them."""
    kwargs = dict(marker=[(-0.5, -1.0), (0.5, 1.0)], markersize=size, linestyle="none", color="black", mec="black", mew=0.8, clip_on=False)
    left_ax.plot([1], [0], transform=left_ax.transAxes, **kwargs)
    right_ax.plot([0], [0], transform=right_ax.transAxes, **kwargs)


def measured_width_in(path) -> float | None:
    """Width of a saved PDF's page in inches, read from its MediaBox; ``None`` when the box cannot be read."""
    match = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]", Path(path).read_bytes())
    return None if match is None else float(match.group(1)) / 72.0


def save(fig, path) -> float | None:
    """Vector PDF with embedded fonts, no bounding-box cropping; returns the page width read back (inches).

    Raises ``AssertionError`` when the page is not 6.5 in wide.
    """
    fig.savefig(path, format="pdf")
    plt.close(fig)
    width = measured_width_in(path)
    if width is not None and abs(width - FULL_WIDTH_IN) > 0.005:
        raise AssertionError(f"{path} is {width:.3f} in wide; every figure must be {FULL_WIDTH_IN} in")
    return width


# ---------------------------------------------------------------------------
# The figures
# ---------------------------------------------------------------------------


DATASETS = tuple(DATASET_LABELS.items())            # every legend and axis label comes from evaluation.formatting: no code identifier is drawn
SMALL = SMALL_PT
NOISE_ARMS = ("aer_noise_model", "fake_marrakesh")  # the two noisy simulated arms, the columns of the noise figures
# The instance sizes the per-step noise figures draw (one colour each); the other sizes stay in the CSV.
K_SUBSET = (2, 3, 4, 5, 6, 8, 12)
K_TICKS = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96)


def _read(scope: str, name: str) -> pl.DataFrame | None:
    """One result CSV (schema inferred over the whole file), or ``None`` when it is absent."""
    return read_result(scope, name)


def _read_datasets(name: str) -> pl.DataFrame | None:
    """Both datasets' copies of a result table stacked, each row carrying its dataset; ``None`` when neither exists."""
    parts = []
    for ds, _label in DATASETS:
        frame = _read(ds, name)
        if frame is not None:
            parts.append(frame if "dataset" in frame.columns else frame.with_columns(pl.lit(ds).alias("dataset")))
    return pl.concat(parts, how="diagonal") if parts else None


def _out(name: str):
    return manuscript_dir("figures") / name


def _spots(figure: str, pairs) -> None:
    """Print the values read back off the CSV that the figure draws, one line each."""
    for label, values in pairs:
        values = values if isinstance(values, (list, tuple)) else [values]
        shown = ", ".join(f"{v:.4g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v) for v in values) or "n/a"
        print(f"  spot-check {figure} {label}: {shown}")


def _finish(fig, name: str) -> Path:
    """Save under ``outputs/manuscript/figures`` and print the page width read back from the PDF."""
    out = _out(name)
    width = save(fig, out)
    print(f"  page width {'unreadable' if width is None else f'{width:.3f} in'}")
    return out


def fig1_coupling() -> Path | None:
    """(a) Spearman(QPPR, classical) against eta, the classical-vs-primary check dashed; (b) partially observed share of the QPPR top k.

    Each panel has a separate segment for eta = 0 (the axis is broken), then a log axis for eta > 0.
    """
    frames = [(label, _read(ds, "T_eta")) for ds, label in DATASETS]
    frames = [(label, f.sort("eta")) for label, f in frames if f is not None]
    if not frames:
        print("fig1: no T_eta table; skipped")
        return None
    fig, axes = figure(4, width_ratios=[1, 9, 1, 9])
    panels = ((axes[0, 0], axes[0, 1], "rho_qc", "rho_c_vs_primary_c", r"Spearman $\rho$(QPPR, classical)"),
              (axes[0, 2], axes[0, 3], "partial_share_topk_q", None, "Partially observed share\nof the QPPR top-$k$"))
    for zero_ax, log_ax, *_ in panels:
        log_ax.sharey(zero_ax)                      # before anything is drawn: the two segments share one y scale
    styles = colour_markers()
    handles, spots = [], []
    positive = []
    for label, f in frames:
        colour, marker = next(styles)
        pos, zero = f.filter(pl.col("eta") > 0.0), f.filter(pl.col("eta") == 0.0)
        positive += pos["eta"].to_list()
        for zero_ax, log_ax, primary, check, _ylabel in panels:
            for column, linestyle, face in ((primary, "-", colour), (check, "--", "none")):
                if column is None or column not in f.columns:
                    continue
                log_ax.plot(pos["eta"], pos[column], color=colour, marker=marker, linestyle=linestyle, markerfacecolor=face)
                if zero.height:
                    zero_ax.plot([0.0], zero[column], color=colour, marker=marker, linestyle="none", markerfacecolor=face)
        handles.append(Line2D([], [], color=colour, marker=marker, linestyle="-", label=label))
        at_one = pos.filter(pl.col("eta") == 1.0)
        spots += [(f"{label} rho_qc at eta=1", at_one["rho_qc"].to_list()),
                  (f"{label} QPPR top-k partially observed share at eta=1", at_one["partial_share_topk_q"].to_list())]
    handles.append(Line2D([], [], color=GREY, marker="o", markerfacecolor="none", linestyle="--",
                          label=r"classical at $\eta$ vs classical at the primary $\eta$ (a)"))
    lo, hi = (min(positive) / 1.5, max(positive) * 1.5) if positive else (0.5, 2.0)
    for letter, (zero_ax, log_ax, *_rest) in zip("ab", panels):
        ylabel = _rest[-1]
        zero_ax.set_xlim(-0.6, 0.6)
        zero_ax.set_xticks([0.0])
        zero_ax.set_xticklabels(["0"])
        zero_ax.set_ylabel(ylabel)
        log_ax.set_xscale("log")
        log_ax.set_xlim(lo, hi)
        log_ax.set_xlabel(r"interlayer weight $\eta$")
        log_ax.spines["left"].set_visible(False)
        log_ax.tick_params(left=False, labelleft=False)
        break_marks(zero_ax, log_ax)
        panel_letter(zero_ax, letter)
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=SMALL)
    out = _finish(fig, "fig1_coupling.pdf")
    _spots("fig1", spots)
    return out


STAR_TICK = 0.14  # height of the tau* tick, as a share of the axis


def _darker(colour: str, factor: float = 0.55) -> tuple[float, float, float]:
    """``colour`` scaled towards black: the tau* tick reads against the curve it belongs to."""
    r, g, b = to_rgb(colour)
    return (r * factor, g * factor, b * factor)


def fig2_convergence() -> Path | None:
    """Left: L1 distance to the limit against tau (log-log) with the tolerance and c/tau; right: Spearman with the limit."""
    config = DEFAULT_CONFIG
    parts = [(ds, label, _read(ds, "EC_convergence_curves"), _read(ds, "T_convergence")) for ds, label in DATASETS]
    parts = [p for p in parts if p[2] is not None and p[3] is not None]
    if not parts:
        print("fig2: no convergence curves; skipped")
        return None
    fig, axes = figure(2)
    ax_l1, ax_rho = axes[0, 0], axes[0, 1]
    styles = colour_markers()
    handles, spots, caps = [], [], {}
    for ds, label, frame, summary in parts:
        single_layer_styles = cycle(("--", "-."))
        for row in summary.to_dicts():
            network = row["network"]
            f = frame.filter(pl.col("network") == network).sort("tau")
            colour, marker = next(styles)
            two_layer = network.endswith("2L")
            linestyle, width = ("-", 1.6) if two_layer else (next(single_layer_styles), 1.0)
            every = max(1, f.height // 7)
            for ax, column in ((ax_l1, "l1_tau"), (ax_rho, "rho_tau")):
                ax.plot(f["tau"], f[column], color=colour, marker=marker, linestyle=linestyle, linewidth=width, markevery=every,
                        markersize=3.5)
            handles.append(Line2D([], [], color=colour, marker=marker, linestyle=linestyle, linewidth=width, markersize=3.5,
                                  label=f"{label}: {NETWORK_LABELS[network]}"))
            if two_layer:
                taus = f["tau"].to_numpy().astype(float)
                ax_l1.plot(taus, row["c_exact"] / taus, color=colour, linestyle=(0, (1, 1.5)), linewidth=0.8)
                if row["tau_star_reached"]:
                    for ax in (ax_l1, ax_rho):                  # a tick on the x axis (drawn as a line so the y limits are untouched): long and dark
                        star = float(row["tau_star"])
                        ax.plot([star, star], [0.0, STAR_TICK], transform=ax.get_xaxis_transform(), color=_darker(colour), linewidth=2.2,
                                solid_capstyle="butt", zorder=5)
                spots += [(f"{label} {network} L1 at tau=100", f.filter(pl.col("tau") == 100)["l1_tau"].to_list()),
                          (f"{label} {network} rho at tau=1000", f.filter(pl.col("tau") == 1000)["rho_tau"].to_list()),
                          (f"{label} {network} tau*", [row["tau_star"]])]
        caps.setdefault(config.tau_max_for(ds), []).append(label)
    for tau_max in caps:                                    # one cap line per distinct cap; its value is in the legend, not drawn over the curves
        for ax in (ax_l1, ax_rho):
            ax.axvline(tau_max, color=GREY, linestyle="--", linewidth=0.8)
    cap_text = "; ".join(f"{' / '.join(labels)} {tau_max:,}" for tau_max, labels in caps.items())
    ax_l1.axhline(config.conv_tol_l1, color=GREY, linestyle=":", linewidth=0.8)
    ax_rho.axhline(config.conv_min_spearman, color=GREY, linestyle=":", linewidth=0.8)
    for ax in (ax_l1, ax_rho):
        ax.set_xscale("log")
        ax.set_xlabel(r"$\tau$")
    ax_l1.set_yscale("log")
    ax_l1.set_ylabel(r"$\|\bar\omega_\tau - \omega\|_1$")
    ax_rho.set_ylabel(r"Spearman $\rho(\bar\omega_\tau, \omega)$")
    panel_letter(ax_l1, "a")
    panel_letter(ax_rho, "b")
    handles += [Line2D([], [], color="black", linestyle=(0, (1, 1.5)), linewidth=0.8, label=r"$c/\tau$ (two-layer networks)"),
                Line2D([], [], color="black", marker="|", linestyle="none", markersize=9, markeredgewidth=2.2, label=r"$\tau^*$ (tick on the axis)"),
                Line2D([], [], color=GREY, linestyle=":", linewidth=0.8, label="tolerance"),
                Line2D([], [], color=GREY, linestyle="--", linewidth=0.8, label=rf"$\tau$ cap ({cap_text})")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=SMALL)
    out = _finish(fig, "fig2_convergence.pdf")
    _spots("fig2", spots[:3])
    return out


def _band(ax, x, mean, lo, hi, colour, marker, linestyle, label):
    ax.plot(x, mean, color=colour, marker=marker, linestyle=linestyle, label=label)
    ax.fill_between(x, lo, hi, color=colour, alpha=0.15, linewidth=0)


def fig3_synthetic() -> Path | None:
    s1, s2, s3 = (_read("synthetic", f"EC_synthetic_{name}_summary") for name in ("S1", "S2", "S3"))
    if s1 is None or s2 is None or s3 is None:
        print("fig3: synthetic summaries missing; skipped")
        return None
    fig, axes = figure(2, 2, height=2.6)
    styles = series_styles()
    a = s1.filter(pl.col("mode") == "a_nonreference_test").sort("dangling_share")
    colour, marker, linestyle = next(styles)
    one_minus = 1.0 - np.array(a["rho_qc_mean"], dtype=float)
    axes[0, 0].plot(a["dangling_share"], one_minus, color=colour, marker=marker, linestyle=linestyle, label="S1")
    axes[0, 0].set_xlabel("dangling share")
    axes[0, 0].set_ylabel(r"$1-\rho$ (non-dangling test)")
    b = s2.filter(pl.col("ref_share_spread") == 0.01).sort("size_dispersion")
    for levels in sorted(set(b["levels"].to_list())):
        colour, marker, linestyle = next(styles)
        f = b.filter(pl.col("levels") == levels)
        _band(axes[0, 1], f["size_dispersion"], f["one_minus_rho_mean"], f["one_minus_rho_lo"], f["one_minus_rho_hi"], colour, marker, linestyle,
              f"S2, $C = {levels}$")
    axes[0, 1].set_xlabel("size dispersion")
    axes[0, 1].set_ylabel(r"$1-\rho$")
    ymax = max(float(np.nanmax(b["one_minus_rho_hi"].to_numpy())), float(np.nanmax(s3["rho_qc_all_hi"].to_numpy())) if s3.height else 0.0, 1e-6)
    axes[0, 0].set_ylim(-0.02 * ymax, ymax)
    for cfg in sorted(set(s3["config"].to_list())):                       # (c): the mean of 1 - rho is the solid line, the share dashed
        colour, marker, _ = next(styles)
        f = s3.filter((pl.col("config") == cfg) & (pl.col("kappa") == 1.0)).sort("partial_share")
        _band(axes[1, 0], f["partial_share"], 1.0 - np.array(f["rho_qc_all_mean"], dtype=float), 1.0 - np.array(f["rho_qc_all_hi"], dtype=float),
              1.0 - np.array(f["rho_qc_all_lo"], dtype=float), colour, marker, "-", f"S3 {SYNTHETIC_LABELS[cfg]}: $1-\\rho$")
        axes[1, 0].plot(f["partial_share"], f["partial_share_top_q_mean"], color=colour, marker=marker, linestyle="--", markerfacecolor="none",
                        label=f"S3 {SYNTHETIC_LABELS[cfg]}: QPPR top-5% share")
        for kappa in sorted(set(s3["kappa"].to_list())):
            colour2, marker2, linestyle2 = next(styles)
            g = s3.filter((pl.col("config") == cfg) & (pl.col("kappa") == kappa)).sort("partial_share")
            _band(axes[1, 1], g["partial_share"], g["d_prauc_mean"], g["d_prauc_lo"], g["d_prauc_hi"], colour2, marker2, linestyle2,
                  f"S3 {SYNTHETIC_LABELS[cfg]}, $\\kappa={kappa:g}$")
    axes[1, 0].set_xlabel("partially observed test share $m$")
    axes[1, 0].set_ylabel(r"$1-\rho$ / top-5% share")
    axes[1, 1].axhline(0.0, color=GREY, linewidth=0.8)
    axes[1, 1].set_xlabel("partially observed test share $m$")
    axes[1, 1].set_ylabel(r"$\Delta$PR-AUC (QPPR $-$ classical)")
    for ax, letter in zip(axes.ravel(), "abcd"):
        panel_letter(ax, letter)
    seen: dict[str, Line2D] = {}
    for ax in axes.ravel():                                                 # one legend below the panels, outside every band
        for handle, label in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(label, handle)
    fig.legend(list(seen.values()), list(seen.keys()), loc="outside lower center", ncol=3, fontsize=SMALL)
    out = _finish(fig, "fig3_synthetic.pdf")
    _spots("fig3", [("(a) 1-rho at share 0", one_minus[:1].tolist()),
                    ("(b) 1-rho C=24 dispersion 2", b.filter((pl.col("levels") == 24) & (pl.col("size_dispersion") == 2.0))["one_minus_rho_mean"].to_list()),
                    ("(c) rho at m=0.3 kappa=1", s3.filter((pl.col("partial_share") == 0.3) & (pl.col("kappa") == 1.0))["rho_qc_all_mean"].to_list())])
    return out


def _simulated() -> pl.Expr:
    """Rows of a simulated cell: ``status`` is empty or does not start with ``not simulated``."""
    return ~pl.col("status").cast(pl.String, strict=False).str.starts_with("not simulated").fill_null(False)


def _noise_grid(name: str, metric: str) -> pl.DataFrame | None:
    """The simulated rows of the assembled noise grid with ``metric`` defined; ``None``, saying why, when there are none."""
    grid = _read_datasets("EC_noise_grid")
    if grid is None:
        print(f"{name}: EC_noise_grid missing; skipped")
        return None
    if "status" in grid.columns:
        grid = grid.filter(_simulated())
    if metric not in grid.columns or not grid.filter(pl.col(metric).is_not_null()).height:
        print(f"{name}: EC_noise_grid has no simulated row with {metric}; skipped")
        return None
    return grid.filter(pl.col(metric).is_not_null())


def _entry_k(ds: str, arm: str) -> int | None:
    """``noise_k_band`` of ``arm`` for ``ds``, from the hardware header: the smallest ``k`` from which no instance carries signal; ``None`` when none."""
    header = _read(ds, "T6_hardware_header")
    column = f"noise_k_band_{arm}"
    if header is None or not header.height or column not in header.columns:
        return None
    try:
        return int(float(header[column].to_list()[0]))
    except (TypeError, ValueError):
        return None


def _longest(frame: pl.DataFrame, min_tau: int = 1) -> pl.DataFrame:
    """Per number of classes ``k``, the row of the largest recorded ``tau`` at or above ``min_tau``, in ``k`` order."""
    mine = frame.filter(pl.col("tau") >= min_tau)
    if not mine.height:
        return mine
    return mine.sort(["k_classes", "tau"]).group_by("k_classes", maintain_order=True).last().sort("k_classes")


def fig4_noise() -> Path | None:
    """Fidelity ratio of the raw simulated output against the number of classes ``k``, per dataset (rows) and noisy arm (columns).

    One line per recorded step ``tau`` (light to dark grey, one marker each) where that step was simulated, the readout-mitigated value at each
    ``k``'s longest simulated step dashed, and the pure-noise band of each ``k`` (judged at that longest walk, the step the signal rule reads)
    as one ribbon along ``k``. A short tick on the axis marks ``noise_k_band``, the smallest ``k`` from which no instance carries signal.
    """
    name = "fig4_noise.pdf"
    grid = _noise_grid(name, "fidelity_ratio")
    if grid is None:
        return None
    datasets = [(ds, label) for ds, label in DATASETS if grid.filter(pl.col("dataset") == ds).height]
    if not datasets:
        print(f"{name}: no dataset with a simulated instance; skipped")
        return None
    steps = [int(t) for t in DEFAULT_CONFIG.hardware_steps]
    min_tau = int(DEFAULT_CONFIG.hardware_min_tau)
    shades = matplotlib.colormaps["Greys"](np.linspace(0.3, 1.0, len(steps)))
    markers = ("o", "s", "^", "D", "v", "P", "X")
    band_columns = ("pure_noise_fidelity_lo", "pure_noise_fidelity_hi")
    has_band = all(column in grid.columns for column in band_columns)
    band_colour = PALETTE[5]
    fig, axes = figure(len(NOISE_ARMS), len(datasets), sharex=True, sharey="row")
    spots = []
    for i, (ds, label) in enumerate(datasets):
        for j, arm in enumerate(NOISE_ARMS):
            ax = axes[i, j]
            both = grid.filter((pl.col("dataset") == ds) & (pl.col("arm") == arm))
            raw = both.filter(~pl.col("mitigated"))
            if has_band:
                judged = _longest(raw, min_tau).filter(pl.col(band_columns[0]).is_not_null() & pl.col(band_columns[1]).is_not_null())
                if judged.height:
                    ax.fill_between(judged["k_classes"], judged[band_columns[0]], judged[band_columns[1]], color=band_colour, alpha=0.4, linewidth=0, zorder=1)
            for t, shade, marker in zip(steps, shades, markers):
                g = raw.filter(pl.col("tau") == t).sort("k_classes")
                if g.height:
                    ax.plot(g["k_classes"], g["fidelity_ratio"], color=shade, marker=marker, markersize=3.2, linewidth=1.0, zorder=3)
            mitigated = _longest(both.filter(pl.col("mitigated")))
            if mitigated.height:
                ax.plot(mitigated["k_classes"], mitigated["fidelity_ratio"], color="black", linestyle="--", linewidth=0.9, zorder=2)
            entry = _entry_k(ds, arm)
            if entry is not None:
                ax.plot([entry, entry], [0.0, STAR_TICK], transform=ax.get_xaxis_transform(), color="black", linewidth=2.2, solid_capstyle="butt", zorder=4)
            ax.axhline(0.0, color=GREY, linewidth=0.5, zorder=0)
            panel_letter(ax, "abcd"[i * len(NOISE_ARMS) + j])
            if i == 0:
                ax.set_title(ARM_LABELS[arm], loc="center", fontsize=FONT_PT)
            if arm == "fake_marrakesh" and raw.height:
                largest = int(raw["k_classes"].max())
                first = raw.filter((pl.col("k_classes") == largest) & (pl.col("tau") == 1))
                edges = first[band_columns[0]].to_list() + first[band_columns[1]].to_list() if has_band else []
                spots += [(f"{label} {arm} k={largest} tau=1 fidelity ratio", first["fidelity_ratio"].to_list()),
                          (f"{label} {arm} k={largest} tau=1 band edges", edges), (f"{label} {arm} noise_k_band", [entry if entry is not None else "none"])]
        axes[i, 0].set_ylabel(f"{label}\nfidelity ratio of the raw output")
    ks = sorted(set(grid["k_classes"].to_list()))
    ticks = [k for k in K_TICKS if ks[0] <= k <= ks[-1]]
    for ax in axes.ravel():
        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(k) for k in ticks])
        ax.minorticks_off()
    for ax in axes[-1]:
        ax.set_xlabel("classes $k$ of the retained instance")
    lo, hi = DEFAULT_CONFIG.hardware_noise_band_quantiles
    handles = [Line2D([], [], color=shade, marker=marker, markersize=3.2, linewidth=1.0, label=rf"$\tau = {t}$") for t, shade, marker in zip(steps, shades, markers)]
    handles += [Line2D([], [], color="black", linestyle="--", linewidth=0.9, label="readout-mitigated, longest simulated $\\tau$"),
                Patch(facecolor=band_colour, alpha=0.4, linewidth=0, label=f"pure-noise band ({lo:g}th to {hi:g}th percentile)"),
                Line2D([], [], color="black", marker="|", linestyle="none", markersize=9, markeredgewidth=2.2, label="$k$ from which no signal remains")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=SMALL)
    out = _finish(fig, name)
    _spots("fig4", spots[:4])
    return out


def noise_by_step(metric: str, name: str) -> Path | None:
    """Simulated arms against the exact lumped average at each recorded step, per dataset (rows) and noisy arm (columns), for ``K_SUBSET``.

    One colour and marker per number of classes ``k`` (a legend, not a colour bar), solid for the raw output and dashed for the readout-mitigated
    one, with each instance's own pure-noise band (its 5th--95th percentile of pure-noise outputs graded against its own exact average) shaded
    in the same colour.
    """
    grid = _noise_grid(name, metric)
    if grid is None:
        return None
    grid = grid.filter(pl.col("k_classes").is_in(list(K_SUBSET)))
    if not grid.height:
        print(f"{name}: no simulated instance among k in {K_SUBSET}; skipped")
        return None
    band_columns = ("pure_noise_spearman_lo", "pure_noise_spearman_hi") if metric == "spearman_supra" else ("pure_noise_fidelity_lo", "pure_noise_fidelity_hi")
    has_band = all(column in grid.columns for column in band_columns)
    datasets = [(ds, label) for ds, label in DATASETS if grid.filter(pl.col("dataset") == ds).height]
    sizes_all = [k for k in K_SUBSET if k in set(grid["k_classes"].to_list())]
    colours = {k: (PALETTE[i % len(PALETTE)], MARKERS[i % len(MARKERS)]) for i, k in enumerate(sizes_all)}
    fig, axes = figure(len(NOISE_ARMS), len(datasets), sharex=True, sharey="row")
    ylabel = "Spearman vs exact" if metric == "spearman_supra" else "fidelity ratio"
    spots = []
    for i, (ds, label) in enumerate(datasets):
        for j, arm in enumerate(NOISE_ARMS):
            ax = axes[i, j]
            both = grid.filter((pl.col("dataset") == ds) & (pl.col("arm") == arm))
            raw = both.filter(~pl.col("mitigated"))
            sizes = sorted(set(raw["k_classes"].to_list()))
            for k in sizes:
                colour, marker = colours[k]
                g = raw.filter(pl.col("k_classes") == k).sort("tau")
                band = g.filter(pl.col(band_columns[0]).is_not_null() & pl.col(band_columns[1]).is_not_null()) if has_band else g.clear()
                if band.height:
                    ax.fill_between(band["tau"], band[band_columns[0]], band[band_columns[1]], color=colour, alpha=0.12, linewidth=0)
                ax.plot(g["tau"], g[metric], color=colour, marker=marker, markersize=3.5, linewidth=1.0)
                mitigated = both.filter(pl.col("mitigated") & (pl.col("k_classes") == k)).sort("tau")
                if mitigated.height:
                    ax.plot(mitigated["tau"], mitigated[metric], color=colour, linestyle="--", linewidth=0.8)
            largest = max(sizes) if sizes else None
            if largest is not None:
                shots = grid.filter((pl.col("dataset") == ds) & (pl.col("arm") == "noiseless_shots") & (pl.col("k_classes") == largest)
                                    & ~pl.col("mitigated")).sort("tau")
                if shots.height:
                    ax.plot(shots["tau"], shots[metric], color=GREY, linewidth=0.6)
                if arm == "fake_marrakesh":
                    first = raw.filter((pl.col("k_classes") == largest) & (pl.col("tau") == 1))
                    edges = first[band_columns[0]].to_list() + first[band_columns[1]].to_list() if has_band else []
                    spots += [(f"{label} {arm} k={largest} tau=1 {metric}", first[metric].to_list()),
                              (f"{label} {arm} k={largest} tau=1 band edges", edges)]
            panel_letter(ax, "abcd"[i * len(NOISE_ARMS) + j])
            if i == 0:
                ax.set_title(ARM_LABELS[arm], loc="center", fontsize=FONT_PT)
        axes[i, 0].set_ylabel(f"{label}\n{ylabel}")
    for ax in axes[-1]:
        ax.set_xlabel(r"recorded steps $\tau$")
        ax.set_xticks(sorted(set(grid["tau"].to_list())))
    lo, hi = DEFAULT_CONFIG.hardware_noise_band_quantiles
    handles = [Line2D([], [], color=colour, marker=marker, markersize=3.5, linewidth=1.0, label=f"$k = {k}$") for k, (colour, marker) in colours.items()]
    handles += [Line2D([], [], color="black", linestyle="--", linewidth=0.8, label="readout-mitigated"),
                Line2D([], [], color=GREY, linewidth=0.6, label="noiseless shots, largest $k$"),
                Patch(facecolor=GREY, alpha=0.3, linewidth=0, label=f"pure-noise band of each $k$ ({lo:g}th to {hi:g}th percentile)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=SMALL)
    out = _finish(fig, name)
    _spots(name.removesuffix(".pdf"), spots[:3])
    return out


def ecfig_scaling() -> Path | None:
    frame = _read("runtime", "EC_scaling")
    if frame is None:
        print("ECfig_scaling: EC_scaling missing; skipped")
        return None
    fig, axes = figure(2)
    styles = series_styles()
    spots = []
    for cfg in sorted(set(frame["config"].to_list())):
        f = frame.filter(pl.col("config") == cfg).sort("n")
        colour, marker, linestyle = next(styles)
        name = SYNTHETIC_LABELS[cfg]
        axes[0, 0].plot(f["N"], f["t_lumped_limit"], color=colour, marker=marker, linestyle="-", label=f"{name}: lumped limit")
        full = f.filter(pl.col("t_full_limit").is_not_null())
        if full.height:
            axes[0, 0].plot(full["N"], full["t_full_limit"], color=colour, marker=marker, linestyle="--", markerfacecolor="none", label=f"{name}: full limit")
        axes[0, 1].plot(f["N"], f["R"], color=colour, marker=marker, linestyle=linestyle, label=name)
        spots += [(f"{cfg} R at the largest N", f["R"].to_list()[-1:]), (f"{cfg} lumped limit seconds at the largest N", f["t_lumped_limit"].to_list()[-1:])]
    for ax in axes[0]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$N$ (supra-nodes)")
    axes[0, 0].set_ylabel("time to compute (s)")
    axes[0, 1].set_ylabel("$R$ (blocks)")
    panel_letter(axes[0, 0], "a")
    panel_letter(axes[0, 1], "b")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=SMALL)
    out = _finish(fig, "ECfig_scaling.pdf")
    _spots("ECfig_scaling", spots[:3])
    return out


MAKERS = {"fig1": fig1_coupling, "fig2": fig2_convergence, "fig3": fig3_synthetic, "fig4": fig4_noise, "ECfig_scaling": ecfig_scaling,
          "ECfig_noise_spearman": lambda: noise_by_step("spearman_supra", "ECfig_noise_spearman.pdf"),
          "ECfig_noise_tv": lambda: noise_by_step("fidelity_ratio", "ECfig_noise_tv.pdf")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=sorted(MAKERS), action="append")
    args = parser.parse_args(argv)
    apply_style()
    manuscript_dir("figures")
    for name, maker in MAKERS.items():
        if args.only and name not in args.only:
            continue
        print(f"== {name}")
        out = maker()
        if out is not None:
            print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
