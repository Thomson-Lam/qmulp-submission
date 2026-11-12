"""The one number formatter, the one label mapping and the one bare-tabular renderer of the repository.

Every printed number -- in ``numbers.tex`` through ``evaluation.number_map``
and in ``outputs/manuscript/tables/*.tex`` through
``scripts.make_manuscript_tables`` -- goes through :func:`format_value` with
one of the rules below. A second formatter anywhere is a defect.

Rules
-----
``corr``     correlations, AUCs, shares, probabilities: three decimals
``diff``     signed differences: sign and three decimals
``count``    integers with a ``{,}`` thousands separator
``int``      small integers, no separator
``real``     plain three decimals; ``$a\\times10^{b}$`` only below ``1e-3``
``mean``     means and ranks: one plain decimal
``seconds``  a time in seconds to three significant figures (``299``, ``0.0310``), ``{,}`` above a thousand,
             ``$a\\times10^{b}$`` below ``1e-3``
``mb``       memory in MB: an integer with ``{,}``
``small``    always ``$a\\times10^{b}$`` (residuals, errors); zero prints ``0``
``tau``      ``tau*``: an integer with ``{,}``, or ``>\\,$\\tau_{\\max}$ = 50{,}000``
``text``     text, through :func:`human_label`, escaped outside math
``route``    the discriminant the router selected, as ``$\\bar{D}$`` or ``$H$`` (:data:`ROUTE_LABELS`)

A sign is typeset as math (``$-$0.019``, ``$+$0.001``), never as a text hyphen. Text escaping touches only what
lies outside ``$...$``: a header such as ``$\\Delta$PR-AUC`` keeps its math and gets its text escaped. No code
identifier reaches a table or ``numbers.tex``: a value that is a known identifier prints its label from
:data:`TEXT_LABELS`, the one mapping.
"""

from __future__ import annotations

import math
import re
from typing import Any, NamedTuple, Sequence

RULES: tuple[str, ...] = ("corr", "diff", "count", "int", "real", "mean", "seconds", "mb", "small", "tau", "text", "route")
MISSING = "--"
# What the router's verdict prints as: one mapping for the tables and the number map, so a caption that names a
# route is generated from the route column (a macro of the number map), never typed.
ROUTE_LABELS = {"D_bar": r"$\bar{D}$", "H": "$H$", "classical": "--"}
ABOVE_MAX_PREFIX = ">"

# Labels shared by the tables and the figures.
DATASET_LABELS = {"hmeq": "HMEQ", "fmac": "Freddie Mac"}
NETWORK_LABELS = {"NET_JOB": "employment", "NET_REASON": "loan purpose", "NET_2L": "two-layer",
                  "NET_FMAC_POSTAL": "postal code", "NET_FMAC_SELLER": "seller", "NET_FMAC_2L": "two-layer"}
LAYER_LABELS = {"JOB": "employment", "REASON": "loan purpose", "PostalCode": "postal code", "SellerName": "seller"}
SYNTHETIC_LABELS = {"hmeq_like": "HMEQ-like", "fmac_like": "Freddie Mac-like"}
ARM_LABELS = {"qppr": "QPPR", "classical": "classical", "qppr_limit": "QPPR", "exact": "exact", "noiseless_shots": "noiseless shots",
              "aer_noise_model": "Aer noise model", "fake_marrakesh": "FakeMarrakesh", "qppr_minus_classical": "QPPR $-$ classical"}
VARIANT_LABELS = {"primary": "primary", "own_copy_restart": "own-copy restart", "rank_one_restart": "rank-one restart (control)",
                  "flattened": "flattened", "eta_0": r"$\eta = 0$", "no_partial_copies": "no partial copies",
                  "best_classical_alpha_eta": r"best classical $(\alpha, \eta)$"}

# The one mapping from a code identifier, or a label a stage wrote, to the words the manuscript prints. A value that is
# not a key prints as it is (escaped); the patterns of :func:`human_label` cover the families (layers, instance sizes).
# The families above (datasets, networks, layers, synthetic configurations, arms, variants) are part of it, so a table cell,
# a group title and a number-map value that holds one of their codes all print the same words.
TEXT_LABELS = {
    **DATASET_LABELS, **NETWORK_LABELS, **LAYER_LABELS, **SYNTHETIC_LABELS, **ARM_LABELS, **VARIANT_LABELS,
    # the competing scores of the decision table (best_comp_name) and the labels the decision stage wrote for them
    "blank_field_count": "blank-field count", "one_hop_seed_count": "one-hop reference count", "cell_rate": "joint-category default rate",
    "logit_on_layer_dummies": "logit on layer indicators", "one-hop seed count": "one-hop reference count",
    "logit on layer dummies": "logit on layer indicators", "QPPR minus classical (paired)": "QPPR $-$ classical (paired)",
    "no_skill": "no skill",
    # the strata of the strata table
    "fully_recorded": "fully recorded", "any_blank_field": "any blank field", "seller_known": "seller known", "seller_unseen": "seller unseen",
    # the feature sets of the downstream check
    "covariates_only": "covariates only", "plus_classical": "plus classical", "plus_qppr_limit": "plus QPPR limit",
    # the operations of the runtime table
    "full_operator_build": "full operator build", "class_construction_lumped_operator_residuals": "class construction and residuals",
    "full_walk_per_step": "full walk, per step", "lumped_walk_per_step": "lumped walk, per step",
    "limit_lumped_eigendecomposition": "limit, lumped eigendecomposition", "limit_full_eigendecomposition": "limit, full eigendecomposition",
    "classical_stationary_full_sparse": "classical stationary, full (sparse)", "classical_stationary_lumped_dense": "classical stationary, lumped (dense)",
    # the status of a scaling row whose full eigendecomposition was skipped
    "not run (N > FULL_EIG_MAX)": "not run",
}
_LABEL_PATTERNS = (
    (re.compile(r"single_layer_seed_default_rate_(\w+)"), lambda m: f"single-layer reference share ({layer_label(m.group(1))})"),
    (re.compile(r"single-layer reference share \((\w+)\)"), lambda m: f"single-layer reference share ({layer_label(m.group(1))})"),
    (re.compile(r"cell default rate \(.*\)"), lambda m: "joint-category default rate"),
    (re.compile(r"top_(\d+)_classes"), lambda m: f"top {m.group(1)} classes"),
)
# Characters escaped outside math. A backslash outside math is left alone: it
# is a LaTeX command the author wrote on purpose, never data.
_ESCAPES = (("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"))
_ABOVE_CAP = re.compile(r">\s*tau_max\s*\(\s*(\d+)\s*\)")


def layer_label(name: str) -> str:
    """The words for a layer (a column of the dataset): ``JOB`` -> ``employment``."""
    return LAYER_LABELS.get(str(name).strip(), str(name).strip())


def human_label(value: Any) -> str:
    """The printed words for a code identifier or a stage label; any other text is returned as it is."""
    text = str(value).strip()
    if text in TEXT_LABELS:
        return TEXT_LABELS[text]
    for pattern, label in _LABEL_PATTERNS:
        found = pattern.fullmatch(text)
        if found:
            return label(found)
    return text


def _strip_prefix(rest: str, layer: str) -> str:
    """``JOB_Mgr`` -> ``Mgr``: the one-hot column's own layer prefix is dropped, underscores become spaces."""
    return (rest[len(layer) + 1:] if rest.startswith(f"{layer}_") else rest).replace("_", " ")


def category_label(text: Any) -> str:
    """A category of a single-layer table (``JOB:JOB_Mgr``, ``PostalCode:PostalCode_321``, ``unrecorded``) without its layer prefix."""
    layer, sep, rest = str(text).strip().partition(":")
    return _strip_prefix(rest, layer) if sep else str(text).strip()


def node_label(text: Any) -> str:
    """A supra-node of an instance: ``JOB:group_23`` -> ``employment, class 23``, ``JOB:JOB_Other`` -> ``employment: Other``."""
    layer, sep, rest = str(text).strip().partition(":")
    if not sep:
        return human_label(text)
    cls = re.fullmatch(r"group_(\d+)", rest)
    return f"{layer_label(layer)}, class {cls.group(1)}" if cls else f"{layer_label(layer)}: {_strip_prefix(rest, layer)}"


def tex_escape(text: Any) -> str:
    """Escape ``& % # _`` outside ``$...$``; math segments are written verbatim."""
    s = str(text)
    parts = s.split("$")
    if len(parts) % 2 == 0:
        raise ValueError(f"unbalanced $ in {s!r}")
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            out.append(part)
            continue
        for old, new in _ESCAPES:
            part = part.replace(old, new)
        out.append(part)
    return "$".join(out)


def is_missing(value: Any) -> bool:
    """Whether there is nothing to print: ``None``, a non-finite float, or a blank or ``nan``/``none``/``null`` string."""
    if value is None:
        return True
    if isinstance(value, float) and not math.isfinite(value):
        return True
    if isinstance(value, str) and value.strip().lower() in ("", "nan", "none", "null"):
        return True
    return False


def coerce_value(value: Any) -> Any:
    """A value read back from a CSV or a JSON file: a number where the text is one, ``None`` where it is blank.

    A string is stripped; the empty string and ``null``, ``none``, ``nan`` in any case are blank; a string holding a
    ``.`` or an ``e``, or ``inf`` / ``-inf``, is read as a float, any other number as an int, and a string that is no
    number is returned stripped. A value that is not a string is returned as it is. The one coercion of the
    repository: the table builders read stacked text frames through it and the number map its CSV and JSON sources.
    """
    if isinstance(value, str):
        s = value.strip()
        if s == "" or s.lower() in ("null", "none", "nan"):
            return None
        try:
            return float(s) if ("." in s or "e" in s.lower() or s in ("inf", "-inf")) else int(s)
        except ValueError:
            return s
    return value


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return float(int(value))
    return float(value)


def _signed(text: str) -> str:
    """A typeset number: a leading minus or plus sign is a math sign (``$-$0.019``), never a text hyphen."""
    return f"${text[0]}${text[1:]}" if text and text[0] in "-+" else text


def sci(value: float, digits: int = 2) -> str:
    """``$a\\times10^{b}$`` with ``digits`` decimals in the mantissa; zero is ``0``."""
    v = float(value)
    if v == 0.0:
        return "0"
    exponent = int(math.floor(math.log10(abs(v))))
    mantissa = v / 10**exponent
    if round(abs(mantissa), digits) >= 10.0:
        exponent += 1
        mantissa = v / 10**exponent
    return rf"${mantissa:.{digits}f}\times10^{{{exponent}}}$"


def grouped_int(value: float) -> str:
    return f"{int(round(float(value))):,}".replace(",", "{,}")


def seconds_text(v: float) -> str:
    """Three significant figures: ``299``, ``2.31``, ``0.0310``; ``{,}`` from a thousand; scientific below ``1e-3``; zero is ``0``."""
    if v == 0.0:
        return "0"
    if abs(v) < 1e-3:
        return sci(v)
    decimals = max(0, 2 - int(math.floor(math.log10(abs(v)))))
    return _signed(f"{v:,.{decimals}f}".replace(",", "{,}"))


def format_value(value: Any, rule: str) -> str | None:
    """The printed form of ``value`` under ``rule``; ``None`` when there is nothing to print."""
    if rule not in RULES:
        raise ValueError(f"unknown formatting rule {rule!r}; expected one of {RULES}")
    if rule == "text":
        # A text value keeps its string as written: "none" is a legitimate entry (no equal-share
        # pairs, no seller found). Only an absent value or a blank string is nothing to print.
        if value is None or (isinstance(value, float) and not math.isfinite(value)) or str(value).strip() == "":
            return None
        return tex_escape(human_label(value))
    if rule == "route":
        if is_missing(value):
            return None
        key = str(value).strip()
        return ROUTE_LABELS.get(key, tex_escape(key))
    if is_missing(value):
        return None
    if rule == "tau":
        if isinstance(value, str) and value.strip().startswith(ABOVE_MAX_PREFIX):
            text = value.strip()
            cap = _ABOVE_CAP.fullmatch(text)
            if cap:
                return ">\\,$\\tau_{\\max}$ = " + grouped_int(int(cap.group(1)))
            return ">\\," + text[len(ABOVE_MAX_PREFIX):].strip().replace("tau_max", "$\\tau_{\\max}$")
        return grouped_int(_number(value))
    v = _number(value)
    if rule == "count":
        return _signed(grouped_int(v))
    if rule == "int":
        return _signed(str(int(round(v))))
    if rule == "corr":
        return _signed(f"{v:.3f}")
    if rule == "diff":
        return _signed(f"{v:+.3f}")
    if rule == "mean":
        return _signed(f"{v:.1f}")
    if rule == "seconds":
        return seconds_text(v)
    if rule == "mb":
        return _signed(grouped_int(v))
    if rule == "small":
        return sci(v)
    # real
    if 0.0 < abs(v) < 1e-3:
        return sci(v)
    return _signed(f"{v:.3f}")


def format_cell(value: Any, rule: str) -> str:
    """A table cell: the formatted value or ``--``."""
    printed = format_value(value, rule)
    return MISSING if printed is None else printed


def ci_cell(point: Any, lo: Any, hi: Any, rule: str) -> str:
    """``point [lo, hi]``; the bare point when either bound is missing; ``--`` when the point is."""
    centre = format_value(point, rule)
    if centre is None:
        return MISSING
    low, high = format_value(lo, rule), format_value(hi, rule)
    if low is None or high is None:
        return centre
    return f"{centre} [{low}, {high}]"


_COLUMN = re.compile(r"p\{[^}]*\}|[lrc]")
_SPAN = re.compile(r"\s*\\multicolumn\{(\d+)\}\{")


def grid_columns(align: str) -> list[str]:
    """The column specifiers of an alignment string: ``l``, ``r``, ``c`` and ``p{width}``."""
    tokens = _COLUMN.findall(align)
    if "".join(tokens) != align:
        raise ValueError(f"alignment {align!r} holds something other than l, r, c and p{{width}}")
    return tokens


def cell_span(cell: str) -> int:
    """The number of grid columns a cell occupies: ``k`` for ``\\multicolumn{k}{..}{..}``, else one."""
    found = _SPAN.match(cell)
    return int(found.group(1)) if found else 1


def spanned(text: str, span: int = 1, align: str | None = None) -> str:
    """``text`` over ``span`` grid columns with an explicit alignment (``r`` by default); a plain cell when it is one column in the grid's own alignment."""
    if span == 1 and align is None:
        return text
    return rf"\multicolumn{{{span}}}{{{align or 'r'}}}{{{text}}}"


class Block(NamedTuple):
    """One panel of a stacked table: an optional title, its header cells, its rows and the dataset groups of those rows."""

    title: str | None
    headers: Sequence[str]
    rows: Sequence[Sequence[str]]
    groups: Sequence[tuple[str, int]] | None = None


def _grid_count(cells: Sequence[str]) -> int:
    return sum(cell_span(cell) for cell in cells)


def tabular(
    headers: Sequence[str] | None = None,
    rows: Sequence[Sequence[str]] | None = None,
    *,
    align: str | None = None,
    groups: Sequence[tuple[str, int]] | None = None,
    panels: Sequence[Block] | None = None,
) -> str:
    """A bare booktabs ``tabular``: no ``table`` environment, caption, label or notes.

    ``headers`` are written verbatim (escape them with :func:`tex_escape`
    first); every row must have as many cells as there are grid columns, a
    ``\\multicolumn{k}{..}{..}`` cell counting for ``k``. ``groups`` is an
    optional sequence of ``(label, n_rows)`` consecutive blocks; each block is
    introduced by a ``\\midrule`` and a full-width italic label row.

    ``panels`` stacks several :class:`Block` s in one tabular (``headers``, ``rows`` and ``groups`` are then not
    given): each panel opens with a full-width bold title row (when it has a title) and its own header row, and a
    ``\\midrule`` separates the panels. Panels share the grid of columns, so a wide cell of one panel spans
    several narrow columns of another.
    """
    if panels is None:
        if headers is None or rows is None:
            raise ValueError("a tabular needs headers and rows, or panels")
        blocks = [Block(None, headers, rows, groups)]
    else:
        if headers is not None or rows is not None or groups is not None:
            raise ValueError("headers, rows and groups belong to the panels when panels are given")
        blocks = list(panels)
    if not blocks:
        raise ValueError("a tabular needs at least one panel")
    n = len(grid_columns(align)) if align else _grid_count(blocks[0].headers)
    if n == 0:
        raise ValueError("a tabular needs at least one column")
    spec = align or ("l" + "r" * (n - 1))
    if len(grid_columns(spec)) != n:
        raise ValueError(f"alignment {spec!r} does not match {n} columns")
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule"]
    for index, block in enumerate(blocks):
        if _grid_count(block.headers) != n:
            raise ValueError(f"header has {_grid_count(block.headers)} grid cells for {n} columns: {list(block.headers)!r}")
        for row in block.rows:
            if _grid_count(row) != n:
                raise ValueError(f"row has {_grid_count(row)} grid cells for {n} columns: {row!r}")
        if index > 0:
            lines.append(r"\midrule")
        if block.title:
            lines.append(rf"\multicolumn{{{n}}}{{l}}{{\textbf{{{tex_escape(block.title)}}}}} \\")
        lines += [" & ".join(block.headers) + r" \\", r"\midrule"]
        if block.groups:
            if sum(size for _label, size in block.groups) != len(block.rows):
                raise ValueError("group sizes do not add up to the number of rows")
            cursor = 0
            for group_index, (label, size) in enumerate(block.groups):
                if group_index > 0:
                    lines.append(r"\midrule")
                lines.append(rf"\multicolumn{{{n}}}{{l}}{{\textit{{{tex_escape(label)}}}}} \\")
                for row in block.rows[cursor:cursor + size]:
                    lines.append(" & ".join(row) + r" \\")
                cursor += size
        else:
            for row in block.rows:
                lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"
