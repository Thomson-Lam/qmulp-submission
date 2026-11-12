"""The single-layer proposition on the real networks: the classical score is a function of the reference share.

With one layer and ``d = v``, every non-reference member of category ``c``
with reference share ``f_c = s_c / n_c`` -- ``n_c`` counting development
references, development non-references and masked test members -- has
stationary score

    alpha f_c R / [(1 - alpha)(1 + alpha - alpha f_c)],
    R = (alpha pi_dangling + (1 - alpha)(1 - S)) / |D|,

where ``S = pi(D)`` is the stationary mass on the reference copies,
``pi_dangling`` the stationary mass on the individual copies with no
membership edge, and ``R`` is common to every category. Derivation (``d = v``,
the own-copy restart of a reference copy, ``x_i`` the score of individual
``i`` and ``X_c`` the total of category ``c``): a category node holds
``alpha X_c``; a non-reference member holds ``alpha^2 X_c / n_c``; a reference
member holds ``alpha X_c / n_c + R / alpha`` where every dangling column
(whose out-mass goes to ``v``) and every non-reference column (whose restart
goes to ``v``) add ``(alpha pi_dangling + (1 - alpha)(1 - S)) / |D|`` per
reference copy. Summing the members gives ``X_c`` and then the formula. A
dangling *reference* has mass ``R / alpha`` and so enters ``pi_dangling``; a
non-reference individual without a membership edge has score zero and is
inert in both arms (``v`` is zero on it and nothing flows in). The earlier
constant omitted the ``alpha pi_dangling`` term and was therefore exact only
when no reference individual lacks the category. The QPPR limit depends on
``n_c`` as well, so two categories with equal shares and different sizes get
one classical level and two QPPR levels. Per category this module records
``n_c``, ``s_c``, ``f_c``, the test members, the classical score of a test
member, the formula's value, their relative error, the QPPR limit of a test
member, and the ranks by ``f_c``, by the classical score and by QPPR.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import rankdata

from evaluation.context import DatasetContext, ScoredOperator
from evaluation.provenance import write_result
from evaluation.ties import canonicalize, distinct_levels
from quantum.lumping import UNKNOWN_LABEL
from quantum.mechanism.variants import MEMBERSHIP, UNRECORDED_FAMILY

TABLE_NAME = "EC_single_layer_share"
EQUAL_SHARE_TOL = 1e-12


def formula(alpha: float, f: float, R: float) -> float:
    """``alpha f R / [(1 - alpha)(1 + alpha - alpha f)]``."""
    return float(alpha * f * R / ((1.0 - alpha) * (1.0 + alpha - alpha * f)))


def category_table(scored: ScoredOperator, alpha: float, decimals: int) -> list[dict[str, Any]]:
    """One record per category (plus the unrecorded family) of a single-layer scored operator."""
    nodes, edges = scored.variant.nodes, scored.variant.edges
    pi = scored.classical_positions
    omega = scored.quantum.limit.omega
    copies = nodes[nodes["node_type"].astype(str) == "borrower_copy"]
    position = {int(n): p for p, n in enumerate(nodes["node_id"].astype(int))}
    size = {int(r.node_id): int(round(float(r.weight))) for r in copies.itertuples(index=False)}
    level = {int(r.node_id): int(r.BAD_masked) for r in copies.itertuples(index=False)}
    attached: dict[int, list[int]] = {}
    for s, t, rel in zip(edges["source"], edges["target"], edges["relation"]):
        if str(rel).endswith(MEMBERSHIP):
            attached.setdefault(int(t), []).append(int(s))
    seed_mass = float(sum(pi[position[n]] for n in copies["node_id"].astype(int) if level[int(n)] == 1))
    n_ref = int(sum(size[n] for n in size if level[n] == 1))
    # Individual copies with no membership edge: their columns are dangling (d = v). A dangling reference
    # carries mass R / alpha, so this mass enters the constant common to every category (Prop. 3b).
    attached_copies = {m for ms in attached.values() for m in ms}
    loose = [int(n) for n in copies["node_id"].astype(int) if int(n) not in attached_copies]
    pi_dangling = float(sum(pi[position[m]] for m in loose))
    R = (alpha * pi_dangling + (1.0 - alpha) * (1.0 - seed_mass)) / n_ref
    records = []
    categories = nodes[nodes["node_type"].astype(str) == "category"]
    for cat in categories.itertuples(index=False):
        members = attached.get(int(cat.node_id), [])
        n_c = sum(size[m] for m in members)
        s_c = sum(size[m] for m in members if level[m] == 1)
        tests = [m for m in members if level[m] == UNKNOWN_LABEL]
        n_test = sum(size[m] for m in tests)
        per_member_c = float(pi[position[tests[0]]] / size[tests[0]]) if tests else float("nan")
        per_member_q = float(omega[position[tests[0]]] / size[tests[0]]) if tests else float("nan")
        f_c = s_c / n_c if n_c else float("nan")
        value = formula(alpha, f_c, R) if n_c else float("nan")
        records.append({"category": str(cat.label), "n_c": int(n_c), "s_c": int(s_c), "f_c": f_c, "n_test": int(n_test),
                        "classical_test_member": per_member_c, "formula": value,
                        "rel_error": abs(per_member_c - value) / value if (tests and value > 0) else float("nan"),
                        "qppr_test_member": per_member_q, "is_dangling_family": False})
    if loose:
        tests = [m for m in loose if level[m] == UNKNOWN_LABEL]
        records.append({"category": UNRECORDED_FAMILY, "n_c": int(sum(size[m] for m in loose)),
                        "s_c": int(sum(size[m] for m in loose if level[m] == 1)), "f_c": float("nan"),
                        "n_test": int(sum(size[m] for m in tests)),
                        "classical_test_member": float(pi[position[tests[0]]] / size[tests[0]]) if tests else float("nan"),
                        "formula": 0.0, "rel_error": float("nan"),
                        "qppr_test_member": float(omega[position[tests[0]]] / size[tests[0]]) if tests else float("nan"),
                        "is_dangling_family": True})
    _add_ranks(records, decimals)
    for r in records:
        r["R"] = R
        r["seed_mass"] = seed_mass
        r["pi_dangling"] = pi_dangling
        r["n_ref"] = n_ref
    return records


def _add_ranks(records: list[dict[str, Any]], decimals: int) -> None:
    for key, name in (("f_c", "rank_by_f"), ("classical_test_member", "rank_by_classical"), ("qppr_test_member", "rank_by_qppr")):
        values = np.array([r[key] for r in records], dtype=float)
        out = np.full(values.size, np.nan)
        present = np.isfinite(values)
        if present.any():
            out[present] = rankdata(-canonicalize(values[present], decimals), method="average")
        for r, rank in zip(records, out):
            r[name] = float(rank)


def equal_share_pairs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Category pairs with ``|f_c - f_c'| < 1e-12`` and their classical and QPPR values."""
    genuine = [r for r in records if not r["is_dangling_family"] and np.isfinite(r["f_c"])]
    pairs = []
    for i, a in enumerate(genuine):
        for b in genuine[i + 1:]:
            if abs(a["f_c"] - b["f_c"]) < EQUAL_SHARE_TOL:
                pairs.append({"category_a": a["category"], "category_b": b["category"], "f": a["f_c"],
                              "n_a": a["n_c"], "n_b": b["n_c"], "classical_a": a["classical_test_member"],
                              "classical_b": b["classical_test_member"], "qppr_a": a["qppr_test_member"],
                              "qppr_b": b["qppr_test_member"]})
    return pairs


def single_layer_rows(ctx: DatasetContext, network: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Category records of one single-layer network and the summary keys."""
    spec = ctx.specs[network]
    if spec.n_layers != 1:
        raise ValueError(f"{network} is not a single-layer network")
    scored = ctx.primary(network)
    records = category_table(scored, ctx.config.resolved_alpha(), ctx.decimals)
    rows = [ctx.row(network, **r) for r in records]
    q, c = ctx.evaluate(scored.quantum.omega_source), ctx.evaluate(scored.classical)
    errors = [r["rel_error"] for r in records if np.isfinite(r["rel_error"])]
    pairs = equal_share_pairs(records)
    summary = {"network": network, "prop3_max_relerr": float(max(errors)) if errors else float("nan"),
               "equal_share_pairs": "; ".join(f"{p['category_a']}/{p['category_b']}" for p in pairs) or "none",
               "n_equal_share_pairs": len(pairs), "levels_q": distinct_levels(q, ctx.decimals),
               "levels_c": distinct_levels(c, ctx.decimals), "pairs": pairs}
    summary["levels_q_minus_c"] = summary["levels_q"] - summary["levels_c"]
    return rows, summary


def write_single_layer_table(ctx: DatasetContext, rows, *, source_stage: str) -> Path:
    """``EC_single_layer_share`` under the dataset's scope."""
    return write_result(
        ctx.name, TABLE_NAME, pl.DataFrame(list(rows), strict=False, infer_schema_length=None), stage=source_stage,
        columns={"dataset": "dataset name", "network": "single-layer network name",
                 "category": "category of the layer (the unrecorded family is the individuals without a membership edge)",
                 "n_c": "members: development references, development non-references and masked test members",
                 "s_c": "development references in the category", "f_c": "reference share s_c / n_c", "n_test": "masked test members",
                 "classical_test_member": "classical stationary score of one test member",
                 "formula": "value of the single-layer formula alpha f_c R / [(1 - alpha)(1 + alpha - alpha f_c)]",
                 "rel_error": "|classical - formula| / formula", "qppr_test_member": "QPPR limit of one test member",
                 "is_dangling_family": "whether the row is the unrecorded family",
                 "rank_by_f": "rank of the category by reference share (1 = largest)", "rank_by_classical": "rank by the classical score",
                 "rank_by_qppr": "rank by the QPPR limit", "R": "the constant R = (alpha pi_dangling + (1 - alpha)(1 - seed_mass)) / n_ref common to every category",
                 "seed_mass": "stationary mass S on the reference copies",
                 "pi_dangling": "stationary mass on the individual copies with no membership edge (dangling references included)",
                 "n_ref": "development references |D|"},
    )
