"""Table EC.1: the eight-individual counterexample to compatibility, and the exactness of H.

Eight individuals, two layers (employment, loan purpose), all with loan
purpose X; an employment edge iff ``J = 1``; two of each type
``(J, y~) in {(0, 1), (1, 1), (1, UNK), (0, UNK)}``; the ``y~ = 1``
individuals are the reference set; unit edges; ``alpha = 0.85``; ``d = v``;
``v = 1/8`` on each reference copy; the multilayer (combined) restart against
the rank-one restart, at interlayer weight ``eta = 1`` and ``eta = 0``;
``Psi_0 = A sqrt(v)``; the time average over ``t = 0 .. 99``.

For each of the four cases the script reports the compatibility residual,
the route, and ``e100``, the largest absolute difference between the
reduced and the unlumped individual time averages at ``tau = 100`` for the
lumped discriminant ``D-bar`` and for the compressed discriminant ``H``.

    python -m scripts.ec_counterexample          # writes outputs/results/synthetic/EC_counterexample.csv and a manifest
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_CONFIG  # noqa: E402
from evaluation.provenance import RunTimer, write_manifest, write_result  # noqa: E402
from paths import stage_dir  # noqa: E402
from quantum.lumped_operator import personalization_vector, unfold_class_scores  # noqa: E402
from quantum.lumping import UNKNOWN_LABEL, build_lumped_subgraph  # noqa: E402
from quantum.lumping_router import discriminant, route_graphs  # noqa: E402
from quantum.mechanism.running_average import step_marginals  # noqa: E402
from quantum.mechanism.variants import COMBINED_RESTART, RANK_ONE_RESTART  # noqa: E402

TABLE_NAME = "EC_counterexample"
STAGE = "07_counterexample"
SCOPE = "synthetic"
TAU = 100
ALPHA = 0.85
LAYERS = [("JOB", ["JOB_J1", "JOB_Missing"]), ("REASON", ["REASON_X", "REASON_Missing"])]
# (J, y~) per individual; y~ = 1 is a reference individual, UNK a masked one.
TYPES = [(0, 1), (0, 1), (1, 1), (1, 1), (1, UNKNOWN_LABEL), (1, UNKNOWN_LABEL), (0, UNKNOWN_LABEL), (0, UNKNOWN_LABEL)]
CASES = ((1.0, COMBINED_RESTART), (1.0, RANK_ONE_RESTART), (0.0, COMBINED_RESTART), (0.0, RANK_ONE_RESTART))
# The values the manuscript's Table EC.1 reports (e100 under D-bar / compatibility residual).
EXPECTED = {
    (1.0, COMBINED_RESTART): (5.83e-16, 5.55e-17),
    (1.0, RANK_ONE_RESTART): (4.08e-4, 7.32e-4),
    (0.0, COMBINED_RESTART): (4.29e-3, 2.14e-2),
    (0.0, RANK_ONE_RESTART): (2.39e-15, 1.11e-16),
}


def frame() -> tuple[pd.DataFrame, np.ndarray]:
    """The eight-individual frame and its masked labels."""
    rows = []
    for j, label in TYPES:
        rows.append({"JOB_J1": int(j == 1), "JOB_Missing": int(j == 0), "REASON_X": 1, "REASON_Missing": 0,
                     "BAD": int(label == 1)})
    levels = np.array([label for _j, label in TYPES], dtype=int)
    return pd.DataFrame(rows), levels


def individual_average(nodes: pd.DataFrame, G: np.ndarray, v: np.ndarray, K: np.ndarray) -> np.ndarray:
    """The time average over ``t = 0 .. TAU - 1`` of the source marginal, summed over each individual's copies."""
    history = step_marginals(G, v, K, TAU)
    _ids, scores = unfold_class_scores(nodes, history.mean(axis=0))
    return scores


def compute_rows(router_tol: float = DEFAULT_CONFIG.router_tol) -> list[dict[str, Any]]:
    df, levels = frame()
    ids = list(range(len(df)))
    rows: list[dict[str, Any]] = []
    for eta, mode in CASES:
        nodes, edges = build_lumped_subgraph(df, LAYERS, ids, lumping_level="attr_bad", stickiness=eta, bad_masked=levels)
        nodes_f, edges_f = build_lumped_subgraph(df, LAYERS, ids, lumping_level="none", stickiness=eta, bad_masked=levels)
        rank_one = mode == RANK_ONE_RESTART
        routed = route_graphs(
            nodes, edges, nodes_f, edges_f, alpha=ALPHA, influence_type="combined", router_tol=router_tol,
            personalization_mode=mode, keep_full=True,
            personalization_lumped=personalization_vector(nodes) if rank_one else None,
            personalization_full=personalization_vector(nodes_f) if rank_one else None,
        )
        full = individual_average(nodes_f, routed.G_full, routed.v_full, discriminant(routed.G_full))
        e_dbar = float(np.max(np.abs(individual_average(nodes, routed.G, routed.v, routed.D_bar) - full)))
        e_h = float(np.max(np.abs(individual_average(nodes, routed.G, routed.v, routed.H) - full)))
        expected_e, expected_r = EXPECTED[(eta, mode)]
        rows.append({
            "eta": float(eta), "restart": "multilayer" if mode == COMBINED_RESTART else "rank-one",
            "n_full": int(routed.residuals["n_full"]), "n_reduced": int(routed.residuals["n_reduced"]),
            "lumpability_residual": float(routed.residuals["lumpability_residual"]),
            "compatibility_residual": float(routed.residuals["compatibility_residual"]),
            "route": routed.route, "e100_dbar": e_dbar, "e100_h": e_h, "tau": TAU,
            "expected_e100_dbar": expected_e, "expected_compatibility_residual": expected_r,
        })
    return rows


def write(rows) -> Path:
    """``outputs/results/synthetic/EC_counterexample.csv``."""
    return write_result(
        SCOPE, TABLE_NAME, pl.DataFrame(rows), stage=STAGE,
        columns={"eta": "interlayer weight", "restart": "multilayer (combined) or rank-one restart",
                 "n_full": "supra-nodes of the unlumped operator", "n_reduced": "supra-nodes of the lumped operator",
                 "lumpability_residual": "max |C G - G_bar C|", "compatibility_residual": "max |D B - B D_bar|",
                 "route": "discriminant the router selected",
                 "e100_dbar": "max |reduced - full| individual time average at tau = 100 under K = D_bar",
                 "e100_h": "the same under K = H", "tau": "horizon of the time average",
                 "expected_e100_dbar": "e100 under D_bar as the manuscript's Table EC.1 reports it",
                 "expected_compatibility_residual": "compatibility residual as Table EC.1 reports it"},
    )


def main() -> int:
    timer = RunTimer()
    rows = compute_rows()
    path = write(rows)
    for row in rows:
        print(f"eta={row['eta']:g} {row['restart']:<10} route={row['route']:<5} "
              f"residual={row['compatibility_residual']:.3e} e100(D-bar)={row['e100_dbar']:.3e} "
              f"e100(H)={row['e100_h']:.3e}")
    manifest = write_manifest(stage_dir(STAGE), arm_name=STAGE, config=DEFAULT_CONFIG, inputs=[], outputs=[path], seeds={},
                              timer=timer, execution_mode="simulation", extra={"tau": TAU, "alpha": ALPHA, "cases": len(rows)})
    print(path)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
