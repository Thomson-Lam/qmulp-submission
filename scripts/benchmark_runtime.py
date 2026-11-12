"""Time to compute, on one node: the full and the lumped route for each dataset, and synthetic scaling.

    python -m scripts.benchmark_runtime --dataset hmeq --threads 1 8
    python -m scripts.benchmark_runtime --dataset fmac --threads 1 8
    python -m scripts.benchmark_runtime --scaling                # synthetic n = 1e3 .. 1e5
    python -m scripts.benchmark_runtime --dataset hmeq --smoke

Per dataset on its two-layer network, median of ``--repeats`` runs at each
thread count, with the peak traced memory of the step: (a) the full operator
build; the class construction, the lumped operator and both residuals; (b)
the full matrix-free walk per recorded step and, scaled, to ``tau*``; the
lumped recurrence the same; (c) the limit by dense eigendecomposition of the
full discriminant (run only while ``N <= FULL_EIG_MAX``; otherwise ``not run``
with an ``O(N^3)`` estimate from the lumped timing) against the lumped ``K``;
(d) the classical stationary distribution, full sparse ARPACK against the
lumped dense solve; (e) the unfolding. Everything is reported as a time to
compute; nothing here is a claim about anything else.

``--smoke`` is a wiring check, never reportable: it runs the first thread
setting only, one step sample, and skips the dense eigendecomposition of the
full operator (about 80% of an untrimmed smoke run: 300-440 s per thread
setting and dataset, peaking near 22 GB). The scaling rows still time that
code path on small synthetic instances.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable

import polars as pl
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_CONFIG  # noqa: E402
from datasets import DATASET_NAMES, get_dataset  # noqa: E402
from evaluation.provenance import RunTimer, machine, write_manifest, write_result  # noqa: E402
from fingerprint import build_ranker, load_frozen_partitions, resolve_config  # noqa: E402
from paths import convergence_path, stage_dir  # noqa: E402
from quantum.inspired.mulp import QuantumWalkRanker  # noqa: E402
from quantum.inspired.walk_cache import time_steps  # noqa: E402
from quantum.lumped_operator import build_lumped_matrices, unfold_class_scores  # noqa: E402
from quantum.lumping_router import discriminant  # noqa: E402
from quantum.mechanism.classical import stationary  # noqa: E402
from quantum.mechanism.limit import limiting_distribution  # noqa: E402
from quantum.mechanism.running_average import step_marginals  # noqa: E402
from quantum.mechanism.variants import primary_variant, route_variant  # noqa: E402
from synthetic.generator import generate_attribute_graph  # noqa: E402

STAGE = "10_runtime"
FULL_EIG_MAX = 15_000
SCALING_N = (1_000, 3_000, 10_000, 30_000, 100_000)
SCALING_LEVELS = {"hmeq_like": (6, 2), "fmac_like": (24, 19)}
STEP_SAMPLES = 3

T5_COLUMNS = {
    "dataset": "dataset name", "network": "the dataset's primary network", "threads": "BLAS threads of the run",
    "tau_star": "convergence point of the network (tau_max when not reached), used to scale the per-step times",
    "operation": "what was timed", "N_full": "supra-nodes of the unlumped operator after unused slots are dropped",
    "N_ranker": "supra-nodes of the matrix-free full ranker (reserved slots included)",
    "R": "supra-nodes of the lumped operator", "seconds": "median wall seconds over the repeats (per step for the walk rows)",
    "peak_mb": "peak traced memory in MiB", "repeats": "repeats timed", "seconds_to_tau_star": "per-step seconds times tau_star",
    "route": "discriminant the router selected", "status": "'not run' when the full eigendecomposition was skipped",
    "estimate_seconds_O_N3": "O(N^3) estimate of the full eigendecomposition from the lumped timing",
}
SCALING_COLUMNS = {
    "config": "HMEQ-like or Freddie-Mac-like level counts", "n": "individuals", "N": "supra-nodes of the unlumped operator",
    "classes": "lumping classes", "R": "supra-nodes of the lumped operator",
    "t_lumped_build": "seconds to build the lumped operator", "mem_lumped_build_mb": "its peak traced memory in MiB",
    "t_lumped_limit": "seconds of the lumped spectral limit", "mem_lumped_limit_mb": "its peak traced memory in MiB",
    "t_residuals": "seconds to route the operator with both residuals (while N <= FULL_EIG_MAX)",
    "mem_residuals_mb": "its peak traced memory in MiB", "route": "discriminant the router selected",
    "t_full_limit": "seconds of the full spectral limit (while N <= FULL_EIG_MAX)", "mem_full_limit_mb": "its peak traced memory in MiB",
    "t_full_step": "seconds per step of the full coefficient recurrence", "t_lumped_step": "seconds per step of the lumped recurrence",
    "full_status": "'not run' when N exceeds FULL_EIG_MAX",
}



def timed(fn: Callable[[], Any], repeats: int, threads: int) -> dict[str, Any]:
    """Median wall seconds of ``repeats`` runs under ``threads`` BLAS threads, and the peak traced memory."""
    seconds, peaks, result = [], [], None
    with threadpool_limits(limits=int(threads)):
        for _ in range(int(repeats)):
            tracemalloc.start()
            began = time.perf_counter()
            result = fn()
            seconds.append(time.perf_counter() - began)
            peaks.append(tracemalloc.get_traced_memory()[1])
            tracemalloc.stop()
    return {"seconds": float(statistics.median(seconds)), "peak_mb": float(max(peaks)) / 2**20,
            "repeats": int(repeats), "threads": int(threads), "result": result}


def dataset_rows(name: str, config, threads: list[int], repeats: int, *, smoke: bool = False) -> list[dict[str, Any]]:
    step_samples = 1 if smoke else STEP_SAMPLES
    full_eig_max = 0 if smoke else FULL_EIG_MAX
    dataset = get_dataset(name)
    partitions = load_frozen_partitions(dataset)
    spec = dataset.network_specs(partitions.frame)[dataset.primary]
    conv = convergence_path(dataset.primary, stage=dataset.stage("quantum"))
    tau_star = None
    if conv.exists():
        record = json.loads(conv.read_text())
        tau_star = int(record["tau_star"]) if record.get("reached") else int(record["tau_max"])
    rows = []
    with tempfile.TemporaryDirectory() as td:
        masked = Path(td) / "masked.parquet"
        partitions.mask_labels(partitions.test_idx).write_parquet(masked)
        for t in threads:
            base = {"dataset": name, "network": dataset.primary, "threads": t, "tau_star": tau_star}

            def build_full():
                ranker = build_ranker(masked, spec, config)
                ranker.buildAdjMatrix()
                ranker.buildPersonalizationMatrix()
                ranker.buildGoogleOperator(config.resolved_alpha())
                return ranker

            r = timed(build_full, repeats, t)
            ranker = r.pop("result")

            def build_lumped():
                variant = primary_variant(partitions, spec, config, partitions.test_idx)
                return variant, route_variant(variant, config)

            r_lumped = timed(build_lumped, repeats, t)
            variant, routed = r_lumped.pop("result")
            R = int(routed.n)
            # N_full is the one definition every table shares (unused slots dropped);
            # the matrix-free ranker also carries the reserved slots.
            base = {**base, "N_full": int(routed.residuals["n_full"]), "N_ranker": int(ranker.N)}
            rows.append({**base, "operation": "full_operator_build", **r})
            rows.append({**base, "operation": "class_construction_lumped_operator_residuals", "R": R, **r_lumped})

            walker = QuantumWalkRanker(ranker=ranker, alpha=config.resolved_alpha())
            walker.buildAdjMatrix()
            walker.buildPersonalizationMatrix()
            walker.buildTransitionMatrix()
            walker.buildPersonalizationVector()
            walker.buildSzegedyWalk()
            r = timed(lambda: statistics.median(time_steps(walker.szegedy_walk, step_samples)), 1, t)
            per_step = float(r.pop("result"))
            rows.append({**base, "operation": "full_walk_per_step", **r, "seconds": per_step,
                         "seconds_to_tau_star": None if tau_star is None else per_step * tau_star})
            r = timed(lambda: step_marginals(routed.G, routed.v, routed.K, 10), 1, t)
            r.pop("result")
            rows.append({**base, "operation": "lumped_walk_per_step", "R": R, **r, "seconds": r["seconds"] / 10.0,
                         "seconds_to_tau_star": None if tau_star is None else r["seconds"] / 10.0 * tau_star})

            r = timed(lambda: limiting_distribution(routed.G, routed.v, K=routed.K, d_tol=config.d_tol,
                                                    phase_tol=config.phase_tol, mass_tol=config.mass_tol), repeats, t)
            r.pop("result")
            rows.append({**base, "operation": "limit_lumped_eigendecomposition", "R": R, **r})
            N = int(ranker.N)
            if N <= full_eig_max:
                full_routed = route_variant(variant, config, keep_full=True)

                def full_limit():
                    return limiting_distribution(full_routed.G_full, full_routed.v_full, K=discriminant(full_routed.G_full),
                                                 d_tol=config.d_tol, phase_tol=config.phase_tol, mass_tol=config.mass_tol)

                r = timed(full_limit, 1, t)
                r.pop("result")
                rows.append({**base, "operation": "limit_full_eigendecomposition", **r})
                del full_routed
            else:
                estimate = rows[-1]["seconds"] * (N / R) ** 3
                rows.append({**base, "operation": "limit_full_eigendecomposition", "seconds": None, "peak_mb": None,
                             "repeats": 0, "threads": t, "status": "not run (smoke)" if smoke else "not run",
                             "estimate_seconds_O_N3": estimate})
            r = timed(lambda: ranker.pageRank(config.resolved_alpha()), repeats, t)
            r.pop("result")
            rows.append({**base, "operation": "classical_stationary_full_sparse", **r})
            r = timed(lambda: stationary(routed.G), repeats, t)
            pi = r.pop("result")
            rows.append({**base, "operation": "classical_stationary_lumped_dense", "R": R, **r})
            r = timed(lambda: unfold_class_scores(variant.nodes, pi), repeats, t)
            r.pop("result")
            rows.append({**base, "operation": "unfolding", "R": R, **r})
    return rows


def scaling_rows(config, repeats: int, sizes=SCALING_N) -> list[dict[str, Any]]:
    """Synthetic two-layer populations: N, classes, R, time and memory, full against lumped."""
    rows = []
    for label, levels in SCALING_LEVELS.items():
        for n in sizes:
            instance = generate_attribute_graph(n, 2, levels, 0.5, 0.2, 0.03, 0.2, 0.05, 1.0, 0.0, 0.0, config.seed_synthetic("scaling", label, n))
            partitions = instance.partitions()
            variant = primary_variant(partitions, instance.spec, config, partitions.test_idx)
            N = int(len(variant.nodes_full))
            copies = variant.nodes[variant.nodes["node_type"].astype(str) == "borrower_copy"]
            n_classes = int((copies["layer_index"].astype(int) == 0).sum())
            row: dict[str, Any] = {"config": label, "n": n, "N": N, "classes": n_classes}
            r = timed(lambda: build_lumped_matrices(variant.nodes, variant.edges, alpha=config.resolved_alpha(),
                                                    influence_type=config.influence_type), repeats, 1)
            G, v = r.pop("result")
            row.update(R=int(G.shape[0]), t_lumped_build=r["seconds"], mem_lumped_build_mb=r["peak_mb"])
            r = timed(lambda: limiting_distribution(G, v, K=discriminant(G), d_tol=config.d_tol, phase_tol=config.phase_tol, mass_tol=config.mass_tol), 1, 1)
            r.pop("result")
            row.update(t_lumped_limit=r["seconds"], mem_lumped_limit_mb=r["peak_mb"])
            if N <= FULL_EIG_MAX:
                r = timed(lambda: route_variant(variant, config, keep_full=True), 1, 1)
                routed = r.pop("result")
                row.update(t_residuals=r["seconds"], mem_residuals_mb=r["peak_mb"], route=routed.route)
                r = timed(lambda: limiting_distribution(routed.G_full, routed.v_full, K=discriminant(routed.G_full), d_tol=config.d_tol,
                                                        phase_tol=config.phase_tol, mass_tol=config.mass_tol), 1, 1)
                r.pop("result")
                row.update(t_full_limit=r["seconds"], mem_full_limit_mb=r["peak_mb"])
                r = timed(lambda: step_marginals(routed.G_full, routed.v_full, discriminant(routed.G_full), 3), 1, 1)
                r.pop("result")
                row.update(t_full_step=r["seconds"] / 3.0)
            else:
                row.update(t_residuals=None, t_full_limit=None, t_full_step=None, route=None, full_status="not run (N > FULL_EIG_MAX)")
            r = timed(lambda: step_marginals(G, v, discriminant(G), 3), 1, 1)
            r.pop("result")
            row.update(t_lumped_step=r["seconds"] / 3.0)
            rows.append(row)
            print(json.dumps(row, default=str))
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=DATASET_NAMES, action="append", default=None)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--scaling", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    config, smoke = resolve_config(DEFAULT_CONFIG, smoke=args.smoke)
    threads = list(args.threads[:1]) if smoke else list(args.threads)
    timer = RunTimer()
    output = stage_dir(STAGE)
    outputs = []
    info = machine()
    (output / "machine.json").write_text(json.dumps(info, indent=2))
    for name in args.dataset or ():
        # One T5 file per dataset scope: two datasets never write to the same file.
        rows = dataset_rows(name, config, threads, 1 if smoke else args.repeats, smoke=smoke)
        outputs.append(write_result(name, "T5_runtime", pl.DataFrame(rows, strict=False, infer_schema_length=None), stage=STAGE,
                                    columns=T5_COLUMNS))
    if args.scaling:
        rows = scaling_rows(config, 1, sizes=(300, 1_000) if smoke else SCALING_N)
        outputs.append(write_result("runtime", "EC_scaling", pl.DataFrame(rows, strict=False, infer_schema_length=None), stage=STAGE,
                                    columns=SCALING_COLUMNS))
    write_manifest(output, arm_name=STAGE, config=config, inputs=[], outputs=outputs, seeds={}, timer=timer,
                   execution_mode="simulation", extra={"machine": info, "smoke": smoke, "threads": threads})
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
