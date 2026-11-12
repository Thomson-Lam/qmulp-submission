"""Stage 03: the classical stationary score on the full operator, per network, and the cross-fitted folds.

Every operator is built on a label-masked frame written once to
``outputs/notebook_pipeline/masked/<network>/`` with its digests recorded, so
the quantum stage can prove it walks on the same operator. The scores are
the independent full-operator check of the lumped stationary score.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from classical import scores as classical
from config import ExperimentConfig
from datasets import get_dataset
from evaluation.provenance import RunTimer, write_stage_manifest
from fingerprint import load_frozen_partitions, resolve_config

STAGE_KEY = "classical"


def run(dataset_name: str, config: ExperimentConfig | None = None) -> dict[str, Any]:
    """Run the classical stage of one dataset."""
    from config import DEFAULT_CONFIG

    config, smoke = resolve_config(config or DEFAULT_CONFIG)
    dataset = get_dataset(dataset_name)
    timer = RunTimer()
    output = dataset.stage_dir(STAGE_KEY)
    partitions = load_frozen_partitions(dataset)
    specs = dataset.network_specs(partitions.frame)
    digest_log = classical.load_digests(output)
    holders: dict[str, dict[str, Any]] = {}
    outputs: list[Path] = []
    diagnostics: dict[str, Any] = {}
    ranking: list[dict[str, Any]] = []
    crossfit_record: dict[str, Any] = {}
    for network in dataset.ladder:
        spec = specs[network]
        holder: dict[str, Any] = {}
        scorer = classical.make_scorer(spec, config, digest_log, holder)
        omega_test = classical.score_test_operator(partitions, spec, config, scorer)
        bundle = holder[classical.TEST_OPERATOR_TAG]
        diagnostics[network] = classical.operator_diagnostics(bundle)
        classical.assert_operator_diagnostics(diagnostics[network])
        ranking.append(classical.holdout_ranking_row(omega_test, partitions, config, f"omega_C:{network}"))
        crossfit = None
        if network == dataset.primary:
            crossfit, validity = classical.crossfit_classical(partitions, spec, config, scorer, output)
            crossfit_record = {"validity": validity, "folds": crossfit.manifest_entry()}
        outputs.append(classical.write_classical_scores(output, network, partitions, omega_test, crossfit))
        holders[network] = holder
    outputs.append(classical.write_digests(output, digest_log))
    classical.assert_digests_complete(digest_log, config, dataset)
    manifest = write_stage_manifest(
        dataset, "classical", config, outputs=outputs, timer=timer,
        seeds={"crossfit_seed": config.seed_folds(),
               "placebo_seeds": [config.seed_placebo(p) for p in range(int(config.n_placebo_permutations))]},
        extra={"solver": "sparse ARPACK on the matrix-free operator (restart never materialised)", "alpha": config.resolved_alpha(),
               "diagnostics": diagnostics, "ranking": ranking, "crossfit": crossfit_record, "smoke": smoke})
    return {"diagnostics": diagnostics, "ranking": ranking, "crossfit": crossfit_record, "artifacts": [*outputs, manifest]}
