# Quantum Personalized PageRank for credit-risk networks

This repository compares Quantum Personalized PageRank, or QPPR, with classical personalized PageRank on multilayer credit-risk networks. The experiment code supports HMEQ and the author-supplied Freddie Mac Florida extracts. The saved Python verification records cover HMEQ, not a completed Freddie Mac run.

## Cite

The associated manuscript is *Improving Credit-Risk Exposure Detection Using a Personalized Quantum PageRank*, dated June 2026. The supplied draft omits author names for review.

This checkout does not provide an assigned paper DOI or code-archive DOI. A final citation must identify both the paper and the archived code release. The authors must supply those identifiers and the author list before publication. No placeholder DOI is presented as a valid citation here.

The development repository is [Thomson-Lam/qmulp](https://github.com/Thomson-Lam/qmulp). This submission copy uses the directory layout described below.

## What the code computes

Both methods use the same column-stochastic operator, `G = alpha*T + (1-alpha)*P`. The default damping factor is `alpha = 0.85`, and the interlayer weight is `eta = 1`.

The combined MuLP restart sends a reference borrower's column to that borrower's own layer copies. Other columns use the shared reference distribution `v`. Classical PageRank computes the stationary distribution of this operator.

QPPR uses a Szegedy walk, starts from the same reference distribution, and measures the source register. Its reported score is the exact limit of the time-averaged distribution, not a score at a fixed finite horizon. Borrower scores sum their layer copies.

The reduction router uses the lumped discriminant when `max|DB - B D-bar| <= 1e-12`. Otherwise it uses the compressed discriminant `H = B^T D B`. The code reports convergence, rank agreement, default detection, mechanism controls, runtime, and circuit resources. Local circuit experiments use simulation. They do not demonstrate quantum speedup or execution on a physical QPU.

The [classical guide](docs/classical-pipelines.md) and [quantum guide](docs/quantum-pipelines.md) explain the operators, masking rules, stages, and numerical checks.

## Install

Run commands from the repository root on Linux. The local verification environment used Python 3.12.13. The upstream dependency file names Python 3.11, but the port checks did not test that version.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

[requirements.txt](requirements.txt) pins the dependencies. The main numerical packages are NumPy 2.4.2, SciPy 1.17.0, pandas 3.0.0, Polars 1.37.1, scikit-learn 1.8.0, and XGBoost 3.2.0. Circuit simulation uses Qiskit 2.5.0, Qiskit Aer 0.17.2, and Qiskit IBM Runtime 0.47.0. Notebook execution uses nbclient 0.10.4, nbformat 5.10.4, and ipykernel 7.2.0. Tests use pytest 8.4.1.

Use `bash scripts/run.sh` instead of a bare Python command. The wrapper adds `src/` and the repository root to `PYTHONPATH`, then uses `.venv/bin/python`. To select another interpreter, set `PYTHON` to its absolute path.

Local runs need no commercial solver or IBM credentials. Qiskit IBM Runtime supplies the FakeMarrakesh calibration snapshot used by the simulator. A separate physical-device run requires provider access and compliance with provider terms. The repository does not submit device jobs.

## Obtain the data

Raw datasets are not supplied as tracked submission inputs. A fresh checkout must obtain them before the corresponding pipeline can run. [data/README.md](data/README.md) records sources and access restrictions.

### HMEQ

Download the public HMEQ CSV and check its hash:

```bash
mkdir -p data
curl --fail --location https://www.creditriskanalytics.net/uploads/1/9/5/1/19511601/hmeq.csv --output data/hmeq.csv
sha256sum data/hmeq.csv
```

The expected SHA-256 is `dfdbc2b7cdf728a15b53e323cde6127995715dfa6b178bd3c1e3d9916d0367aa`. The raw file has 5,960 records and 13 columns. Stage 01 cleans the data and creates the processed Parquet file, frozen split, and bootstrap indices. Old processed training and test files are not required.

Public download access does not grant redistribution rights. Confirm the dataset terms before including a copy in a public archive.

### Freddie Mac

Obtain the exact author-supplied Florida extracts under their access terms. Put them in `data/` with names matching these patterns:

| Cohort | Filename pattern | SHA-256 |
|---|---|---|
| Development, March 2023 | `Quantum_florida_202303*.csv` | `f9740a85ac7bed545908bbe069298bb9e07b43bcc75aba354d3f75ea5fda7927` |
| Test, March 2024 | `Quantum_florida_202403*.csv` | `dddab754e9d6bc2c167896268151e0b4aa4709e8f92ed5399925e4a79fcb5a86` |

The pipeline uses the supplied `target` without reconstructing it. These extracts are absent from this checkout. The full two-dataset experiment cannot run without them.

## Run the experiments

### Check for previous outputs

Before an independent run, inspect the output directory:

```bash
bash scripts/run.sh -m scripts.preflight --inventory
bash scripts/run.sh -m scripts.preflight --fresh
```

`--fresh` fails if `results/outputs/` contains files outside `logs/`. It does not delete files. Archive previous outputs before a new run, especially when changing from smoke to full settings. Keep outputs between stages of the same run because later stages need them.

Generated pipeline outputs under `results/outputs/` are excluded from Git. They are not restored by a fresh checkout. Stage 01 also rewrites processed data in `data/`. `scripts/run_all.py` does not perform the clean-start check automatically.

### Check HMEQ locally

After setup, data acquisition, and a clean-start check, run:

```bash
bash scripts/reproduce.sh classical
bash scripts/reproduce.sh hardware
```

The first command executes HMEQ stages 01 through 03. The second executes stage 06 after the classical command has prepared its inputs. These commands use smoke settings and skip manuscript assembly. They do not execute the quantum score and evaluation stages 04, 05, and 05a.

Smoke mode uses two cross-fitting folds, 25 bootstrap replicates, 256 circuit shots, and recorded hardware steps 1 and 2. These results check execution, not paper estimates. Hardware here means local Aer simulation. Resource-limited cells receive explicit skipped statuses in `EC_noise_grid.csv`.

To include the quantum score, evaluation, mechanism, and hardware stages in the same smoke run:

```bash
bash scripts/run.sh -m scripts.run_all --dataset hmeq --smoke --from-stage 04 --to-stage 06 --skip-figures
```

Stage 05a needs the placebo results from stage 05. Do not skip that dependency.

### Reproduce the full experiment

Obtain both datasets and start with clean outputs, then run:

```bash
bash scripts/reproduce.sh paper
```

This command runs preflight, stages 01 through 10 for both datasets, and figure and table assembly. The local verification record does not establish a completed full-paper run. Runtime and memory needs depend on the dataset and circuit size. The quantum guide describes simulation limits and skipped-cell reporting.

To run only HMEQ with the full settings, use a separate clean output tree:

```bash
bash scripts/run.sh -m scripts.run_all --dataset hmeq --from-stage 01 --to-stage 06 --skip-figures
```

This omits the counterexample, synthetic, variability, runtime, and manuscript-assembly stages. For a partial run, keep `--skip-figures` because the assemblers need additional result CSVs. `--require-clean` makes `run_all` reject uncommitted changes. Without that flag, it records the dirty state rather than blocking the run.

### Run experiments without credit data

```bash
bash scripts/run.sh -m scripts.ec_counterexample
bash scripts/run.sh -m scripts.run_synthetic --smoke
```

The counterexample uses eight synthetic individuals and has no smoke flag. Remove `--smoke` from the synthetic command for its full configuration.

### Export a circuit for a separate device run

After stage 06 has selected an instance:

```bash
bash scripts/run.sh -m scripts.export_hardware_instance --dataset hmeq
```

The exporter writes circuits, an exact comparison vector, an index map, and submission instructions. `--k` selects another retained class count at the selected walk length. Export does not submit a device job.

## Pipeline and outputs

`scripts/run_all.py` executes the stage notebooks with nbclient. It sets the dataset in memory and saves executed copies under `results/outputs/executed_notebooks/`. Source notebooks remain unchanged.

| Stage | Driver | Main outputs |
|---|---|---|
| 01 data | `src/stages/data.py` | Processed data, frozen split, bootstrap indices, observation flags, and `T1_data` |
| 02 network | `src/stages/network.py` | Network facts, lumped classes, and lumpability checks |
| 03 classical | `src/stages/classical.py` | Classical test and cross-fitted scores, masked frames, and operator digests |
| 04 quantum | `src/stages/quantum.py` | Reduction checks, exact limits, convergence results, and quantum scores |
| 05 evaluation | `src/stages/evaluation.py` | Agreement, decision metrics, placebo tests, and downstream checks |
| 05a mechanism | `src/stages/mechanism.py` | Coupling, restart variants, partial-observation controls, and single-layer checks |
| 06 hardware | `src/stages/hardware.py` | Circuit resources, statevector checks, noise grids, and selected-instance results |
| 07 counterexample | `scripts/ec_counterexample.py` | `EC_counterexample` |
| 08 synthetic | `scripts/run_synthetic.py` | Synthetic experiments S1, S2, S3, and S3b |
| 09 variability | `scripts/run_multisplit.py` | Split and reference-resampling comparisons |
| 10 runtime | `scripts/benchmark_runtime.py` | Runtime and scaling results |

Result CSVs live under `results/outputs/results/<scope>/`, where the scope is `hmeq`, `fmac`, `synthetic`, or `runtime`. Stage intermediates and manifests live under `results/outputs/notebook_pipeline/`. Invocation timings live under `results/outputs/timings/`.

Manifests record configuration, seeds, dependency versions, input and output digests, elapsed time, Git state, and the source digest. Run outputs can differ when resource budgets change which simulation cells finish.

### Assemble the manuscript files

After the required experiments finish:

```bash
bash scripts/run.sh -m scripts.make_figures
bash scripts/run.sh -m scripts.make_manuscript_tables
```

Figures go to `results/outputs/manuscript/figures/`. Table fragments go to `results/outputs/manuscript/tables/`. The assembler also writes `table_columns.csv`, `number_map.csv`, and `numbers.tex` under `results/outputs/manuscript/`. Missing result sources leave manuscript numbers unresolved.

The following maps match `src/evaluation/table_specs.py` and `scripts/make_figures.py`. Source entries are relative to `results/outputs/results/` and omit the `.csv` suffix. `<dataset>` means both `hmeq` and `fmac`.

| Table file | In the manuscript | Source CSVs |
|---|---|---|
| `T1_data_reduction.tex` | Table 1 | `<dataset>/T1_data`, `<dataset>/T_convergence` |
| `T2_agreement.tex` | Table 2 | `<dataset>/T2_agreement`, `<dataset>/T2_decision`, `<dataset>/T2_variability` |
| `T3_variants.tex` | Table 3 | `<dataset>/T3_variants` |
| `T4_partial.tex` | Table 4 | `<dataset>/T4_partial` |
| `T5_runtime.tex` | Table 5 | `<dataset>/T5_runtime` |
| `T6_hardware.tex` | Table 6 | `<dataset>/T6_hardware`, `<dataset>/T6_hardware_header` |
| `EC_counterexample.tex` | Table EC.1 | `synthetic/EC_counterexample` |
| `EC_reduction_residuals.tex` | e-companion, reduction | `<dataset>/EC_reduction_residuals` |
| `EC_convergence.tex` | e-companion, convergence | `<dataset>/T_convergence`, `<dataset>/EC_convergence_sensitivity` |
| `EC_registers.tex` | e-companion, registers | `<dataset>/EC_registers` |
| `EC_classical_horizon.tex` | e-companion, classical horizon | `<dataset>/EC_classical_horizon` |
| `EC_single_layer_share.tex` | e-companion, single layer | `<dataset>/EC_single_layer_share` |
| `EC_multisplit.tex` | e-companion, variability | `<dataset>/T2_variability` |
| `EC_decision_full.tex` | e-companion, decision | `<dataset>/EC_decision_full` |
| `EC_placebo.tex` | e-companion, placebo | `<dataset>/EC_placebo` |
| `EC_strata.tex` | e-companion, strata | `<dataset>/EC_strata` |
| `EC_downstream.tex` | e-companion, downstream | `<dataset>/EC_downstream` |
| `EC_feature_importance.tex` | e-companion, feature importance | `<dataset>/EC_feature_importance` |
| `EC_synthetic.tex` | e-companion, synthetic | `synthetic/EC_synthetic_S1_summary`, `synthetic/EC_synthetic_S2_summary`, `synthetic/EC_synthetic_S3_summary`, `synthetic/EC_synthetic_S3b_summary` |
| `EC_scaling.tex` | e-companion, scaling | `runtime/EC_scaling` |
| `EC_circuit_resources.tex` | e-companion, circuit resources | `<dataset>/EC_circuit_resources` |
| `EC_statevector.tex` | e-companion, statevector rung | `<dataset>/EC_statevector` |
| `EC_noise_grid.tex` | e-companion, noise grid | `<dataset>/EC_noise_grid` |
| `EC_hardware_scores.tex` | e-companion, hardware scores | `<dataset>/EC_hardware_scores` |

| Figure file | In the manuscript | Source CSVs |
|---|---|---|
| `fig1_coupling.pdf` | Figure 1 | `<dataset>/T_eta` |
| `fig2_convergence.pdf` | Figure 2 | `<dataset>/EC_convergence_curves`, `<dataset>/T_convergence` |
| `fig3_synthetic.pdf` | Figure 3 | `synthetic/EC_synthetic_S1_summary`, `synthetic/EC_synthetic_S2_summary`, `synthetic/EC_synthetic_S3_summary` |
| `fig4_noise.pdf` | Figure 4 | `<dataset>/EC_noise_grid`, `<dataset>/T6_hardware_header` |
| `ECfig_scaling.pdf` | e-companion, scaling | `runtime/EC_scaling` |
| `ECfig_noise_spearman.pdf` | e-companion, noise | `<dataset>/EC_noise_grid` |
| `ECfig_noise_tv.pdf` | e-companion, noise | `<dataset>/EC_noise_grid` |

## Test and inspect verification

For the full test suite:

```bash
bash scripts/reproduce.sh tests
```

For tests without the slow, Qiskit, and IBM Runtime markers:

```bash
bash scripts/run.sh -m pytest tests/ -m "not slow and not qiskit and not ibm_runtime" -q
```

Do not run the suite while another process writes pipeline outputs. Some tests inspect those files, so the pass and skip counts depend on the artifacts present.

[results/verification/README.md](results/verification/README.md) records the earlier classical and local hardware checks, numerical comparisons, and known limits. [environment.txt](results/verification/environment.txt) records the installed package versions for those checks. These records are smoke-run evidence, not complete paper results.

The old port-integrity script, `scripts/verify-port.sh`, needs checksum manifests that are no longer in this checkout. It is not a prerequisite for running experiments or tests.

## Lean proof verification

The [Lean proof project](src/lean/README.md) formalizes manuscript Propositions 1, 2, and 3 and supporting electronic-companion results.
Its README links to the theorem declarations and explains their hypotheses.
The proof is separate from the numerical experiments and needs no credit data. It does not verify the Python implementation.

With Git and the Lean/Lake toolchain manager [elan](https://github.com/leanprover/elan)
installed, run from the repository root:

```bash
bash scripts/verify-lean.sh
```

The project pins Lean 4.31.0 and Mathlib dependencies. The wrapper builds the
library and runs the 19-theorem axiom audit, saving fresh logs in `results/lean/`
without overwriting the supplied records in `src/lean/`.
The saved submission logs report a successful build with 2,967 jobs and a passing 19-theorem audit.
First use requires network access to fetch the toolchain and dependencies.

## Repository layout

| Directory | Contents |
|---|---|
| `src/` | Classical and quantum methods, data preparation, evaluation, configuration, and stage drivers |
| `src/lean/` | Lean proofs, pinned dependency configuration, project README, audit scripts, and supplied verification records |
| `scripts/` | Run wrappers, Lean verification, experiment drivers, preflight checks, and figure and table assemblers |
| `results/lean/` | Saved submission build and axiom-audit logs, overwritten by `scripts/verify-lean.sh` |
| `notebooks/` | Dataset preparation and stage notebooks |
| `tests/` | Operator, circuit, evaluation, pipeline, and output-contract tests |
| `data/` | Dataset instructions and locally obtained inputs |
| `results/verification/` | Selected local verification evidence |
| `results/outputs/` | Generated run outputs, excluded from Git |
| `docs/` | Classical and quantum pipeline guides |

Run parameters live in `src/config.py`. Dataset definitions live in `src/hmeq/config.py` and `src/fmac/config.py`. `src/paths.py` anchors data and output paths to this repository.

## License and submission status

[LICENSE](LICENSE) is currently an incomplete license notice, not a license grant. [AUTHORS](AUTHORS) does not yet contain the author list. The authors must complete both files and the final citation before public deposit. Dataset permissions require separate review.
