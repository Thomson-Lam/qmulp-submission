# Classical pipelines

This guide covers data preparation, network construction, and classical scores in stages 01, 02, and 03. Paths are relative to the `sub-qmulp/` root. The [quantum guide](quantum-pipelines.md) covers the later stages and manuscript outputs.

## Reproduce the classical run

### Set up the environment

Run commands from the `sub-qmulp/` root. Create a Python 3.12 virtual environment, which keeps this project's installed packages separate from other Python installations. The recorded port checks used Python 3.12.13. The upstream requirements specify Python 3.11, which those checks did not test.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Use `bash scripts/run.sh` for Python commands. It adds `src/` and the repository root to `PYTHONPATH`, Python's module search path, and uses `.venv/bin/python`. Set `PYTHON` to an interpreter path if you use another environment.

### Obtain the input data

The Home Equity dataset, or HMEQ, contains loan records and repayment outcomes. Its data file is not included in Git. Download the raw input to `data/hmeq.csv` from the source in [data/README.md](../data/README.md). For example:

```bash
mkdir -p data
curl --fail --location https://www.creditriskanalytics.net/uploads/1/9/5/1/19511601/hmeq.csv --output data/hmeq.csv
sha256sum data/hmeq.csv
```

Secure Hash Algorithm 256-bit, or SHA-256, produces a checksum used to detect changes in a file. The expected checksum is `dfdbc2b7cdf728a15b53e323cde6127995715dfa6b178bd3c1e3d9916d0367aa`. Check the hash before a run. Public access does not establish redistribution rights.

For Freddie Mac, obtain the author-supplied March 2023 and March 2024 Florida extracts under their access terms. Put the files matching `Quantum_florida_202303*.csv` and `Quantum_florida_202403*.csv` in `data/`. The data README gives their hashes. The supplied `target` is the outcome label used for evaluation. Do not replace it with a label reconstructed from other fields.

Stage 01 generates processed Parquet files. Parquet stores tabular data in columns, retaining field types. The old training and test Parquet copies are not substitutes for the raw inputs. Ignored data and output files must be obtained or regenerated after a fresh checkout.

### Start with a separate output directory

Before an independent run, inspect existing outputs:

```bash
bash scripts/run.sh -m scripts.preflight --inventory
bash scripts/run.sh -m scripts.preflight --fresh
```

`--fresh` fails if `results/outputs/` contains files outside `logs/`. It does not remove anything. Move previous outputs to a separate archive before a new run. Smoke mode uses reduced experiment settings to check execution. Do not mix its outputs with full-run outputs. Stage 01 rewrites processed data in `data/`.

Do not clear outputs between stages of the same run. Later stages need the files produced by earlier stages. `run_all` does not run preflight, the output-directory check, automatically.

### Run the stages

For a short HMEQ check:

```bash
bash scripts/reproduce.sh classical
```

This runs stages 01 through 03 with `--smoke` and skips figure and table assembly. Smoke mode uses two cross-fitting folds and 25 bootstrap replicates.

A fold is a subset of development borrowers. Cross-fitting scores each fold with its outcome labels hidden from the operator. The operator determines how probability moves between network nodes.

A bootstrap replicate samples test borrowers with replacement to estimate uncertainty. Repeated samples show how evaluation results vary with the borrowers selected. These outputs are execution checks, not paper estimates.

For the full classical configuration on HMEQ, start from clean outputs and run:

```bash
bash scripts/run.sh -m scripts.run_all --dataset hmeq --from-stage 01 --to-stage 03 --skip-figures
```

Replace `hmeq` with `fmac` only after you obtain its inputs. Omit `--dataset` to run both datasets. Keep `--skip-figures` for a partial run because manuscript assembly needs later-stage results.

A stage manifest records inputs, settings, software versions, and output details. These records appear under `results/outputs/notebook_pipeline/`. HMEQ classical scores appear in `03_classical/classical_scores_<network>.parquet` below that directory. `NET_JOB` links borrowers through employment categories. `NET_REASON` uses loan purposes. `NET_2L` combines both relationship layers.

Result files use comma-separated values, or CSV. These files go to `results/outputs/results/hmeq/`. Executed notebooks and invocation timings go to `results/outputs/executed_notebooks/` and `results/outputs/timings/`.

Run the tests with:

```bash
bash scripts/reproduce.sh tests
```

Some tests inspect generated output files. Test results therefore depend on which pipeline outputs exist. The [recorded verification](../results/verification/README.md) describes the earlier local checks and their limits.

## Operator definitions

Run parameters live in `src/config.py`. `src/paths.py` and each dataset's `DatasetConfig` define file locations. `DatasetConfig` holds the dataset's filenames, label definition, and network definitions.

### Graph and missing categories

A graph represents entities as nodes and their connections as edges. An edge weight controls the strength of a connection. The adjacency matrix `A` stores these weights.

`A[i, j]` is the edge weight from source node `i` to destination node `j`. A layer represents one relationship type, such as employment or loan purpose. Each borrower has a separate node, called a layer copy, in each layer.

Each layer block contains these borrower copies and reserved positions for category nodes. `src/classical/mulp/indexing.py` assigns numeric node identifiers. Unused category positions have no edges or probability mass. The two-layer HMEQ layout reserves 16 category slots and uses 8. The expected Freddie Mac layout reserves 86 and uses 43.

An individual copy connects to its category node in both directions with weight 1. This connection is a membership edge. No membership edge exists when that layer has no recorded category. The individual's two layer copies connect in both directions with weight `eta`, called `stickiness` in the configuration. Its default is 1.

One-hot encoding creates a binary column for each category. `src/preprocessing/missing_levels.py` treats the `Missing` column as an absence marker, instead of a category node.

### Transition, restart, and stationary score

The transition builder divides each nonempty adjacency row by its sum, then exchanges rows and columns. The resulting matrix `T` stores destination probabilities in each source-node column.

A dangling node has no outgoing edges. The builder replaces its column with the personalization vector `v`, a list of restart probabilities. The code must define `v` before it builds `T`.

The development set supplies reference borrowers whose defaults determine restart probabilities. The test set supplies outcomes for evaluation, not references.

`L` counts layers, `D` is the development reference set, and `|D|` counts its borrowers. `v` assigns probability `1 / (L * |D|)` to each reference's layer copies. Other nodes receive zero. A reference column of the restart matrix `P` distributes mass over that reference's own copies, with `1/L` per copy. Other columns equal `v`.

`FactoredRestart` computes the restart matrix's action without storing every matrix entry.

The operator is `G = alpha*T + (1-alpha)*P`, with default `alpha = 0.85`. Each column sums to one, so the operator preserves total probability. `alpha` weights movement along edges against restart.

`src/classical/mulp/mulp.py` uses the Arnoldi Package, or ARPACK, to compute the stationary distribution. This distribution remains unchanged when multiplied by `G`. Matrix-free computation applies `G` without storing all its entries.

The code checks that the result has nonnegative probabilities, sums to one, and satisfies the stationary equation within tolerance. A borrower's score sums probabilities across that borrower's layer copies.

## Stage behavior

### Stage 01 prepares data

`notebooks/01_fmac_cleaning.ipynb` and `src/fmac/cleaning.py` prepare Freddie Mac data. The category vocabulary is the set of values observed in the March 2023 development cohort. A cohort groups loans from the same first-payment period.

The code combines both cohorts while retaining a temporal split: March 2023 supplies development data, and March 2024 supplies test data. It uses the supplied `target` outcomes.

For each dataset, the data stage writes a bootstrap index matrix. Each row identifies sampled test borrowers, with default and non-default counts preserved. This sampling rule is outcome stratification.

The stage also writes `observation_flags.csv`, which records missing-category status, and `T1_data.csv`, which summarizes the data. Its manifest records the run details. It compares observed counts with expected counts and reports differences.

### Stage 02 builds networks

`src/stages/network.py` and `src/classical/network.py` compute node counts, edge counts, degree distributions, and category-combination limits. Here, a node's degree sums its outgoing edge weights. A degree distribution shows how those sums vary across nodes.

The category-combination limit bounds how many distinct patterns of layer categories the network can contain. Lumping groups borrowers with matching layer categories and visible outcome labels. Test-masked labels hide test outcomes from this grouping.

The stage records the number of quantum bits needed to encode the reduced network's nodes. This number is the register width.

The primary network combines both layers: `NET_2L` for HMEQ and `NET_FMAC_2L` for Freddie Mac. The latter uses postal-code and seller categories.

The stage distributes each grouped probability back across its member nodes, called unfolding. It checks that the unfolded stationary distribution matches the full-network distribution within tolerance.

### Stage 03 computes classical scores

`src/stages/classical.py` and `src/classical/scores.py` build operators from data tables with evaluation outcomes hidden. These are label-masked frames. The code saves them under `results/outputs/notebook_pipeline/masked/<network>/`.

Operator digests are hashes used to detect differences in operator inputs. Stage 04 uses the saved tables and checks that it builds the same operator.

`omega_C_test` contains classical scores from the operator whose test labels are hidden. `omega_C_crossfit` contains development scores computed with each borrower's fold hidden. Its stratified folds preserve outcome proportions, and the default has five folds.

The code checks the cross-fitted scores' receiver operating characteristic area under the curve, or ROC-AUC. This statistic measures whether defaults tend to rank above non-defaults. The default check rejects ROC-AUC values of 0.95 or greater as a warning of possible label leakage.

Label leakage means evaluation outcomes influenced the operator that produced the scores. Later stages compare lumped stationary and fold scores with these full-operator results.

## Shared evaluation code

The evaluation modules use these terms:

- A partition is a development or test subset. A three-level indicator distinguishes visible non-defaults, visible defaults, and hidden labels.
- A tie occurs when rounded scores are equal. Top-k metrics select the `k` highest-scoring borrowers and account for ties at the selection boundary.
- Spearman correlation and Kendall tau-b measure rank agreement. Kendall tau-b accounts for tied rankings.
- Expected overlap measures shared membership between top-k selections under independent tie-breaking. Jaccard divides their expected intersection size by their corresponding union size.
- Precision at k measures the default fraction among selected borrowers. Lift divides that fraction by the test-set default rate.
- Recall measures the fraction of all test defaults included in a selection. Precision-recall area under the curve, or PR-AUC, summarizes precision across recall levels. This code computes it with scikit-learn's average precision function.
- A comparator supplies another score for the same borrowers. The reference-permutation placebo samples development reference sets independently of outcomes while preserving the reference-set size.
- A placebo p-value measures how often those random-reference results meet or exceed the observed statistic, with a plus-one correction.

| File | Responsibility |
|---|---|
| `src/evaluation/masked_eval.py` | Partitions, label masking, three-level indicators, and influence checks |
| `src/evaluation/ties.py` | Score rounding and rank-k boundaries |
| `src/evaluation/reporting.py` | Rank agreement, overlap, Jaccard, precision, lift, PR-AUC, ROC-AUC, distribution distances, bootstrap comparisons, and placebo p-values |
| `src/evaluation/controls.py` | Development-label comparators and reference-permutation placebo |
| `src/evaluation/crossfit.py` | Fold assignment and cross-fitting |
| `src/evaluation/provenance.py` | Stage manifests and result files |

| Output field | Meaning |
|---|---|
| `l1_sum` | Sum of absolute differences between the two vectors |
| `mean_abs_diff` | Average absolute difference between corresponding entries |
| `total_variation` | Half the sum of absolute differences between probability vectors |
| `p_delta_pos` | Fraction of finite bootstrap differences above zero |
| `p_delta_zero` | Fraction of finite bootstrap differences equal to zero |
| `p_delta_neg` | Fraction of finite bootstrap differences below zero |

Paired comparisons evaluate both methods on the same bootstrap samples. Tie handling rounds scores to the configured number of decimal places before metrics.
