# Quantum pipelines

This guide covers the quantum score, evaluation, mechanism studies, local circuit simulation, and manuscript outputs. Paths are relative to `sub-qmulp/`. The experiments compare scores from the same operator, a matrix that determines how probability moves between network nodes. Reference borrowers are development-set borrowers whose defaults supply restart probability. Evaluation borrowers' outcomes are hidden when the operator is built. They do not establish quantum speedup or improved prediction.

## Reproduce the quantum stages

Complete the environment and data steps in the [classical guide](classical-pipelines.md). Use `bash scripts/run.sh` so that Python can find modules under `src/`.

### Run a local hardware smoke check

A smoke check uses reduced experiment settings to test execution, not to reproduce paper estimates. HMEQ is the Home Equity loan dataset.

Start from clean outputs as described in the classical guide. Run these commands in order:

```bash
bash scripts/reproduce.sh classical
bash scripts/reproduce.sh hardware
```

The first command prepares HMEQ data and classical operators. The second runs the stage 06 noise grid and hardware notebook with smoke settings. The noise grid tests combinations of network size, walk length, and simulated errors. It does not run stages 04, 05, or 05a. Qiskit Aer simulates circuits on your computer. FakeMarrakesh supplies recorded device characteristics for the noise model. Neither submits work to a quantum processing unit, or QPU.

The command runs a reduced experiment to check that the code executes:

- A shot is one circuit execution followed by measurement. Smoke mode uses 256 shots per circuit.
- The recorded length `tau` counts the states included in the average. At `tau = 1`, the simulator measures only the initial prepared state.
- At `tau = 2`, it averages measurements of the initial state and the state after one quantum-walk step.
- A borrower class groups borrowers with the same category in each network layer and the same label after evaluation labels are hidden. Missing categories also affect grouping.
- The simulator tests smaller networks that retain six or four borrower classes. It also includes a candidate containing every class in the grouped network.

Each simulation case combines a candidate network, a recorded length, and a noise setting. The code skips cases that exceed its configured size, memory, or estimated runtime limits. 

A row marked `not simulated: resource budget` records a skipped case, not a completed simulation. Inspect the `status` column in `results/outputs/results/hmeq/EC_noise_grid.csv` for details after running the commands.

A statevector lists the quantum amplitudes whose squared magnitudes give measurement probabilities. The statevector check compares circuit simulation with the mathematical walk. It has a 60-second estimated budget in smoke mode, not a hard timeout. The earlier [verification record](../results/verification/README.md) reports completed local checks, skipped cells, and timing limits.

### Run the quantum score and evaluation

After stages 01 through 03 of a full HMEQ run, continue without clearing outputs. See the [classical run instructions](classical-pipelines.md) for those stages.

```bash
bash scripts/run.sh -m scripts.run_all --dataset hmeq --from-stage 04 --to-stage 06 --skip-figures
```

The stage order is 04, 05, 05a, then 06. Stage 05a needs stage 05's `EC_placebo.csv`. It raises `FileNotFoundError` if that table is absent. The pipeline runner computes grid shards before it assembles the hardware results. A shard is a saved group of simulation results for one candidate network.

For a smoke run through these stages, add `--smoke` and use classical outputs from the same smoke configuration. Smoke mode checks time averages through 2,000 recorded states. It uses two placebo draws, each selecting a reference set from development borrowers without using their outcomes. Each draw preserves the original reference-set size. It compares the full and grouped networks at four recorded states. Do not report those results as full-run estimates.

### Run the complete submission pipeline

Obtain both datasets first. Archive previous outputs, then run:

```bash
bash scripts/reproduce.sh paper
```

This checks for clean outputs, runs both datasets through stages 01 through 10, and assembles figures and tables. The full Freddie Mac run requires the author-supplied extracts. The recorded port checks do not establish full-paper reproduction.

Stages 07 through 10 call `scripts/ec_counterexample.py`, `scripts/run_synthetic.py`, `scripts/run_multisplit.py`, and `scripts/benchmark_runtime.py`. The counterexample and synthetic studies can also run without credit datasets:

```bash
bash scripts/run.sh -m scripts.ec_counterexample
bash scripts/run.sh -m scripts.run_synthetic --smoke
```

### Assemble manuscript outputs

Result files use comma-separated values, or CSV. After all required result CSVs exist, run:

```bash
bash scripts/run.sh -m scripts.make_figures
bash scripts/run.sh -m scripts.make_manuscript_tables
```

The assemblers read `results/outputs/results/<scope>/`. Replace `<scope>` with `hmeq`, `fmac`, `synthetic`, or `runtime`, according to the experiment. Figures go to `results/outputs/manuscript/figures/`. Tables go to `results/outputs/manuscript/tables/`. `src/evaluation/table_specs.py` defines table inputs and columns.

The table assembler also writes `table_columns.csv`, `number_map.csv`, and `numbers.tex` under `results/outputs/manuscript/`. `src/evaluation/number_map.py` lists the manuscript numbers and the result fields that supply them. `number_map.csv` records these sources. `numbers.tex` contains commands that insert the numbers into the manuscript.

### Export circuits for a separate device run

After a full stage 06 run, export its selected instance:

```bash
bash scripts/run.sh -m scripts.export_hardware_instance --dataset hmeq
```

`--k` selects another retained borrower-class count. The exporter writes circuits before device mapping, expected probabilities for comparison, a map from circuit codes to network nodes, and submission instructions. It does not submit a device job. A physical run requires separate provider access and compliance with the provider's terms. No zero-noise extrapolation is implemented. That method estimates an error-free result from runs at several noise levels.

## Quantum score definitions

### Walk and measured register

The quantum and classical stages use `G = alpha*T + (1-alpha)*P`. `T` describes movement along network edges, `P` describes restarts, and `alpha` weights movement against restart.

A layer copy represents one borrower in one relationship layer, such as employment or loan purpose. `v` is the starting probability distribution over reference borrowers' layer copies. A dangling node has no outgoing edges. Its column in `T` is replaced by `v`.

Quantum Personalized PageRank, or QPPR, is the quantum score computed here. A quantum register is a group of quantum bits, called qubits. The two registers encode source and destination nodes.

The Szegedy state is `|psi_j> = |j> tensor sum_k sqrt(G_kj)|k>`. Here, `j` identifies the source node, `k` identifies a destination, and `G_kj` gives the transition probability. The notation `|j>` denotes the register state encoding node `j`. `tensor` combines the two registers. Let `A` map basis states to these states and let `S` swap the registers. Then `Pi = A A^dagger`, `W = S(2 Pi - I)`, and one recorded walk step is `U = W^2`. The initial state is `Psi_0 = A sqrt(v)`. `A^dagger` means the conjugate transpose of `A`, and `I` is the identity matrix. `Pi` projects onto the states prepared by `A`.

The reported score measures the source register and sums each individual's layer copies. Per-step distributions can oscillate. Their time average has the limit used for QPPR.

### Discriminant and exact limit

`src/quantum/mechanism/limit.py` computes the discriminant `D = A^dagger S A`, where `D_ij = sqrt(G_ij G_ji)`. This matrix combines transitions in both directions and determines the walk's oscillation frequencies. An eigenvalue is the factor multiplying an associated direction when a matrix acts on it. A discriminant eigenvalue `d = cos(theta)` gives walk eigenvalues `exp(+-2 i theta)`. Here, `theta` is an angle, and `i` is the imaginary unit. At `|d| = 1`, the corresponding walk component has eigenvalue one. A phase is the angle of a complex walk eigenvalue. The implementation combines components with matching phases before it squares their amplitudes.

The limiting score is `omega(j) = sum_lambda sum_k |<j,k|P_lambda|Psi_0>|^2`. Here, `lambda` identifies a walk eigenvalue, and `P_lambda` projects the starting state onto its associated components. Squared amplitude magnitudes give probabilities. Summing over destination nodes `k` gives the probability for source node `j`.

It is not a score at a fixed finite horizon, meaning a specified number of recorded states.

`src/quantum/mechanism/running_average.py` represents the state as `A y + S A z`, using coefficient vectors `y` and `z`. `K` is the matrix selected by the reduction router, defined below. One W step maps `(y, z)` to `(-z, y + 2 K z)`. Two such steps make U. The source marginal is `|y|^2 + G|z|^2 + 2 Re(conj(y) * (K z))`. A marginal sums probabilities over the other register. `Re` takes the real part, `conj` takes the complex conjugate, and `*` multiplies entries element by element.

The limit calculation records `c` for `||omega-bar_tau - omega||_1 <= c/tau`. It also records the bound `c <= 2(r-1)/delta_chord` over active phase groups. `omega-bar_tau` is the average over `tau` recorded states. The L1 norm, written `||...||_1`, sums absolute differences between probabilities.

`c` bounds the averaging error. An active phase group has starting probability mass above the code's cutoff. `r` counts these groups, and `delta_chord` is their minimum eigenvalue separation on the complex unit circle.

### Lumping and route selection

Lumping combines equivalent nodes into blocks to reduce the calculation size. `src/quantum/lumping.py` groups individuals with the same category state in each layer and the same masked label in `{0, 1, UNK}`.

A masked label hides an outcome during evaluation. Visible non-defaults have label `0`, visible defaults have label `1`, and hidden outcomes have label `UNK`, stored as `-1`.

`src/quantum/lumped_operator.py` builds the reduced operator. In the formulas below, `C` sums node probabilities into blocks, `M` contains block sizes on its diagonal, and `B` distributes normalized block coordinates.

`G-bar` is the grouped transition-and-restart operator. `D-bar` is its discriminant. Superscript `T` means matrix transpose. With `B = C^T M^(-1/2)`, `src/quantum/lumping_router.py` checks `max|CG - G-bar C|`. This residual is the largest absolute entry in the difference between aggregating before and after applying the operator. The router rejects an operator whose residual exceeds its tolerance. It then computes `max|DB - B D-bar|` and `H = B^T D B`.

The second residual tests whether the grouped discriminant reproduces the full discriminant's action on block coordinates. `H` expresses the full discriminant in those coordinates.

`route_operator` chooses `K = D-bar` when the compatibility residual is at most `router_tol`, whose default is `1e-12`. Otherwise it chooses `K = H`. The route records which matrix the computation uses.

A reference class with at least two members and a dangling copy can fail compatibility. `scripts/ec_counterexample.py` demonstrates this with eight individuals. The full comparison uses the same builder with the same conventions, `lumping_level="none"`.

### Convergence point and score columns

`src/quantum/mechanism/convergence.py` checks horizons 1 through 1,000, then every tenth step through 10,000, then every hundredth step to `tau_max`. It includes the cap itself.

`tau*` is the first grid point after which all remaining grid points satisfy three criteria. Default thresholds are L1 distance at most 0.01, test-borrower Spearman at least 0.99, and expected top-k overlap within 0.01 of self-overlap. L1 measures probability differences across network nodes. Spearman correlation measures agreement between borrower rankings. Expected top-k overlap measures shared membership among the `k` highest-scoring borrowers, accounting for tied scores.

Self-overlap compares a ranking with itself under the same tie rule. It provides the overlap ceiling when ties prevent a unique top-k set. The valid tail must contain at least 20 grid points. A tail is the sequence from the candidate convergence point through the final checked point.

If these conditions fail, the result is `> tau_max`, with the last violation and tail length. Default caps are 50,000 for HMEQ and 200,000 for Freddie Mac. The tables also report `tau_star_l1`, which uses only L1, and `tau_star_rank`, which uses the ranking criteria. Tolerance sensitivity shows how these results change when the acceptance thresholds change.

The code treats discriminant eigenvalues within its configured tolerance of `-1` or `1` as boundary values. `d_collapsed` counts these values. `d_inner_max` is the largest remaining eigenvalue magnitude, and `d_gap_min` is its distance from 1. The code warns when that distance is below `1e-6`.

`src/quantum/mechanism/scores_stage.py` writes `scores_<network>.parquet`, `convergence_<network>.json`, and convergence curves. Parquet stores the score table, while JavaScript Object Notation, or JSON, stores convergence metadata. The observation flags identify missing categories and partially observed borrowers.

| Score column | Meaning |
|---|---|
| `classical` | Stationary classical probability summed over each borrower's layer copies |
| `qppr_limit` | Quantum time-average limit measured on the source register |
| `qppr_limit_destination` | The corresponding limit measured on the destination register, used for comparison |
| `qppr_tau_star` | Quantum time average at the evaluation length `tau_eval` |
| `classical_tau_star` | Classical time average from `v` at the same evaluation length |

`qppr_tau_star` uses `tau_eval`. That value is `tau*` if reached, otherwise `tau_max`. `T_convergence` and the stage manifest record this distinction.

## Stage behavior

### Stage 04 computes quantum scores

`src/stages/quantum.py` reads stage 03's data tables with evaluation labels hidden. It checks their digests, which are hashes used to detect changes. It checks the full walk against the reduced walk. Checkpoints save intermediate progress and identify the operator used, so incompatible results are not reused. The default check has 21 recorded steps.

The stage writes scores, limits, convergence results, `T_convergence`, `EC_convergence_sensitivity`, `EC_convergence_curves`, and `EC_reduction_residuals`. The step check covers all HMEQ networks and the Freddie Mac two-layer network.

### Stage 05 evaluates the scores

`src/stages/evaluation.py` writes `T2_agreement`, `EC_registers`, `EC_classical_horizon`, `T2_decision`, `EC_decision_full`, and `EC_strata`. QPPR-minus-classical comparisons use the shared bootstrap. A bootstrap repeatedly samples test borrowers with replacement to estimate uncertainty. Both methods use the same samples for each comparison.

`src/evaluation/controls.py` rebuilds both operators for each reference-permutation placebo. The default has 99 permutations and uses `p = (1 + count(placebo >= true)) / (1 + B)`. Here, `B` counts permutations, not the block-coordinate matrix defined earlier. The numerator counts random-reference results at least as large as the observed statistic, plus one.

`src/evaluation/downstream.py` checks whether the scores help prediction models. Cross-fitting computes each development borrower's score with that borrower's fold excluded from reference labels. A percentile expresses the score's rank relative to scores from the same operator.

The models are logistic regression and Extreme Gradient Boosting, or XGBoost. Logistic regression models outcome probabilities; XGBoost combines decision trees. Paired differences compare models on the same samples. Layer-dummy importance measures contributions from binary indicators of layer categories. These results belong to the electronic companion, the paper's supplementary material.

### Stage 05a tests mechanisms

`src/stages/mechanism.py` writes `T_eta`, `T3_variants`, `EC_alpha_eta_grid`, `T4_partial`, and `EC_single_layer_share`.

Own-copy restart sends each reference borrower's restart probability to that borrower's layer copies. Rank-one restart uses the same distribution `v` for every node. Flattened networks combine layers into one network.

`eta` is the connection weight between layer copies. Setting `eta = 0` removes those connections. Partial copies are layer copies without a category membership edge. The classical grid tests combinations of damping and interlayer weights. `T4_partial` includes a label-free control. Its restart, dangling-node replacement, and initial distribution are uniform over borrower copies, without default-based weighting. It reads placebo Jaccards and p-values from stage 05's `EC_placebo` table. Jaccard similarity divides the intersection size of two selected sets by their union size. The placebo p-value compares the observed statistic with random-reference results.

### Stage 06 simulates circuits

A logical circuit describes operations before mapping them to device qubits. Register width counts qubits. State-preparation angles are circuit rotation parameters that encode probabilities.

Matrix density is the fraction of nonzero entries; support identifies those entries. A pure-noise band gives metric ranges from uniformly sampled register codes. It provides a comparison for output with no network signal. Raw results use measured probabilities without correction; mitigated results correct estimated measurement errors.

`src/stages/hardware.py` and `src/quantum/hardware/` produce the following outputs:

| Output | Contents |
|---|---|
| `EC_circuit_resources.csv` | Class counts, state counts, register widths, qubits, density, support, preparation angles, and two-qubit gate counts |
| `EC_statevector.csv` | Logical HMEQ statevector comparisons against the NumPy walk |
| `EC_noise_grid.csv` | Raw and mitigated results, status, resource estimates, route diagnostics, and pure-noise bands |
| `hardware_candidates.json` | Candidate selection, calibration convention, and rankability results |
| Device-mapped circuit files | Circuits translated into the target device's supported gates and qubit connections |

Result CSVs go to `results/outputs/results/<dataset>/`. Candidate metadata, circuit files, and grid shards go under the dataset's stage 06 directory in `results/outputs/notebook_pipeline/`.

The full configuration uses 4,096 shots and recorded steps 1 through 5. At `tau = 1`, the circuit measures the prepared state without a walk step. Candidate selection therefore requires at least `hardware_min_tau = 2`.

An arm is one noise setting. The noise arms are `noiseless_shots`, `aer_noise_model`, and `fake_marrakesh`. `noiseless_shots` samples circuit outcomes without simulated device errors. The middle arm uses depolarizing and readout noise at device medians, without thermal relaxation.

Depolarizing noise mixes a state with random quantum states. Readout noise changes the reported measurement outcome. Thermal relaxation models energy loss and loss of phase information over time. The last arm uses error parameters from the recorded device calibration.

Readout mitigation estimates measurement errors using known input states, then corrects the measured probabilities. Raw results have no such correction. Circuit routing maps operations onto device qubits and can change which qubits are measured.

Each distinct group of measured qubits therefore gets its own calibration. A supra-node is one node in the multilayer graph, such as a borrower-layer copy or category node.

| Metric | Meaning |
|---|---|
| `spearman_supra` | Rank correlation between measured and expected probabilities across supra-nodes |
| `spearman_masked_classes` | Rank correlation across borrower classes with hidden labels, after summing layer copies |
| `tv` | Total variation: half the sum of absolute differences between measured and expected probabilities |
| `tv_uniform` | Total variation between equal probabilities on all retained nodes and the expected distribution |
| `fidelity_ratio` | `1 - tv / tv_uniform`. A value of 1 means agreement; 0 means the same distance as uniform probabilities. |
| `padding_mass` | Measured probability assigned to binary codes that do not represent retained network nodes |
| `depth_max` | Maximum number of sequential gate layers across the circuits |
| `two_qubit_sum` | Total number of gates acting on two qubits across the circuits |
| `predicted_survival_min` | Lowest estimated probability of avoiding gate errors across the circuits, using the configured error model |
| `estimated_qpu_seconds` | Estimated device time for circuit repetitions and measurement calibration, not measured simulator runtime |

The fidelity ratio can be negative. It is undefined when `tv_uniform` is zero, and it is not quantum-state fidelity.

The pure-noise band's endpoints are configured quantiles, meaning cutoff values at specified percentages of the sampled metric results.

A register's width is its number of qubits. Its binary codes can outnumber the network nodes, which creates padding codes. The code removes padding outcomes and rescales the remaining probabilities before comparison.
