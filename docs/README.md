# Pipeline documentation

The pipeline prepares credit data, builds borrower networks, computes classical and quantum scores, and compares the results. This page points to the run instructions and the code behind each step.

## Where to start

| Your task | Guide |
|---|---|
| Install dependencies and run the submission | [Root README](../README.md) |
| Obtain inputs and check field meanings, hashes, and access terms | [Data README](../data/README.md) |
| Prepare data, build networks, and compute classical scores | [Classical pipelines](classical-pipelines.md) |
| Compute quantum scores, run circuit simulations, and assemble paper outputs | [Quantum pipelines](quantum-pipelines.md) |
| Inspect the recorded local checks and their limits | [Verification record](../results/verification/README.md) |
| Read the classical graph and PageRank implementation | [MuLP module guide](../src/classical/mulp/README.md) |
| Build the Lean proofs and check theorem hypotheses | [Lean project README](../src/lean/README.md) |

## Run entry points

Commands in the guides run from the repository root, not from `docs/`. They require the environment and inputs described in the root README.

| Command | What it runs |
|---|---|
| `bash scripts/reproduce.sh classical` | HMEQ stages 01 through 03 with smoke settings |
| `bash scripts/reproduce.sh hardware` | Local stage 06 simulation using inputs from the classical smoke run |
| `bash scripts/reproduce.sh paper` | Clean-start check, both datasets, all stages, and manuscript assembly |
| `bash scripts/reproduce.sh tests` | The pytest suite |

The two smoke commands do not run quantum scoring and evaluation in stages 04, 05, and 05a. The quantum guide gives the command for those stages. Smoke results are execution checks, not paper estimates. The hardware command does not submit a physical QPU job.

The paper command requires the Freddie Mac extracts as well as HMEQ (see `data/README.md`).

## How execution reaches the code

[scripts/run.sh](../scripts/run.sh) selects the Python interpreter and adds `src/` to the import path. [scripts/run_all.py](../scripts/run_all.py) selects datasets and stages, then executes the stage notebooks. Each notebook calls a driver under `src/stages/`.

The driver passes intermediate files to the next stage. Stage 03 saves masked data and operator digests so that stage 04 can check that both methods use the same operator.

| Stage | Purpose | Driver |
|---|---|---|
| 01 | Clean raw data and create the split and bootstrap indices | [data.py](../src/stages/data.py) |
| 02 | Build networks and check their reduced representations | [network.py](../src/stages/network.py) |
| 03 | Compute classical scores on masked labels | [classical.py](../src/stages/classical.py) |
| 04 | Compute quantum limits and check convergence | [quantum.py](../src/stages/quantum.py) |
| 05 | Compare rankings, default detection, and controls | [evaluation.py](../src/stages/evaluation.py) |
| 05a | Test changes to coupling, restart, and partial observations | [mechanism.py](../src/stages/mechanism.py) |
| 06 | Simulate circuits and report noise and resource limits | [hardware.py](../src/stages/hardware.py) |

Stage 05a reads the placebo results from stage 05 and fails if they are missing. Before the stage 06 notebook, `run_all.py` computes missing noise-grid shards.

Stages 07 through 10 run scripts for the counterexample, synthetic experiments, variability checks, and runtime measurements. The figure and table scripts then assemble the manuscript files. Check `README.md` in the repo root for a list of the scripts and their result inputs.

## Where to find configuration and results

[src/config.py](../src/config.py) defines experiment settings and seeds. [src/fingerprint.py](../src/fingerprint.py) defines the reduced smoke settings. Dataset definitions are in [src/hmeq/config.py](../src/hmeq/config.py) and [src/fmac/config.py](../src/fmac/config.py).

[src/paths.py](../src/paths.py) anchors file locations to the repository root.
Generated Python pipeline outputs under `results/outputs/` are excluded from Git and must be regenerated after a fresh checkout.
The saved verification records under `results/verification/` and `results/lean/` are separate from those outputs.

| Path relative to the repository root | Contents |
|---|---|
| `data/` | Local raw inputs and generated processed data |
| `results/outputs/notebook_pipeline/` | Stage intermediates, operator digests, and manifests |
| `results/outputs/results/<scope>/` | Result CSVs for `hmeq`, `fmac`, `synthetic`, or `runtime` |
| `results/outputs/executed_notebooks/` | Executed copies of the stage notebooks |
| `results/outputs/timings/` | Timings for each driver invocation |
| `results/outputs/manuscript/` | Generated figures, table fragments, and manuscript numbers |

The stage manifests record the inputs, configuration, software versions, and execution details.
