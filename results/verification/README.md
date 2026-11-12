# Reproducing locally

These checks compare the submission code with the original `qmulp` implementation. The original implementation is called the canonical reference below.

Follow the [installation and data instructions](../../README.md) first. Obtain the Home Equity dataset, or HMEQ, as `data/hmeq.csv`. Run commands from the `sub-qmulp/` root.

Before an independent run, archive previous files in `results/outputs/`. Keep outputs between the following commands because the hardware check needs the classical run's inputs.

```bash
bash scripts/reproduce.sh classical
bash scripts/reproduce.sh hardware
bash scripts/reproduce.sh tests
```

The classical and hardware commands use smoke mode, which reduces experiment settings to check execution. These outputs are not paper estimates. Hardware means local circuit simulation, not execution on a quantum processing unit, or QPU.

The active outputs are `results/outputs/`. [run-artifacts/](run-artifacts/) holds saved verification outputs. Git retains selected result tables, timing records, stage manifests, and operator digests from that directory. Other locally saved files are ignored and are not available from a fresh checkout.

A manifest records a stage's configuration, inputs, outputs, and execution details. An operator digest is a file-content hash used to detect changes in the calculation's inputs.

[hardware.log](hardware.log) records the initial simulations. [hardware-rerun.log](hardware-rerun.log) records the subsequent run. A shard stores simulation results for one candidate network. The later run reuses saved shards after checking their configuration and operator identities.

The four saved stage manifests record commit `43847734bcbc618fa65ee8229c75ec401baa1eb8` and `dirty: false`. A commit identifies a saved Git revision. The dirty flag reports uncommitted changes, including untracked files not excluded by Git's ignore rules.

They also record source digest `7ab978246ac5eee0c6addd1a4177ba9f131d4dd4e26c5452e48fffe5cafed2b9`, a hash identifying the source contents used for that run. These fields describe the historical run, not the current checkout. They do not establish that later edits leave results unchanged.

`scripts/verify-port.sh` cannot run in this checkout because `docs/canonical-sha256.json` and `docs/port-manifest.json` are absent. That script checks file integrity, not numerical results. It is not required for the commands above.

## Reproduce the canonical comparison without modifying qmulp

The comparison uses a temporary clone, which is a separate copy of the original Git repository. Both implementations use the same Python environment and HMEQ input.

The commands below run tests and experiments in the clone, not in `../qmulp`. They require that original repository beside `sub-qmulp`. First complete the submission's classical and hardware checks above.

The checkout command selects the original revision used for the port. Without that step, cloning would select the source repository's current default revision.

```bash
PORT="$PWD"
REFERENCE="$(mktemp -d)/qmulp"
git clone --no-hardlinks ../qmulp "$REFERENCE"
git -C "$REFERENCE" checkout --detach aa99d177dc49e9fb419eea32f9d337465deabd8f
mkdir -p "$REFERENCE/data"
cp data/hmeq.csv "$REFERENCE/data/hmeq.csv"
(
  cd "$REFERENCE"
  export PYTHONPATH="$REFERENCE"
  "$PORT/.venv/bin/python" -m pytest tests/ -q
  "$PORT/.venv/bin/python" -m scripts.run_all --dataset hmeq --smoke --from-stage 01 --to-stage 03 --skip-figures
  "$PORT/.venv/bin/python" -m scripts.run_all --dataset hmeq --smoke --from-stage 06 --to-stage 06 --skip-figures
)
bash scripts/compare-runs.sh "$REFERENCE/outputs"
```

[canonical-classical.log](canonical-classical.log) and [canonical-hardware.log](canonical-hardware.log) record timings from the earlier reference run.

`scripts/compare-runs.sh` compares three classical score tables and three hardware result tables. It fails if any compared table differs. It does not compare every output file or elapsed time.

[comparison.json](comparison.json) records the earlier comparison in JavaScript Object Notation, or JSON. All six tables matched. Each classical table had 5,853 rows; the hardware tables had 11, 4, and 1 rows.

A statevector stores quantum amplitudes, whose squared magnitudes give measurement probabilities. The statevector check selects its walk length using a runtime estimate. Different machine speeds can therefore produce different lengths. The comparison script does not require those lengths to match.

## Environment and limits

[machine.json](machine.json) records the verification machine and Python version. [environment.txt](environment.txt) lists 90 installed packages and their versions. [requirements.txt](../../requirements.txt) specifies the package versions to install. The local installation transcript, `install.log`, is ignored by Git.

The recorded environment used Python 3.12.13. A virtual environment keeps the project's installed packages separate from other Python installations. The recorded `uv pip check` used the uv package manager to check dependency compatibility. It reported all 90 installed packages compatible.

The saved test logs use `<venv>` and `<python-runtime>` instead of machine-specific interpreter paths. These names denote the virtual environment and Python installation. Stage manifests express file paths relative to the repository root.

The logs describe earlier executions. They are not test results for subsequent code changes.

The complete two-dataset paper pipeline was not run because the Freddie Mac extracts are absent. [LICENSE](../../LICENSE), [AUTHORS](../../AUTHORS), and publication citation details still require author completion. The current license notice does not grant distribution rights.

These records do not establish physical-device execution, full-paper reproduction, or readiness for public submission.
