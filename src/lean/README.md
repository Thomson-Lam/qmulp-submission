# QPPR Lean formalization

Lean 4 formalization of Propositions 1, 2, and 3 of the quantum personalized
PageRank (QPPR) manuscript, together with supporting results from the electronic
companion. The Lake package version is `0.7.0`.
The proof checks mathematical statements under their hypotheses, not the Python implementation or empirical results.

## Requirements

- Lean 4.31.0, as specified in `lean-toolchain`, with Lake available on your path.
- Git, so Lake can download the dependencies.
- Python 3 when using the Bash audit script. The Windows scripts do not require it.

The exact Mathlib revision and transitive dependency revisions are pinned in
`lake-manifest.json`. Keep this file under version control for reproducibility.

## Build and audit

Open a terminal in the project directory containing `lakefile.toml`. To build
the library, run:

```sh
lake build
```

The audit scripts build the library and then check the selected theorem
statements and their axiom dependencies. Run the script for your environment.

### Windows command prompt

```cmd
scripts\audit.cmd
```

### Windows PowerShell

```powershell
.\scripts\audit.ps1
```

### Linux and macOS

```sh
bash scripts/audit.sh
```

The audit writes `AUDIT_RESULT.txt` and checks 19 selected final and auxiliary
theorems. Every audit target must be a theorem, and its transitive axiom
dependencies may contain only `propext`, `Classical.choice`, and `Quot.sound`.
A successful run prints `AXIOM AUDIT PASSED: 19 theorems` in the log.

The `scripts/verify.cmd` and `scripts/verify.ps1` wrappers invoke the corresponding
audit scripts. On the first build, Lake downloads the dependencies. When updating
an existing project, a compatible `.lake` directory can be retained.

## Formalized results

| Proposition | Result |
| --- | --- |
| P1 | Spectral limit, complete eigenspace projectors, an explicit L1 bound of the form `c/tau`, and computation from the eigendecomposition of `D`. |
| P2 | Agreement between aggregated full-walk scores and quotient scores for both registers at every recorded time, in finite averages, and in limiting averages, derived from `CG = Gbar C` and `DB = B Dbar`. The embedding `B` and aggregated personalization are constructed from the blocks. |
| P3 | Classical and quantum symmetry at every recorded time, in finite averages, and in limiting averages, including existence of both limits. |

The hypotheses appear in the theorem declarations, not in a separate status document:

- [ConcreteLimit.lean](QPPR/Manuscript/ConcreteLimit.lean) states the spectral limit and rate results for a finite stochastic operator.
- [CompletePropositions.lean](QPPR/Manuscript/CompletePropositions.lean) states the combined proposition results.
  Quotient results require a surjective block map, classical lumpability, discriminant intertwining, and a starting vector constant within each block.
- `ExchangeTransitive` in `CompletePropositions.lean` supplies the extra hypothesis for individual score recovery.
  Each pair of members in a block must be related by a permutation preserving both `G` and `v`.
  Under this hypothesis, the member's source-register score equals the block total divided by block size.
- Proposition 3 requires a permutation preserving both `G` and `v`.
  It proves equal scores under that permutation and existence of both classical and quantum time-average limits.

## Project contents

| Path | Purpose |
| --- | --- |
| `QPPR.lean` | Main library entry point. |
| `QPPR/Core/` | Shared evolution and limit definitions. |
| `QPPR/EC1/` through `QPPR/EC4/` | Supporting results for the electronic companion. |
| `QPPR/Manuscript/` | Concrete operators, complete proposition proofs, and the axiom audit. |
| [CompletePropositions.lean](QPPR/Manuscript/CompletePropositions.lean) | Proposition statements, hypotheses, and member-recovery results. |
| [Audit.lean](QPPR/Manuscript/Audit.lean) | The 19 theorem targets and permitted axiom dependencies. |
| `lakefile.toml`, `lake-manifest.json`, `lean-toolchain` | Project configuration, dependency revisions, and Lean version. |

## Verification evidence

[BUILD_RESULT.txt](BUILD_RESULT.txt) and [AUDIT_RESULT.txt](AUDIT_RESULT.txt)
are the records supplied with the proof project.
The local audit scripts above overwrite `AUDIT_RESULT.txt`.

The submission's [build log](../../results/lean/BUILD_RESULT.txt) records a successful build with 2,967 jobs.
Its [audit log](../../results/lean/AUDIT_RESULT.txt) records all 19 targets passing the axiom check.
These logs describe the import verification run, not a new run after documentation edits.

To generate submission logs without overwriting the supplied records, run from the repository root:

```sh
bash scripts/verify-lean.sh
```

The [submission wrapper](../../scripts/verify-lean.sh) writes both logs under `results/lean/` and exits nonzero if the build or audit fails.
It uses Bash, `tee`, and `grep`, and does not require Python.
To download Mathlib's prebuilt cache before building, run `lake exe cache get` from `src/lean/`.
