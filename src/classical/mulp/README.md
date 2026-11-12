# Classical MuLP implementation

This package builds multilayer borrower graphs and computes personalized PageRank.
`__init__.py` exports `MultiLayerRanker` from `mulp.py`.
The pipeline uses one or two layers, with names supplied by each dataset's network specification.

## Ranker API

`MultiLayerRanker(dataset, bidirectional=True, sparse=True, influence_type="combined", stickiness=1.0, layer_names=("JOB", "REASON"))`
loads a processed Parquet file and initializes the graph index and component builders.
The constructor validates the dataset path, influence type, stickiness, layer order, and category encoding.

| Method | Behavior |
|---|---|
| `buildAdjMatrix()` | Builds and stores the adjacency matrix. |
| `toDf(matrix=None, f=None)` | Converts the supplied matrix, or stored adjacency, to a labeled pandas DataFrame. Writes Parquet when `f` is supplied. |
| `buildPersonalizationMatrix()` | Builds restart entries for borrowers with visible `BAD == 1`. Requires adjacency and rejects repeated initialization. |
| `setInfluenceType(new_type)` | Validates and stores `inter`, `intra`, or `combined`. Does not rebuild existing matrices. |
| `buildTransitionMatrix()` | Builds and stores the column-stochastic transition matrix. Requires adjacency and personalization. |
| `buildGoogleOperator(alpha=0.85)` | Builds the matrix-free operator `G = alpha*T + (1-alpha)*P`. Requires `0 < alpha < 1`. |
| `pageRank(alpha=0.85)` | Computes and stores the stationary probability vector using SciPy's ARPACK eigensolver. |
| `finiteHorizon(tau, alpha=None)` | Returns the `(tau, N)` history `G^t v` for `0 <= t < tau`. Its mean is the finite-horizon score. |
| `getLeadingEigenvectorNorm()` | Returns the stored stationary vector. |
| `aggregateBorrowers(vec)` | Sums a node vector across each borrower's layer copies. |
| `getLayerScores()` | Returns category-node maxima for each layer and summed borrower scores. Missing memberships receive zero category scores. |
| `getScores()` | Returns the two-layer tuple `(job_max, reason_max, borrower_scores)`. Rejects single-layer networks. |
| `addScores(job_scores, reason_scores, borrower_scores)` | Appends score columns to the ranker's Polars DataFrame. |

`pageRank()` checks the stationary residual against `1e-9`, negative entries against `-1e-10`, and unit-sum error against `1e-12`.
It clips accepted negative roundoff to zero and renormalizes the vector.

The ranker exposes `last_adjacency_diagnostics`, `last_personalization_diagnostics`, and `last_transition_diagnostics` after the corresponding builds.

## Configuration and category discovery

`config.py` provides `validate_dataset_path()`, `validate_stickiness()`, and `normalize_influence_type()`.
Stickiness accepts finite positive values and positive infinity.
It rejects booleans, NaN, negative infinity, and nonpositive values.

`dataset.py` defines `MuLPDataset` and `MuLPDatasetLoader`.
The loader accepts one or two distinct layer names in the frame's column-block order.
It discovers sorted `<LAYER>_*` columns and separates genuine categories from `Missing` markers.
It excludes columns ending in `_was_missing` from graph categories.
There is no fixed requirement for seven JOB columns or three REASON columns.

`check_at_most_one_hot()` rejects rows with more than one active category and counts rows without an active category.
`active_pairs()` returns borrower-category pairs where the category indicator equals one.
A missing category produces no membership edge.

## Node indexing and adjacency

`indexing.py` defines `LayerIndex` and `MuLPGraphIndex`.
For `L` layers, `layer_size = n_rows + total_states` and `N = L * layer_size`.
`total_states` counts genuine categories across all layers.
Each layer reserves category positions for every layer, but only its own category positions receive edges.
A single-layer network has no unused category positions.

`MuLPGraphIndex.from_dataset()` delegates to `from_layers()`.
The index provides `offset()`, `borrower_node()`, `category_node()`, and ranges for borrower and category nodes.
`placeholder_nodes()` returns the unused category positions.

`adjacency.py` defines `SupraAdjacencyBuilder` and `AdjacencyDiagnostics`.
`A[i, j]` stores the edge weight from source `i` to destination `j`.
Membership edges have weight 1 and follow the `bidirectional` setting.
With two layers, each borrower's copies connect in both directions with weight `stickiness`, regardless of that setting.
Category nodes do not connect across layers.

The builder returns a SciPy compressed sparse row matrix when `sparse=True`, or a NumPy array otherwise.
Diagnostics record dimensions, edge support, membership counts, and coverage for each layer.

## Personalization and restart

`personalization.py` defines `PersonalizationMatrixBuilder`, `PersonalizationDiagnostics`, and `FactoredRestart`.
The builder requires at least one borrower with visible `BAD == 1`.
Each such borrower supplies entries only among its own layer copies:

- `intra` creates self-loops.
- `inter` links different copies of the same borrower. Single-layer networks reject this setting.
- `combined` includes self-loops and cross-copy links.

The resulting raw matrix `R` is not the complete stochastic restart matrix `P`.
`normalized_restart()` normalizes nonempty columns and marks empty columns for replacement by the reference distribution `v`.
`FactoredRestart` stores those columns, the empty-column mask, and `v` without constructing a dense outer product.

For combined restart, `v` assigns `1 / (L * |D|)` to every layer copy of each visible default borrower in `D`.
A reference column of `P` assigns `1/L` to that borrower's copies. Other columns equal `v`.

## Transition normalization

`transition.py` defines `TransitionMatrixNormalizer` and `TransitionDiagnostics`.
The normalizer divides each nonempty adjacency row by its sum, then transposes the result.
Each resulting source column sums to one.

`dangling_patch_vector()` derives `v` from the personalization matrix's normalized row sums.
Missing personalization or nonpositive total restart mass raises `ValueError`. There is no uniform fallback.
Dangling source columns use `v`.

When a row contains positive infinity, normalization splits probability equally among those entries and ignores its finite weights.
NaN, negative infinity, and negative finite weights raise `ValueError`.
The result is a sparse matrix or a dense array matching the requested representation.
Diagnostics record the dangling-column count and maximum column-sum error.

## Consumers and tests

- `src/fingerprint.py` constructs rankers from dataset and experiment settings.
- `src/classical/scores.py` and `src/classical/network.py` compute classical scores and network diagnostics.
- `src/quantum/inspired/mulp.py` and `src/quantum/inspired/szegedy_walk.py` use the classical operator for quantum-walk comparisons.
- `tests/test_operator_contract.py` tests restart, transition, layer, and stationary-score behavior.
- `tests/test_lumping.py` and `tests/test_circuits.py` compare reduced operators and circuits with the full operator.

These paths are relative to the repository root.
