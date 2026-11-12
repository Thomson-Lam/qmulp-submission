# README

This folder contains the classical MuLP graph construction and PageRank scoring implementation.

- `__init__.py`

Package export shim.

Exports:

- `MultiLayerRanker` — re-exported from `classical.mulp.mulp`.

### `mulp.py`

High-level compatibility facade for the MuLP workflow. This file owns the public `MultiLayerRanker` class and keeps the historical methods/attributes used by the stages, preprocessing, the tests, and the quantum-walk wrapper.

Class:

- `MultiLayerRanker`
  - Orchestrates dataset loading, graph indexing, adjacency construction, personalization construction, transition normalization, PageRank solving, score extraction, and score-column insertion.
  - Delegates implementation details to `config.py`, `dataset.py`, `indexing.py`, `adjacency.py`, `personalization.py`, and `transition.py`.
  - Preserves mutable public state such as `df`, `adj_matrix`, `personalization_matrix`, `transition_matrix`, `leading_eigenvector_norm`, `labels`, `job_cols`, `reason_cols`, `n_rows`, `n_job`, `n_reason`, `job_offset`, `reason_offset`, `N`, `influence_type`, `sparse`, `bidirectional`, and `stickiness`.
  - Exposes diagnostic hooks: `last_adjacency_diagnostics`, `last_personalization_diagnostics`, and `last_transition_diagnostics`.

`MultiLayerRanker` methods:

- `__init__(dataset, bidirectional=True, sparse=True, influence_type="combined", stickiness=1.0)` — validates the parquet path/stickiness, loads the dataset, builds graph indexing metadata, and initializes component builders.
- `_validate_stickiness(stickiness)` — compatibility wrapper around `config.validate_stickiness()`.
- `_checkOHE(cols, group_name)` — compatibility wrapper around `MuLPDatasetLoader.check_ohe()`.
- `_build_labels()` — compatibility helper that rebuilds JOB/REASON borrower/category labels from current ranker metadata.
- `_active_pairs(frame, cols)` — compatibility wrapper around `MuLPDatasetLoader.active_pairs()`.
- `buildAdjMatrix()` — builds and stores the raw supra-adjacency matrix using `SupraAdjacencyBuilder`.
- `toDf(matrix=None, f=None)` — converts a sparse/dense matrix to a pandas DataFrame with MuLP labels; optionally writes parquet.
- `buildPersonalizationMatrix()` — builds and stores the BAD-borrower restart/influence matrix using `PersonalizationMatrixBuilder`.
- `setInfluenceType(new_type)` — validates and stores one of `inter`, `intra`, or `combined`.
- `_borrower_pair_by_source()` — returns a mapping between JOB and REASON borrower-copy nodes.
- `_dangling_patch_vector()` — returns the vector used to patch dangling transition columns.
- `_column_normalize_adjacency()` — compatibility method for column-normalizing `adj_matrix`; qwalk and validation code currently call this private method.
- `buildTransitionMatrix()` — public alias that builds/stores the normalized transition matrix.
- `pageRank(alpha=0.85)` — constructs the historical PageRank operator and stores the normalized leading eigenvector.
- `getLeadingEigenvectorNorm()` — returns the stored PageRank vector.
- `_source_neighbors(source_idx, sparse)` — returns outgoing neighbors under the column-stochastic convention.
- `getScores()` — derives per-row JOB max, REASON max, and borrower summed scores from the stored eigenvector.
- `addScores(job_scores, reason_scores, borrower_scores)` — appends score columns to `self.df` and returns the updated Polars DataFrame.

Important invariants:

- Adjacency convention is `A[i, j]` = edge weight from source node `i` to destination node `j` (row = source, column = destination); the transition matrix is column-normalized from it, so each column sums to 1.
- Adjacency/transition/personalization matrices follow the paper's block supra-adjacency layout (Eq. 1-3, see `indexing.py`): `N = 2 * layer_size` with `layer_size = n_rows + n_job + n_reason`, and each layer's block reserves (but never fills) a placeholder segment for the other layer's category nodes.
- Current HMEQ schema expects exactly 7 `JOB_*` columns and 3 `REASON_*` columns.
- `stickiness` accepts finite `S > 0` and positive infinity.
- `influence_type` accepts `combined`, `intra`, and `inter`.

### `config.py`

Validation and configuration helpers.

Constants:

- `VALID_INFLUENCE_TYPES` defines a tuple of accepted influence types: `inter`, `intra`, `combined`.

Functions:

- `validate_dataset_path(dataset)`: confirms the dataset path exists and has `.parquet` suffix.
- `validate_stickiness(stickiness)`: validates finite positive stickiness values and positive infinity; rejects booleans, NaN, negative infinity, and non-positive values.
- `normalize_influence_type(value, *, message)`: lowercases and validates an influence type using a caller-supplied error message.

Class:

- `MuLPConfig`
  - Dataclass holding `dataset`, `bidirectional`, `sparse`, `influence_type`, and `stickiness`.
  - `__post_init__()` validates and normalizes `stickiness` while preserving the historical behavior that invalid influence types are rejected later when set or used.

### `dataset.py`

Parquet loading and one-hot utility logic.

Classes:

- `MuLPDataset`
  - data class for containing `path`, data loading for Polars, `job_cols`, and `reason_cols`.

- `MuLPDatasetLoader`
  - Stateless loader/validation utility.

Methods:

- `load(dataset)`: validates the parquet path, reads the Polars DataFrame, discovers sorted `JOB_*` and `REASON_*` columns, and returns `MuLPDataset`.
- `check_ohe(frame, cols, group_name)`: validates exactly one active one-hot value per row for a layer column group.
- `active_pairs(frame, cols)`: converts one-hot columns to `(borrower_id, category)` active pairs for graph edge construction.

### `indexing.py`

This file handles any operations for indexing or finding nodes and the related data for the corresponding node.

Class:

- `MuLPGraphIndex`
  - Frozen dataclass containing graph dimensions, offsets, column groups, and labels.
  - Fields include `n_rows`, `job_cols`, `reason_cols`, `n_job`, `n_reason`, `layer_size`, `job_offset`, `reason_offset`, `N`, and `labels`.
  - Matches the paper's block supra-adjacency matrix (Eq. 1-3): each layer reserves a complete copy of the shared identity space `layer_size = n_rows + n_job + n_reason` (borrowers plus every category node across *both* layers), not just its own borrowers/categories. Within a layer's block, only that layer's own category segment gets real edges; the other layer's category segment is a reserved but untouched (all-zero, dangling) placeholder segment. Total `N = 2 * layer_size`.

`MuLPGraphIndex` methods:

- `from_dataset(dataset)`: validates expected JOB/REASON column counts, computes `layer_size`/offsets/total node count, and builds labels.
- `build_labels(n_rows, job_cols, reason_cols)`: creates labels for JOB borrower nodes, JOB category nodes, REASON borrower nodes, REASON category nodes, and the placeholder segments in each layer's block.
- `job_category_node(category_index)` / `reason_category_node(category_index)`: compute the global node id of a category node, accounting for the placeholder segment gap.
- `borrower_pair_by_source()`: maps every borrower-copy node to its same-borrower paired node in the other layer.

### `adjacency.py`

The raw supra-adjacency matrix lives and is built here.

Classes:

- `AdjacencyDiagnostics`
  - Dataclass with build metadata: matrix shape, sparse flag, bidirectional flag, stickiness, support count, and layer dimensions.

- `SupraAdjacencyBuilder`
  - Builds the raw JOB/REASON supra-adjacency matrix.
  - Stores the most recent `AdjacencyDiagnostics` on `self.diagnostics`.

`SupraAdjacencyBuilder` methods:

- `__init__()`: initializes with empty diagnostic values.
- `build(frame, index, *, sparse, bidirectional, stickiness)`: validates OHE layer columns, adds borrower-to-category edges, adds inter-layer borrower-copy edges with stickiness weight, logs stickiness behavior, stores diagnostics, and returns the sparse or dense adjacency matrix.

Notes:

- Sparse matrices are constructed as SciPy `lil_matrix`.
- Dense matrices are NumPy `float64` arrays.
- Bidirectional mode makes 2 edges for each inserted edge.

### `personalization.py`

This file is where the personalization matrix is.

Classes:

- `PersonalizationDiagnostics`
  - Dataclass with build metadata: shape, sparse flag, influence type, BAD borrower count, and nonzero count.

- `PersonalizationMatrixBuilder`
  - Builds the personalization matrix used by historical MuLP PageRank scoring.
  - Stores the most recent `PersonalizationDiagnostics` on `self.diagnostics`.

`PersonalizationMatrixBuilder` methods:

- `__init__()`: initializes with empty diagnostic values.
- `build(frame, index, *, sparse, influence_type)`: validates `BAD`, finds BAD borrower rows, builds restart entries for `inter`, `intra`, or `combined`, stores diagnostics, and returns the sparse or dense personalization matrix.

Values for personalization matrix as per paper:

- `inter`: restart links only between same-borrower copies across layers.
- `intra`: restart links only to same-layer BAD borrower copies.
- `combined`: includes both same-layer and cross-layer BAD borrower restart entries.

### `transition.py`

Handles building the transition matrix, and dangling nodes patches.

Classes:

- `TransitionDiagnostics`
  - Dataclass with normalization metadata: shape, sparse flag, dangling column count, and max column-sum error.

- `TransitionMatrixNormalizer`
  - Converts raw supra-adjacency into a finite column-stochastic transition matrix.
  - Stores the most recent `TransitionDiagnostics` on `self.diagnostics`.

`TransitionMatrixNormalizer` methods:

- `__init__()`: initializes with empty diagnostic values.
- `dangling_patch_vector(personalization_matrix, N)`: derives a probability vector from personalization row sums when available, otherwise returns a uniform vector.
- `normalize(adj_matrix, personalization_matrix, N)`: validates adjacency values, handles finite and infinite stickiness weights, patches zero columns, stores diagnostics, and returns a sparse CSR matrix or dense NumPy matrix matching the input style.

Notes:

- Positive infinity is treated as the exact `S -> infinity` limit and converted to finite probability mass during normalization.
- NaN, negative infinity, and negative finite entries are rejected.
- Dangling columns use personalization-derived mass if possible; otherwise uniform mass. Since `adjacency.py`'s block layout always reserves an untouched placeholder category segment in each layer, `dangling_column_count` is routinely `n_job + n_reason`, not just an edge case.

## Tests and validation consumers

This folder's classes and code is used by:

- `classical/preprocessing/create_features.py`: computes PPR-derived feature columns through `MultiLayerRanker`.
- `tests/classical/build_smoke.py`: graph building smoke-test adjacency construction and `toDf()` export.
- `tests/classical/validate_stickiness.py`: tests raw stickiness weights and normalized transition probabilities.
- `tests/classical/test_mulp_components.py`: checks whether the refactored/current split implementation of the MultiLayerRanker is behaving as expected.
- `tests/quantum/test_qwalk_matrix_validation.py`: validates MuLP matrix behavior for quantum walk.
- `qwalk/sim/quantum_inspired_sim.py`: consumes MuLP adjacency, personalization, indexing metadata, and transition normalization for quantum-walk matrix construction.
