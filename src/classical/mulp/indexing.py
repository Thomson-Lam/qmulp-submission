"""Node indexing for the MuLP supra-graph (one or two layers).

Follows the paper's block supra-adjacency (Eq. 1-3): each layer reserves a
complete copy of the shared identity space ``layer_size = n_borrowers +
total_states`` -- every borrower plus every category state across *all*
layers -- but only its own category segment carries edges. The other layer's
segment stays an all-zero placeholder, which is what keeps a node's identity
the same in every layer block. With a single layer there is no placeholder.

Node ids, for layers ``0..L-1`` with ``offset(l) = l * layer_size``:

    offset(l) + k                                borrower k's copy in layer l
    offset(l) + n_borrowers + before(l) + s      layer l's category state s
    offset(l) + n_borrowers + before(m) + s      placeholder for layer m != l

Only *genuine* states get a node: the ``"Missing"`` levels are split out
upstream by ``MuLPDatasetLoader`` (see ``preprocessing.missing_levels``), so a
borrower whose field was never recorded has no membership edge in that layer
and reaches the graph through its copy in the other one.
"""

from __future__ import annotations

from dataclasses import dataclass

from preprocessing.missing_levels import is_missing_column

from .dataset import MuLPDataset, validate_layer_names


@dataclass(frozen=True, slots=True)
class LayerIndex:
    """One layer's name, its genuine category columns, and where they sit."""

    name: str
    columns: tuple[str, ...]
    position: int
    states_before: int

    @property
    def n_states(self) -> int:
        return len(self.columns)


@dataclass(frozen=True, slots=True)
class MuLPGraphIndex:
    n_rows: int
    layers: tuple[LayerIndex, ...]
    layer_size: int
    N: int
    labels: list[str]

    # -- construction --------------------------------------------------------

    @classmethod
    def from_layers(
        cls, n_rows: int, layer_columns: "list[tuple[str, list[str]]]"
    ) -> "MuLPGraphIndex":
        """Build from ``[(layer_name, genuine_columns), ...]`` in layer order."""
        validate_layer_names([name for name, _ in layer_columns])

        layers: list[LayerIndex] = []
        states_before = 0
        for position, (name, columns) in enumerate(layer_columns):
            columns = list(columns)
            if not columns:
                raise ValueError(f"No genuine categories found for layer {name!r}; cannot build it")
            stray = [c for c in columns if is_missing_column(c)]
            if stray:
                raise ValueError(
                    f"Absence markers must not be given category nodes, got {stray}; "
                    "load the dataset through MuLPDatasetLoader.load"
                )
            layers.append(
                LayerIndex(
                    name=str(name),
                    columns=tuple(columns),
                    position=position,
                    states_before=states_before,
                )
            )
            states_before += len(columns)

        layer_size = int(n_rows) + states_before
        return cls(
            n_rows=int(n_rows),
            layers=tuple(layers),
            layer_size=layer_size,
            N=len(layers) * layer_size,
            labels=cls.build_labels(int(n_rows), layers),
        )

    @classmethod
    def from_dataset(cls, dataset: MuLPDataset) -> "MuLPGraphIndex":
        return cls.from_layers(dataset.df.height, list(dataset.layer_columns))

    @staticmethod
    def build_labels(n_rows: int, layers) -> list[str]:
        labels: list[str] = []
        for layer in layers:
            labels.extend(f"{layer.name}:borrower_{i}" for i in range(n_rows))
            for other in layers:
                if other.position == layer.position:
                    labels.extend(other.columns)
                else:
                    labels.extend(f"{layer.name}_layer:unused_{c}" for c in other.columns)
        return labels

    # -- geometry ------------------------------------------------------------

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def total_states(self) -> int:
        return sum(layer.n_states for layer in self.layers)

    @property
    def layer_names(self) -> tuple[str, ...]:
        return tuple(layer.name for layer in self.layers)

    def layer(self, name: str) -> "LayerIndex | None":
        for info in self.layers:
            if info.name == name:
                return info
        return None

    def offset(self, layer: int) -> int:
        """First node id of layer ``layer``'s block."""
        return int(layer) * self.layer_size

    def borrower_node(self, layer: int, borrower_id: int) -> int:
        return self.offset(layer) + int(borrower_id)

    def borrower_nodes(self, layer: int) -> range:
        start = self.offset(layer)
        return range(start, start + self.n_rows)

    def category_node(self, layer: int, state: int) -> int:
        """Node id of layer ``layer``'s ``state``-th genuine category."""
        info = self.layers[layer]
        return self.offset(layer) + self.n_rows + info.states_before + int(state)

    def category_nodes(self, layer: int) -> range:
        first = self.category_node(layer, 0)
        return range(first, first + self.layers[layer].n_states)

    def placeholder_nodes(self, layer: int) -> list[int]:
        """Reserved, never-touched category slots inside layer ``layer``'s block."""
        base = self.offset(layer) + self.n_rows
        return [
            base + other.states_before + s
            for other in self.layers
            if other.position != layer
            for s in range(other.n_states)
        ]

    @property
    def all_placeholder_nodes(self) -> list[int]:
        return [n for l in range(self.n_layers) for n in self.placeholder_nodes(l)]

    def layer_of(self, node_id: int) -> int:
        return int(node_id) // self.layer_size

    # -- JOB / REASON names --------------------------------------------------
    # Most of the repository addresses the two layers by name. Each accessor is
    # defined for whichever layers the network actually has: on a single-layer
    # network the absent layer has no columns, zero states, and no offset.

    @property
    def job_cols(self) -> list[str]:
        info = self.layer("JOB")
        return list(info.columns) if info else []

    @property
    def reason_cols(self) -> list[str]:
        info = self.layer("REASON")
        return list(info.columns) if info else []

    @property
    def n_job(self) -> int:
        return len(self.job_cols)

    @property
    def n_reason(self) -> int:
        return len(self.reason_cols)

    @property
    def job_offset(self) -> "int | None":
        info = self.layer("JOB")
        return self.offset(info.position) if info else None

    @property
    def reason_offset(self) -> "int | None":
        info = self.layer("REASON")
        return self.offset(info.position) if info else None

    def job_category_node(self, category_index: int) -> int:
        info = self.layer("JOB")
        if info is None:
            raise ValueError("this network has no JOB layer")
        return self.category_node(info.position, category_index)

    def reason_category_node(self, category_index: int) -> int:
        info = self.layer("REASON")
        if info is None:
            raise ValueError("this network has no REASON layer")
        return self.category_node(info.position, category_index)
