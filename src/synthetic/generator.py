"""A synthetic individual--attribute population in the frame schema the network builders read.

:func:`generate_attribute_graph` draws ``n`` individuals over ``L`` layers
with ``levels_per_layer`` categories each. Category sizes are
Dirichlet-multinomial (``size_dispersion = 0`` is equal sizes); every category
has a reference share on an equally spaced grid centred on
``ref_share_mean`` with half-width ``ref_share_spread``, assigned
independently of its size. An individual's risk is the mean of its observed
categories' shares. Test labels are Bernoulli draws of the risk, multiplied by
``partial_kappa`` for partially observed test individuals. A ``partial_share``
of the test individuals loses its category in one layer (test only, as new
sellers are); a ``dangling_share`` of individuals has no category anywhere, a
``dangling_ref_share`` fraction of them development references and the rest
test individuals. The base population is drawn first and the dangling
individuals are appended from a derived seed, so a sweep over
``dangling_share`` changes nothing else.

Where the development references are placed (``reference_placement``):

``"joint_cell"`` (the default, used by S1, S2 and S3)
    References are placed per joint cell. A cell is the set of development
    individuals with the same category in every layer, and it receives
    ``round(risk * cell size)`` references, ``risk`` being the mean of the
    shares of the cell's categories. With one layer a cell is a category and
    its realised share is its grid value up to rounding. With two layers the
    realised share of a category is the average of the risks of its cells,
    which pulls it towards the middle of the grid, and a cell of one or two
    individuals rounds to no reference at all.
``"category"`` (the S3b sensitivity run)
    The reference share of every category of every layer is realised: the
    cell counts are fitted to the category totals of both layers
    (:func:`_references_per_category`), whatever the joint cells hold, so a
    category's realised share is its grid value up to rounding and to a
    common rescaling of the two layers' totals (a few per cent). This
    placement draws from a seed derived for it, so for one seed the
    categories, the partition, the test labels and the dangling individuals
    are identical under both settings and only the development labels differ.

``truth["reference_share_gap"]`` is the largest gap between a category's
realised reference share among the development individuals and its grid
share, whichever placement was used.

The frame carries one-hot columns ``L<l>_c<k>`` plus ``L<l>_Missing`` per
layer, ``BAD`` and ``partition``; :meth:`SyntheticInstance.partitions` turns it
into the :class:`evaluation.masked_eval.Partitions` every stage reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from config import LayerSpec, NetworkSpec, derive_seed
from evaluation.masked_eval import DEVELOPMENT_PARTITION, TEST_PARTITION, Partitions
from preprocessing.missing_levels import MISSING_SENTINEL


@dataclass
class SyntheticInstance:
    """One generated population with its network specification and the truth behind it."""

    frame: pl.DataFrame
    partition: np.ndarray
    spec: NetworkSpec
    truth: dict[str, Any]

    @property
    def n(self) -> int:
        return int(self.frame.height)

    def partitions(self) -> Partitions:
        return Partitions(frame=self.frame, borrower_ids=np.arange(self.n), partition=self.partition.astype(str),
                          target_col="BAD", processed_path=None, split_path=None)

    @property
    def test_idx(self) -> np.ndarray:
        return np.flatnonzero(self.partition == TEST_PARTITION)

    @property
    def partially_observed(self) -> np.ndarray:
        return np.asarray(self.truth["partially_observed"], dtype=bool)

    @property
    def dangling(self) -> np.ndarray:
        return np.asarray(self.truth["dangling"], dtype=bool)


REFERENCE_PLACEMENTS = ("joint_cell", "category")
IPF_SWEEPS = 200


def _layer_name(layer: int) -> str:
    return f"L{layer + 1}"


def _category_sizes(rng, n: int, levels: int, dispersion: float) -> np.ndarray:
    if dispersion <= 0.0:
        return np.full(levels, 1.0 / levels)
    return rng.dirichlet(np.full(levels, 1.0 / float(dispersion)))


def _share_grid(rng, levels: int, mean: float, spread: float) -> np.ndarray:
    grid = mean + spread * (np.linspace(-1.0, 1.0, levels) if levels > 1 else np.zeros(1))
    return np.clip(rng.permutation(grid), 0.01, 0.99)


def _ratio(target: np.ndarray, margin: np.ndarray) -> np.ndarray:
    """``target / margin`` with zero where the margin is zero."""
    return np.divide(target, margin, out=np.zeros_like(target, dtype=float), where=margin > 0)


def _references_per_category(rng, categories: np.ndarray, shares: list, dev_ids: np.ndarray) -> np.ndarray:
    """Ids of the development references when every category's reference share is realised (``reference_placement="category"``).

    Category ``c`` of layer ``l`` is to hold ``shares[l][c] * size`` references among the development individuals. With one
    layer the cells are the categories and those are the cell counts. With two layers the cell counts are fitted to the
    targets of both layers by iterative proportional fitting, starting from the per-cell expectation, after the two layers'
    total targets are set to their mean (they differ slightly because the shares are not weighted by the sizes alike). The
    fitted counts are rounded by systematic sampling along the cells, so each layer-0 category ends within about one individual
    of its target and each layer-1 category within about the number of layer-0 categories; a cell's references are a random
    subset of its members.
    """
    cats = categories[dev_ids]
    dims = tuple(len(grid) for grid in shares)
    key = np.ravel_multi_index(tuple(cats.T), dims)
    sizes = np.bincount(key, minlength=int(np.prod(dims))).astype(float).reshape(dims)
    if len(dims) == 1:
        expected = np.asarray(shares[0], dtype=float) * sizes
    else:
        first, second = np.asarray(shares[0], dtype=float), np.asarray(shares[1], dtype=float)
        rows, cols = first * sizes.sum(axis=1), second * sizes.sum(axis=0)
        total = 0.5 * (rows.sum() + cols.sum())
        rows, cols = rows * (total / rows.sum()), cols * (total / cols.sum())
        expected = sizes * (first[:, None] + second[None, :]) / 2.0
        for _ in range(IPF_SWEEPS):
            expected = expected * _ratio(rows, expected.sum(axis=1))[:, None]
            expected = np.minimum(expected * _ratio(cols, expected.sum(axis=0))[None, :], sizes)
    cumulative = np.cumsum(expected.ravel())
    start = rng.random()
    counts = np.floor(cumulative + start) - np.floor(np.concatenate(([0.0], cumulative[:-1])) + start)
    counts = np.minimum(counts, sizes.ravel()).astype(int)
    rank = np.empty(dev_ids.size, dtype=int)
    seen = np.zeros(counts.size, dtype=int)
    for position in rng.permutation(dev_ids.size):
        cell = key[position]
        rank[position] = seen[cell]
        seen[cell] += 1
    return dev_ids[rank < counts[key]]


def _share_gap(labels: np.ndarray, categories: np.ndarray, shares: list, dev_ids: np.ndarray) -> float:
    """Largest gap between a category's realised reference share among the development individuals and its grid share."""
    gap = 0.0
    for layer, grid in enumerate(shares):
        members = categories[dev_ids, layer]
        for category, share in enumerate(grid):
            chosen = members == category
            if chosen.any():
                gap = max(gap, abs(float(labels[dev_ids[chosen]].mean()) - float(share)))
    return gap


def generate_attribute_graph(
    n: int, L: int, levels_per_layer, size_dispersion: float, ref_share_mean: float, ref_share_spread: float,
    test_frac: float, partial_share: float, partial_kappa: float, dangling_share: float, dangling_ref_share: float,
    seed: int, reference_placement: str = "joint_cell",
) -> SyntheticInstance:
    """See the module docstring. Every argument is recorded in ``truth``."""
    if reference_placement not in REFERENCE_PLACEMENTS:
        raise ValueError(f"reference_placement must be one of {REFERENCE_PLACEMENTS}; got {reference_placement!r}")
    rng = np.random.default_rng(int(seed))
    levels = [int(x) for x in levels_per_layer]
    if len(levels) != int(L) or L not in (1, 2):
        raise ValueError("levels_per_layer must give one level count per layer, for one or two layers")
    n_base = int(n)
    categories = np.zeros((n_base, L), dtype=int)
    shares = []
    for layer in range(L):
        sizes = _category_sizes(rng, n_base, levels[layer], float(size_dispersion))
        categories[:, layer] = rng.choice(levels[layer], size=n_base, p=sizes)
        shares.append(_share_grid(rng, levels[layer], float(ref_share_mean), float(ref_share_spread)))
    is_test = rng.random(n_base) < float(test_frac)
    observed = np.ones((n_base, L), dtype=bool)
    if L > 1 and partial_share > 0.0:
        test_ids = np.flatnonzero(is_test)
        lose = rng.choice(test_ids, size=int(round(partial_share * test_ids.size)), replace=False)
        observed[lose, rng.integers(0, L, size=lose.size)] = False
    risk = np.array([np.mean([shares[l][categories[i, l]] for l in range(L) if observed[i, l]]) for i in range(n_base)])
    labels = np.zeros(n_base, dtype=int)
    dev_ids = np.flatnonzero(~is_test)
    if reference_placement == "joint_cell":
        # Development references per joint cell: round(risk * cell size) of each cell's individuals (see the module docstring).
        cells: dict[tuple, list[int]] = {}
        for i in dev_ids:
            cells.setdefault(tuple(categories[i]), []).append(int(i))
        for members in cells.values():
            members = np.array(members)
            n_ref = int(round(float(risk[members].mean()) * members.size))
            labels[members[:n_ref]] = 1
    else:
        # Development references per category: from a derived seed, so nothing else in the population changes.
        labels[_references_per_category(np.random.default_rng(derive_seed(int(seed), "reference_placement")), categories, shares, dev_ids)] = 1
    share_gap = _share_gap(labels, categories, shares, dev_ids)
    partially = ~observed.all(axis=1)
    kappa = np.where(partially, float(partial_kappa), 1.0)
    test_ids = np.flatnonzero(is_test)
    labels[test_ids] = (rng.random(test_ids.size) < np.clip(risk[test_ids] * kappa[test_ids], 0.0, 1.0)).astype(int)

    # Dangling individuals, appended from a derived seed.
    n_dangling = int(round(float(dangling_share) * n_base))
    d_rng = np.random.default_rng(derive_seed(int(seed), "dangling"))
    d_ref = d_rng.random(n_dangling) < float(dangling_ref_share)
    n_total = n_base + n_dangling
    columns: dict[str, Any] = {}
    for layer in range(L):
        name = _layer_name(layer)
        for k in range(levels[layer]):
            active = np.zeros(n_total, dtype=int)
            active[:n_base] = ((categories[:, layer] == k) & observed[:, layer]).astype(int)
            columns[f"{name}_c{k}"] = active
        missing = np.ones(n_total, dtype=int)
        missing[:n_base] = (~observed[:, layer]).astype(int)
        columns[f"{name}_{MISSING_SENTINEL}"] = missing
    bad = np.concatenate([labels, d_ref.astype(int)])
    partition = np.concatenate([np.where(is_test, TEST_PARTITION, DEVELOPMENT_PARTITION),
                                np.where(d_ref, DEVELOPMENT_PARTITION, TEST_PARTITION)]).astype(object)
    columns["BAD"] = bad
    frame = pl.DataFrame(columns)
    spec = NetworkSpec(name=f"SYNTH_{L}L", layers=tuple(
        LayerSpec(_layer_name(l), tuple([f"{_layer_name(l)}_c{k}" for k in range(levels[l])] + [f"{_layer_name(l)}_{MISSING_SENTINEL}"]))
        for l in range(L)))
    truth = {
        "n": n_base, "n_total": n_total, "L": L, "levels_per_layer": levels, "size_dispersion": float(size_dispersion),
        "ref_share_mean": float(ref_share_mean), "ref_share_spread": float(ref_share_spread), "test_frac": float(test_frac),
        "partial_share": float(partial_share), "partial_kappa": float(partial_kappa), "dangling_share": float(dangling_share),
        "dangling_ref_share": float(dangling_ref_share), "seed": int(seed), "reference_placement": reference_placement,
        "reference_share_gap": share_gap,
        "shares": [s.tolist() for s in shares], "categories": categories, "observed": observed, "risk": risk,
        "partially_observed": np.concatenate([partially, np.zeros(n_dangling, dtype=bool)]),
        "dangling": np.concatenate([np.zeros(n_base, dtype=bool), np.ones(n_dangling, dtype=bool)]),
        "n_dangling": n_dangling, "n_dangling_references": int(d_ref.sum()),
    }
    return SyntheticInstance(frame=frame, partition=partition, spec=spec, truth=truth)
