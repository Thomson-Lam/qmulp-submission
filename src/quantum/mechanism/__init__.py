"""The limit, its convergence and the mechanism study, on the lumped operators through the router.

- ``limit``: the Cesàro limit of the measured register (the QPPR score) and its
  spectral constants; ``running_average``: the coefficient recurrence;
  ``convergence``: the convergence point ``tau*`` and its tables.
- ``variants``: the controls (interlayer weight, own-copy restart, rank-one
  restart, flattened monoplex, no partial copies, label-free operator,
  placebo reference sets, single-layer constructions) and a variant's two
  operators through ``quantum.lumping_router``. None is a reported score.
- ``classical``: the stationary classical score on the same operators and the
  classical ``(alpha, eta)`` grid.
- ``scores_stage``: the scoring stage on one dataset's routed and scored primary
  operators (the frozen inputs are ``evaluation.context``), and the
  reduction-residual table.
- ``studies``, ``single_layer_share``: the mechanism tables (Figure 1, Tables 3
  and 4, the single-layer proposition).
"""
