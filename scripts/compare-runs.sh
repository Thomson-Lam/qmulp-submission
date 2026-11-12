#!/usr/bin/env bash
# Compare scientific outputs against a separately executed canonical checkout.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
"${PYTHON:-$ROOT/.venv/bin/python}" - "${1:?pass canonical outputs directory}" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np
import polars as pl

reference = Path(sys.argv[1]).resolve()
ported = Path('results/outputs')
rows = []
for name in ('NET_JOB', 'NET_REASON', 'NET_2L'):
    relative = Path('notebook_pipeline/03_classical') / f'classical_scores_{name}.parquet'
    a, b = pl.read_parquet(ported / relative), pl.read_parquet(reference / relative)
    assert a.equals(b), str(relative)
    rows.append({'artifact': str(relative), 'rows': a.height, 'exact_match': True,
                 'maximum_score_difference': float(np.max(np.abs(a['omega_C_test'].to_numpy() - b['omega_C_test'].to_numpy())))})
for name in ('EC_hardware_scores', 'T6_hardware', 'T6_hardware_header'):
    relative = Path('results/hmeq') / f'{name}.csv'
    a, b = pl.read_csv(ported / relative), pl.read_csv(reference / relative)
    assert a.equals(b), str(relative)
    rows.append({'artifact': str(relative), 'rows': a.height, 'exact_match': True})
print(json.dumps({'comparison': 'canonical versus port, identical environment and smoke settings',
                  'artifacts': rows}, indent=2))
PY
