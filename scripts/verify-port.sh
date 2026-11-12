#!/usr/bin/env bash
# Verify the original checkout stayed untouched and all mapped Python files match the recorded port.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
"${PYTHON:-$ROOT/.venv/bin/python}" - "${1:-../qmulp}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

canonical = Path(sys.argv[1])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
expected = json.loads(Path('docs/canonical-sha256.json').read_text())
current = {
    str(p.relative_to(canonical)): sha(p)
    for p in canonical.rglob('*')
    if p.is_file() and not any(x in p.relative_to(canonical).parts
                              for x in ('.git', '.venv', '__pycache__', '.pytest_cache'))
}
assert current == expected, 'canonical files changed, appeared or disappeared'
rows = json.loads(Path('docs/port-manifest.json').read_text())
for row in rows:
    assert sha(canonical / row['source']) == row['source_sha256'], row['source']
    assert sha(Path(row['destination'])) == row['destination_sha256'], row['destination']
assert len({r['source'] for r in rows}) == len(rows)
assert len({r['destination'] for r in rows}) == len(rows)
assert not any(p.is_symlink() for base in ('src', 'scripts', 'data', 'results', 'docs', 'tests', 'notebooks')
               for p in Path(base).rglob('*')), 'repository symlink found'
print(f'PASS: {len(current)} canonical files unchanged; {len(rows)} one-to-one Python copies verified; no repository symlinks')
print(f"Byte-identical Python copies: {sum(r['source_sha256'] == r['destination_sha256'] for r in rows)}")
PY
