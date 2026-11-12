$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
$lake = Join-Path $env:USERPROFILE '.elan\bin\lake.exe'
if (-not (Test-Path $lake)) { $lake = 'lake' }
& $lake build
if ($LASTEXITCODE -ne 0) { throw 'lake build failed.' }
$lines = & $lake env lean 'QPPR/Manuscript/Audit.lean' 2>&1
$code = $LASTEXITCODE
$lines | Set-Content -Encoding UTF8 'AUDIT_RESULT.txt'
if ($code -ne 0) { throw 'Lean rejected the audit.' }
if (-not ($lines -match 'AXIOM AUDIT PASSED: 19 theorems')) { throw 'Audit confirmation is missing.' }
Write-Host 'Build and audit completed. See AUDIT_RESULT.txt.'
