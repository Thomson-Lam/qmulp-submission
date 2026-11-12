$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'audit.ps1')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
