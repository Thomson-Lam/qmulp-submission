@echo off
setlocal
cd /d "%~dp0.."
if exist "%USERPROFILE%\.elan\bin\lake.exe" (
  set "LAKE=%USERPROFILE%\.elan\bin\lake.exe"
) else (
  set "LAKE=lake"
)
"%LAKE%" build
if errorlevel 1 exit /b 1
"%LAKE%" env lean QPPR\Manuscript\Audit.lean > AUDIT_RESULT.txt 2>&1
if errorlevel 1 (
  type AUDIT_RESULT.txt
  echo ERROR: Lean rejected the audit.
  exit /b 1
)
findstr /C:"AXIOM AUDIT PASSED: 19 theorems" AUDIT_RESULT.txt >nul
if errorlevel 1 (
  type AUDIT_RESULT.txt
  echo ERROR: Audit confirmation was not found.
  exit /b 2
)
echo Build and audit completed.
echo See AUDIT_RESULT.txt and docs\PROPOSITION_STATUS.md.
exit /b 0
