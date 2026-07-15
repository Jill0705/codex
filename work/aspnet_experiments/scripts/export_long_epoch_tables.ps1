param(
    [string]$Python = "python",
    [string]$LogsDir = "logs",
    [string]$TablesDir = "tables"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
& $Python "scripts/export_long_epoch_tables.py" --logs-dir $LogsDir --tables-dir $TablesDir
if ($LASTEXITCODE -ne 0) {
    throw "export_long_epoch_tables.py failed"
}
