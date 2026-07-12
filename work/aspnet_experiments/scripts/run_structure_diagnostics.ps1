param(
    [string]$Python = "C:\Users\x2472\miniconda3\envs\torch\python.exe",
    [string]$Device = "cuda",
    [int]$BatchSize = 256,
    [int]$Workers = 4,
    [int]$MaxClusters = 8,
    [int]$BootstrapIters = 5,
    [switch]$Force
)

$ErrorActionPreference = "Continue"
$env:LOKY_MAX_CPU_COUNT = "1"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = "logs/structure_diagnostics_$stamp.log"

function Run-Analysis {
    param(
        [string]$Name,
        [string]$RunName,
        [string]$Prefix
    )

    $diag = Join-Path "tables" "structure_diagnostics_$Prefix.csv"
    $group = Join-Path "tables" "structure_group_summary_$Prefix.csv"
    $corr = Join-Path "tables" "structure_correlations_$Prefix.csv"
    $stability = Join-Path "tables" "structure_cluster_stability_$Prefix.csv"
    $variance = Join-Path "figures" "structure_${Prefix}_intra_variance.png"
    $estimatedK = Join-Path "figures" "structure_${Prefix}_estimated_k.png"
    $silhouette = Join-Path "figures" "structure_${Prefix}_best_silhouette.png"
    $stabilityFig = Join-Path "figures" "structure_${Prefix}_cluster_stability.png"
    $pca = Join-Path "figures" "structure_${Prefix}_pca_classes.png"
    $tsne = Join-Path "figures" "structure_${Prefix}_tsne_representative_classes.png"
    if ((-not $Force) -and (Test-Path $diag) -and (Test-Path $group) -and (Test-Path $corr) -and (Test-Path $stability) -and (Test-Path $variance) -and (Test-Path $estimatedK) -and (Test-Path $silhouette) -and (Test-Path $stabilityFig) -and (Test-Path $pca) -and (Test-Path $tsne)) {
        $msg = "==== SKIP $Name ($Prefix diagnostics already exist) ===="
        Write-Host $msg
        Add-Content -Path $log -Value $msg
        return
    }

    $argsList = @(
        "analyze_structure.py",
        "--run-name", $RunName,
        "--output-prefix", $Prefix,
        "--device", $Device,
        "--batch-size", "$BatchSize",
        "--workers", "$Workers",
        "--max-clusters", "$MaxClusters",
        "--bootstrap-iters", "$BootstrapIters"
    )

    Write-Host "==== $Name ===="
    Add-Content -Path $log -Value "==== $Name ===="
    Add-Content -Path $log -Value ("COMMAND: " + $Python + " " + ($argsList -join " "))

    & $Python @argsList 2>&1 | ForEach-Object {
        $line = $_.ToString()
        Write-Host $line
        Add-Content -Path $log -Value $line
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Structure diagnostics failed for $Name with exit code $LASTEXITCODE. See $log"
    }
}

Run-Analysis "CE feature structure IF=100" "ce_e50" "if100"
Run-Analysis "CE feature structure IF=50" "ce_if50_e50" "if50"

Write-Host "Log saved to $log"
