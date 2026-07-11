param(
    [string]$Python = "C:\Users\x2472\miniconda3\envs\torch\python.exe",
    [string]$Device = "cuda",
    [int]$Epochs = 50,
    [int]$BatchSize = 128,
    [double]$Lr = 0.1,
    [double[]]$ImbFactors = @(100, 50),
    [int]$DrwEpoch = 30,
    [double]$Scale = 1,
    [double]$MaxM = 0.5,
    [double]$ReweightBeta = 0.9999,
    [string]$Dataset = "cifar100lt",
    [string]$DataRoot = "data",
    [int]$Workers = 4
)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = "logs/ldam_fixed_e50_$stamp.log"

function Format-NumberTag {
    param([double]$Value)
    return ("{0:g}" -f $Value).Replace(".", "p")
}

function Format-NumberArg {
    param([double]$Value)
    return ("{0:g}" -f $Value)
}

function Test-RunComplete {
    param(
        [string]$RunName,
        [int]$TargetEpochs
    )

    $metrics = Join-Path "runs" "$RunName\metrics.csv"
    if (-not (Test-Path $metrics)) {
        return $false
    }
    $rows = @(Import-Csv $metrics)
    if ($rows.Count -eq 0) {
        return $false
    }
    $last = $rows | Select-Object -Last 1
    return ([int]$last.epoch -ge $TargetEpochs)
}

function Run-Step {
    param(
        [string]$Name,
        [string]$RunName,
        [string[]]$ArgsList
    )

    if (Test-RunComplete $RunName $Epochs) {
        $msg = "==== SKIP $Name ($RunName already has >= $Epochs epochs) ===="
        Write-Host $msg
        Add-Content -Path $log -Value $msg
        return
    }

    Write-Host "==== $Name ===="
    Add-Content -Path $log -Value "==== $Name ===="
    Add-Content -Path $log -Value ("COMMAND: " + $Python + " " + ($ArgsList -join " "))

    & $Python @ArgsList 2>&1 | ForEach-Object {
        $line = $_.ToString()
        Write-Host $line
        Add-Content -Path $log -Value $line
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name with exit code $LASTEXITCODE. See $log"
    }
}

$summaryNames = @()
$scaleArg = Format-NumberArg $Scale
$scaleTag = Format-NumberTag $Scale

foreach ($imb in $ImbFactors) {
    $ifArg = Format-NumberArg $imb
    $runName = "ldam_drw_if${ifArg}_e${Epochs}_s${scaleTag}"
    $summaryNames += $runName

    Run-Step "LDAM-DRW IF=$ifArg scale=$scaleArg e$Epochs" $runName @(
        "train.py", "--dataset", $Dataset, "--data-root", $DataRoot,
        "--imb-factor", "$ifArg", "--model", "ce", "--loss", "ldam",
        "--ldam-scale", "$scaleArg", "--ldam-max-m", "$MaxM",
        "--ldam-reweight-beta", "$ReweightBeta", "--drw-epoch", "$DrwEpoch",
        "--epochs", "$Epochs", "--batch-size", "$BatchSize", "--lr", "$Lr",
        "--workers", "$Workers", "--device", $Device, "--run-name", $runName
    )
}

Write-Host "==== Summary ===="
Add-Content -Path $log -Value "==== Summary ===="

foreach ($n in $summaryNames) {
    $p = Join-Path "runs" "$n\metrics.csv"
    if (Test-Path $p) {
        $rows = @(Import-Csv $p)
        $best = $rows | Sort-Object {[double]$_.val_acc} -Descending | Select-Object -First 1
        $msg = "{0}: best_epoch={1}, val={2}, many={3}, medium={4}, few={5}" -f $n, $best.epoch, $best.val_acc, $best.many_acc, $best.medium_acc, $best.few_acc
        Write-Host $msg
        Add-Content -Path $log -Value $msg
    } else {
        $msg = "${n}: missing metrics.csv"
        Write-Host $msg
        Add-Content -Path $log -Value $msg
    }
}

Write-Host "Log saved to $log"
