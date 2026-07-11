param(
    [string]$Python = "C:\Users\x2472\miniconda3\envs\torch\python.exe",
    [string]$Device = "cuda",
    [int]$Epochs = 50,
    [int]$BatchSize = 128,
    [double]$Lr = 0.1,
    [int]$DrwEpoch = 30,
    [double]$LdamScale = 1,
    [double]$LdamMaxM = 0.5,
    [double]$LdamReweightBeta = 0.9999,
    [int]$KMin = 1,
    [int]$KMax = 8,
    [string]$Dataset = "cifar100lt",
    [string]$DataRoot = "data",
    [int]$Workers = 4
)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = "logs/sqrt_loss_compat_e50_$stamp.log"

function Test-RunComplete {
    param([string]$RunName)

    $metrics = Join-Path "runs" "$RunName\metrics.csv"
    if (-not (Test-Path $metrics)) {
        return $false
    }
    $rows = @(Import-Csv $metrics)
    if ($rows.Count -eq 0) {
        return $false
    }
    $last = $rows | Select-Object -Last 1
    return ([int]$last.epoch -ge $Epochs)
}

function Run-Step {
    param(
        [string]$Name,
        [string]$RunName,
        [string[]]$ArgsList
    )

    if (Test-RunComplete $RunName) {
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

function Run-SqrtBalancedSoftmax {
    param(
        [string]$IfTag,
        [int]$Budget
    )

    $runName = "adaptive_sqrt_budget${Budget}_if${IfTag}_e${Epochs}_bs"
    Run-Step "Sqrt budget=$Budget + Balanced Softmax IF=$IfTag e$Epochs" $runName @(
        "train.py", "--dataset", $Dataset, "--data-root", $DataRoot,
        "--imb-factor", $IfTag, "--model", "adaptive_proto",
        "--allocation", "sqrt", "--proto-budget", "$Budget",
        "--k-min", "$KMin", "--k-max", "$KMax",
        "--loss", "balanced_softmax", "--epochs", "$Epochs",
        "--batch-size", "$BatchSize", "--lr", "$Lr", "--workers", "$Workers",
        "--device", $Device, "--temperature", "0.1", "--pool-tau", "0.25",
        "--run-name", $runName
    )
}

function Run-SqrtLdam {
    param(
        [string]$IfTag,
        [int]$Budget
    )

    $runName = "adaptive_sqrt_budget${Budget}_if${IfTag}_e${Epochs}_ldam_s1"
    Run-Step "Sqrt budget=$Budget + LDAM-DRW IF=$IfTag e$Epochs" $runName @(
        "train.py", "--dataset", $Dataset, "--data-root", $DataRoot,
        "--imb-factor", $IfTag, "--model", "adaptive_proto",
        "--allocation", "sqrt", "--proto-budget", "$Budget",
        "--k-min", "$KMin", "--k-max", "$KMax",
        "--loss", "ldam", "--ldam-scale", "$LdamScale", "--ldam-max-m", "$LdamMaxM",
        "--ldam-reweight-beta", "$LdamReweightBeta", "--drw-epoch", "$DrwEpoch",
        "--epochs", "$Epochs", "--batch-size", "$BatchSize", "--lr", "$Lr",
        "--workers", "$Workers", "--device", $Device, "--temperature", "0.1",
        "--pool-tau", "0.25", "--run-name", $runName
    )
}

$summaryNames = @(
    "adaptive_sqrt_budget200_if100_e50_bs",
    "adaptive_sqrt_budget300_if50_e50_bs",
    "adaptive_sqrt_budget200_if100_e50_ldam_s1",
    "adaptive_sqrt_budget300_if50_e50_ldam_s1"
)

Run-SqrtBalancedSoftmax "100" 200
Run-SqrtBalancedSoftmax "50" 300
Run-SqrtLdam "100" 200
Run-SqrtLdam "50" 300

Write-Host "==== Summary ===="
Add-Content -Path $log -Value "==== Summary ===="

foreach ($n in $summaryNames) {
    $p = Join-Path "runs" "$n\metrics.csv"
    if (Test-Path $p) {
        $rows = @(Import-Csv $p)
        $best = $rows | Sort-Object {[double]$_.val_acc} -Descending | Select-Object -First 1
        $protoPath = Join-Path "runs" "$n\prototype_counts.json"
        $protoInfo = ", prototypes=N/A"
        $distInfo = ", k_dist=N/A"
        if (Test-Path $protoPath) {
            $protoJson = Get-Content $protoPath -Raw | ConvertFrom-Json
            $protoInfo = ", prototypes=$($protoJson.total_prototypes)"
            $dist = @($protoJson.proto_counts) | Group-Object | Sort-Object {[int]$_.Name}
            $distInfo = ", k_dist=" + (($dist | ForEach-Object { "K=$($_.Name):$($_.Count)" }) -join " ")
        }
        $msg = "{0}: best_epoch={1}, val={2}, many={3}, medium={4}, few={5}{6}{7}" -f $n, $best.epoch, $best.val_acc, $best.many_acc, $best.medium_acc, $best.few_acc, $protoInfo, $distInfo
        Write-Host $msg
        Add-Content -Path $log -Value $msg
    } else {
        $msg = "${n}: missing metrics.csv"
        Write-Host $msg
        Add-Content -Path $log -Value $msg
    }
}

Write-Host "Log saved to $log"
