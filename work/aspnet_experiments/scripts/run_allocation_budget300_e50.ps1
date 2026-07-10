param(
    [string]$Python = "C:\Users\x2472\miniconda3\envs\torch\python.exe",
    [string]$Device = "cuda",
    [int]$Epochs = 50,
    [int]$BatchSize = 128,
    [double]$Lr = 0.1,
    [double]$ImbFactor = 100,
    [int]$ProtoBudget = 300,
    [int]$KMin = 1,
    [int]$KMax = 8,
    [double]$EffectiveBeta = 0.9999,
    [string]$Dataset = "cifar100lt",
    [string]$DataRoot = "data",
    [int]$Workers = 4
)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = "logs/allocation_budget300_e50_$stamp.log"

function Test-RunComplete {
    param([string]$RunName)
    $metrics = Join-Path "runs" "$RunName\metrics.csv"
    if (-not (Test-Path $metrics)) {
        return $false
    }
    $rows = Import-Csv $metrics
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

function Run-Allocation {
    param(
        [string]$Allocation,
        [string]$RunName
    )

    Run-Step "Adaptive $Allocation budget=$ProtoBudget" $RunName @(
        "train.py",
        "--dataset", $Dataset,
        "--data-root", $DataRoot,
        "--imb-factor", "$ImbFactor",
        "--model", "adaptive_proto",
        "--allocation", $Allocation,
        "--proto-budget", "$ProtoBudget",
        "--k-min", "$KMin",
        "--k-max", "$KMax",
        "--effective-beta", "$EffectiveBeta",
        "--epochs", "$Epochs",
        "--batch-size", "$BatchSize",
        "--lr", "$Lr",
        "--workers", "$Workers",
        "--device", $Device,
        "--temperature", "0.1",
        "--pool-tau", "0.25",
        "--run-name", $RunName
    )
}

Run-Step "Fixed K=3 t=0.1 tau=0.25 e50" "fixed_k3_t01_tau025_e50" @(
    "train.py",
    "--dataset", $Dataset,
    "--data-root", $DataRoot,
    "--imb-factor", "$ImbFactor",
    "--model", "proto",
    "--proto-mode", "fixed",
    "--fixed-k", "3",
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--lr", "$Lr",
    "--workers", "$Workers",
    "--device", $Device,
    "--temperature", "0.1",
    "--pool-tau", "0.25",
    "--run-name", "fixed_k3_t01_tau025_e50"
)

Run-Allocation "log" "adaptive_log_budget300_e50"
Run-Allocation "sqrt" "adaptive_sqrt_budget300_e50"
Run-Allocation "linear" "adaptive_linear_budget300_e50"
Run-Allocation "effective" "adaptive_effective_budget300_e50"

Write-Host "==== Summary ===="
$names = @(
    "fixed_k3_t01_tau025_e50",
    "adaptive_log_budget300_e50",
    "adaptive_sqrt_budget300_e50",
    "adaptive_linear_budget300_e50",
    "adaptive_effective_budget300_e50"
)

foreach ($n in $names) {
    $p = Join-Path "runs" "$n\metrics.csv"
    if (Test-Path $p) {
        $rows = Import-Csv $p
        $best = $rows | Sort-Object {[double]$_.val_acc} -Descending | Select-Object -First 1
        $protoPath = Join-Path "runs" "$n\prototype_counts.json"
        $protoInfo = ""
        $distInfo = ""
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
        Write-Host "${n}: missing metrics.csv"
    }
}

Write-Host "Log saved to $log"
