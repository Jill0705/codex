param(
    [string]$Python = "C:\Users\x2472\miniconda3\envs\torch\python.exe",
    [string]$Device = "cuda",
    [int]$Epochs = 50,
    [int]$BatchSize = 128,
    [double]$Lr = 0.1,
    [double[]]$ImbFactors = @(50, 10),
    [int]$ProtoBudget = 300,
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
$log = "logs/allocation_if_sweep_e50_$stamp.log"

function Format-ImbFactor {
    param([double]$Value)
    return ("{0:g}" -f $Value)
}

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

function Run-Ce {
    param(
        [double]$ImbFactor,
        [string]$RunName
    )

    Run-Step "CE IF=$(Format-ImbFactor $ImbFactor) e$Epochs" $RunName @(
        "train.py",
        "--dataset", $Dataset,
        "--data-root", $DataRoot,
        "--imb-factor", "$(Format-ImbFactor $ImbFactor)",
        "--model", "ce",
        "--epochs", "$Epochs",
        "--batch-size", "$BatchSize",
        "--lr", "$Lr",
        "--workers", "$Workers",
        "--device", $Device,
        "--run-name", $RunName
    )
}

function Run-FixedK3 {
    param(
        [double]$ImbFactor,
        [string]$RunName
    )

    Run-Step "Fixed K=3 IF=$(Format-ImbFactor $ImbFactor) t=0.1 tau=0.25 e$Epochs" $RunName @(
        "train.py",
        "--dataset", $Dataset,
        "--data-root", $DataRoot,
        "--imb-factor", "$(Format-ImbFactor $ImbFactor)",
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
        "--run-name", $RunName
    )
}

function Run-Allocation {
    param(
        [double]$ImbFactor,
        [string]$Allocation,
        [string]$RunName
    )

    Run-Step "Adaptive $Allocation budget=$ProtoBudget IF=$(Format-ImbFactor $ImbFactor) e$Epochs" $RunName @(
        "train.py",
        "--dataset", $Dataset,
        "--data-root", $DataRoot,
        "--imb-factor", "$(Format-ImbFactor $ImbFactor)",
        "--model", "adaptive_proto",
        "--allocation", $Allocation,
        "--proto-budget", "$ProtoBudget",
        "--k-min", "$KMin",
        "--k-max", "$KMax",
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

$summaryNames = @()

foreach ($imb in $ImbFactors) {
    $ifTag = Format-ImbFactor $imb
    $ceRun = "ce_if${ifTag}_e$Epochs"
    $fixedRun = "fixed_k3_if${ifTag}_e$Epochs"
    $linearRun = "adaptive_linear_budget${ProtoBudget}_if${ifTag}_e$Epochs"
    $sqrtRun = "adaptive_sqrt_budget${ProtoBudget}_if${ifTag}_e$Epochs"

    Run-Ce $imb $ceRun
    Run-FixedK3 $imb $fixedRun
    Run-Allocation $imb "linear" $linearRun
    Run-Allocation $imb "sqrt" $sqrtRun

    $summaryNames += $ceRun
    $summaryNames += $fixedRun
    $summaryNames += $linearRun
    $summaryNames += $sqrtRun
}

Write-Host "==== Summary ===="
Add-Content -Path $log -Value "==== Summary ===="

foreach ($n in $summaryNames) {
    $p = Join-Path "runs" "$n\metrics.csv"
    if (Test-Path $p) {
        $rows = Import-Csv $p
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
