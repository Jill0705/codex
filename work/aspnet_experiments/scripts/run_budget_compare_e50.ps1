param(
    [string]$Python = "C:\Users\x2472\miniconda3\envs\torch\python.exe",
    [string]$Device = "cuda",
    [int]$Epochs = 50,
    [int]$BatchSize = 128,
    [double]$Lr = 0.1,
    [double]$ImbFactor = 100,
    [string]$Dataset = "cifar100lt",
    [string]$DataRoot = "data",
    [int]$Workers = 2
)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = "logs/budget_compare_e50_$stamp.log"

function Run-Step {
    param(
        [string]$Name,
        [string[]]$ArgsList
    )

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

Run-Step "Fixed K=2 t=0.1 tau=0.25 e50" @(
    "train.py",
    "--dataset", $Dataset,
    "--data-root", $DataRoot,
    "--imb-factor", "$ImbFactor",
    "--model", "proto",
    "--proto-mode", "fixed",
    "--fixed-k", "2",
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--lr", "$Lr",
    "--workers", "$Workers",
    "--device", $Device,
    "--temperature", "0.1",
    "--pool-tau", "0.25",
    "--run-name", "fixed_k2_t01_tau025_e50"
)

Run-Step "Fixed K=3 t=0.1 tau=0.25 e50" @(
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

Run-Step "Adaptive K<=6 t=0.1 tau=0.25 e50" @(
    "train.py",
    "--dataset", $Dataset,
    "--data-root", $DataRoot,
    "--imb-factor", "$ImbFactor",
    "--model", "adaptive_proto",
    "--k-max", "6",
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--lr", "$Lr",
    "--workers", "$Workers",
    "--device", $Device,
    "--temperature", "0.1",
    "--pool-tau", "0.25",
    "--run-name", "adaptive_k6_t01_tau025_e50"
)

Run-Step "Adaptive K<=8 t=0.1 tau=0.25 e50" @(
    "train.py",
    "--dataset", $Dataset,
    "--data-root", $DataRoot,
    "--imb-factor", "$ImbFactor",
    "--model", "adaptive_proto",
    "--k-max", "8",
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--lr", "$Lr",
    "--workers", "$Workers",
    "--device", $Device,
    "--temperature", "0.1",
    "--pool-tau", "0.25",
    "--run-name", "adaptive_k8_t01_tau025_e50"
)

Write-Host "==== Summary ===="
$names = @(
    "ce_e50",
    "fixed_k2_t01_tau025_e50",
    "fixed_k3_t01_tau025_e50",
    "fixed_k4_t01_tau025_e50",
    "adaptive_k4_t01_tau025_e50",
    "adaptive_k6_t01_tau025_e50",
    "adaptive_k8_t01_tau025_e50"
)

foreach ($n in $names) {
    $p = Join-Path "runs" "$n\metrics.csv"
    if (Test-Path $p) {
        $rows = Import-Csv $p
        $best = $rows | Sort-Object {[double]$_.val_acc} -Descending | Select-Object -First 1
        $protoPath = Join-Path "runs" "$n\prototype_counts.json"
        $protoInfo = ""
        if (Test-Path $protoPath) {
            $protoJson = Get-Content $protoPath -Raw | ConvertFrom-Json
            $protoInfo = ", prototypes=$($protoJson.total_prototypes)"
        }
        $msg = "{0}: best_epoch={1}, val={2}, many={3}, medium={4}, few={5}{6}" -f $n, $best.epoch, $best.val_acc, $best.many_acc, $best.medium_acc, $best.few_acc, $protoInfo
        Write-Host $msg
        Add-Content -Path $log -Value $msg
    } else {
        Write-Host "${n}: missing metrics.csv"
    }
}

Write-Host "Log saved to $log"
