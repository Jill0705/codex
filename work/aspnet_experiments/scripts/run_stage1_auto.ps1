param(
    [string]$Python = "python",
    [string]$Device = "cuda",
    [int]$Epochs = 200,
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
$log = "logs/stage1_$stamp.log"

function Run-Step {
    param(
        [string]$Name,
        [string[]]$ArgsList
    )
    Write-Host "==== $Name ===="
    Add-Content -Path $log -Value "==== $Name ===="
    Add-Content -Path $log -Value ("COMMAND: " + $Python + " " + ($ArgsList -join " "))

    $global:LASTEXITCODE = 0
    & $Python @ArgsList 2>&1 | ForEach-Object {
        $line = $_.ToString()
        Write-Host $line
        Add-Content -Path $log -Value $line
    }
    $code = $LASTEXITCODE
    Add-Content -Path $log -Value "EXITCODE: $code"
    if ($code -ne 0) {
        throw "Step failed: $Name with exit code $code. See $log"
    }
}

Write-Host "Project: $Root"
Write-Host "Python: $Python"
Write-Host "Log: $log"

Run-Step "Environment check" @(
    "-c",
    "import sys, torch; print('python', sys.version); print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
)

Run-Step "Smoke test" @(
    "train.py",
    "--dataset", "synthetic",
    "--model", "adaptive_proto",
    "--num-classes", "10",
    "--epochs", "1",
    "--batch-size", "64",
    "--synthetic-size", "512",
    "--device", "cpu",
    "--run-name", "smoke_adaptive_proto"
)

Run-Step "CE baseline" @(
    "train.py",
    "--dataset", $Dataset,
    "--data-root", $DataRoot,
    "--imb-factor", "$ImbFactor",
    "--model", "ce",
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--lr", "$Lr",
    "--workers", "$Workers",
    "--device", $Device,
    "--run-name", "$($Dataset)_if$([int]$ImbFactor)_ce"
)

Run-Step "Single prototype" @(
    "train.py",
    "--dataset", $Dataset,
    "--data-root", $DataRoot,
    "--imb-factor", "$ImbFactor",
    "--model", "proto",
    "--proto-mode", "single",
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--lr", "$Lr",
    "--workers", "$Workers",
    "--device", $Device,
    "--run-name", "$($Dataset)_if$([int]$ImbFactor)_single_proto"
)

Run-Step "Fixed K=4 prototypes" @(
    "train.py",
    "--dataset", $Dataset,
    "--data-root", $DataRoot,
    "--imb-factor", "$ImbFactor",
    "--model", "proto",
    "--proto-mode", "fixed",
    "--fixed-k", "4",
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--lr", "$Lr",
    "--workers", "$Workers",
    "--device", $Device,
    "--run-name", "$($Dataset)_if$([int]$ImbFactor)_fixed_k4"
)

Run-Step "Adaptive K<=4 prototypes" @(
    "train.py",
    "--dataset", $Dataset,
    "--data-root", $DataRoot,
    "--imb-factor", "$ImbFactor",
    "--model", "adaptive_proto",
    "--k-max", "4",
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--lr", "$Lr",
    "--workers", "$Workers",
    "--device", $Device,
    "--run-name", "$($Dataset)_if$([int]$ImbFactor)_adaptive_k4"
)

Write-Host "All stage-1 experiments finished."
Write-Host "Log saved to $log"
