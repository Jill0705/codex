param(
    [string]$Python = "python",
    [string]$ImageRoot,
    [string]$TrainList,
    [string]$ValList,
    [string]$Device = "cuda",
    [string]$Backbone = "resnet18",
    [switch]$Pretrained,
    [int]$Epochs = 100,
    [int]$DrwEpoch = 60,
    [int]$BatchSize = 128,
    [int]$Workers = 8,
    [double]$Lr = 0.1,
    [int]$ImageSize = 224,
    [int]$FeatureDim = 512,
    [int]$FixedK = 4,
    [int]$ProtoBudget = 2000,
    [int]$KMin = 1,
    [int]$KMax = 8,
    [int]$NumClasses = 0
)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = "logs/imagenetlt_main_e${Epochs}_$stamp.log"

if (-not $ImageRoot -or -not $TrainList -or -not $ValList) {
    throw "ImageRoot, TrainList, and ValList are required."
}

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

function Get-CommonArgs {
    $argsList = @(
        "train.py", "--dataset", "imagenetlt",
        "--image-root", $ImageRoot,
        "--train-list", $TrainList,
        "--val-list", $ValList,
        "--image-size", "$ImageSize",
        "--backbone", $Backbone,
        "--feature-dim", "$FeatureDim",
        "--epochs", "$Epochs",
        "--batch-size", "$BatchSize",
        "--lr", "$Lr",
        "--workers", "$Workers",
        "--device", $Device
    )
    if ($Pretrained) {
        $argsList += "--pretrained"
    }
    if ($NumClasses -gt 0) {
        $argsList += @("--num-classes", "$NumClasses")
    }
    return $argsList
}

function Invoke-Experiment {
    param(
        [string]$Name,
        [string]$RunName,
        [string[]]$ExtraArgs
    )

    if (Test-RunComplete $RunName) {
        $msg = "==== SKIP $Name ($RunName already has >= $Epochs epochs) ===="
        Write-Host $msg
        Add-Content -Path $log -Value $msg
        return
    }

    $argsList = (Get-CommonArgs) + $ExtraArgs + @("--run-name", $RunName)
    Write-Host "==== $Name ===="
    Add-Content -Path $log -Value "==== $Name ===="
    Add-Content -Path $log -Value ("COMMAND: " + $Python + " " + ($argsList -join " "))
    & $Python @argsList 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) {
        throw "FAILED: $Name. See $log"
    }
}

Invoke-Experiment "ImageNet-LT CE e$Epochs" "imagenetlt_ce_e$Epochs" @(
    "--model", "ce", "--loss", "ce"
)

Invoke-Experiment "ImageNet-LT Balanced Softmax e$Epochs" "imagenetlt_balanced_softmax_e$Epochs" @(
    "--model", "ce", "--loss", "balanced_softmax"
)

Invoke-Experiment "ImageNet-LT LDAM-DRW e$Epochs" "imagenetlt_ldam_drw_e${Epochs}_s1" @(
    "--model", "ce", "--loss", "ldam", "--ldam-scale", "1",
    "--ldam-max-m", "0.5", "--ldam-reweight-beta", "0.9999",
    "--drw-epoch", "$DrwEpoch"
)

Invoke-Experiment "ImageNet-LT Fixed K$FixedK e$Epochs" "imagenetlt_fixed_k${FixedK}_e$Epochs" @(
    "--model", "proto", "--proto-mode", "fixed", "--fixed-k", "$FixedK",
    "--temperature", "0.1", "--pool-tau", "0.25"
)

Invoke-Experiment "ImageNet-LT Sqrt B$ProtoBudget e$Epochs" "imagenetlt_sqrt_budget${ProtoBudget}_e$Epochs" @(
    "--model", "adaptive_proto", "--allocation", "sqrt",
    "--proto-budget", "$ProtoBudget", "--k-min", "$KMin", "--k-max", "$KMax",
    "--loss", "ce", "--temperature", "0.1", "--pool-tau", "0.25"
)

Invoke-Experiment "ImageNet-LT Sqrt B$ProtoBudget + LDAM e$Epochs" "imagenetlt_sqrt_budget${ProtoBudget}_e${Epochs}_ldam_s1" @(
    "--model", "adaptive_proto", "--allocation", "sqrt",
    "--proto-budget", "$ProtoBudget", "--k-min", "$KMin", "--k-max", "$KMax",
    "--loss", "ldam", "--ldam-scale", "1", "--ldam-max-m", "0.5",
    "--ldam-reweight-beta", "0.9999", "--drw-epoch", "$DrwEpoch",
    "--temperature", "0.1", "--pool-tau", "0.25"
)

Write-Host "Log saved to $log"
