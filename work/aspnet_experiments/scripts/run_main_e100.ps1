param(
    [string]$Python = "C:\Users\x2472\miniconda3\envs\torch\python.exe",
    [string]$Device = "cuda",
    [int]$Epochs = 100,
    [int]$DrwEpoch = 60,
    [int]$BatchSize = 128,
    [int]$Workers = 4,
    [double]$Lr = 0.1,
    [string]$Dataset = "cifar100lt",
    [string]$DataRoot = "data"
)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = "logs/main_e${Epochs}_$stamp.log"

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

function Invoke-Experiment {
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

    & $Python @ArgsList 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) {
        throw "FAILED: $Name. See $log"
    }
}

function Invoke-PyCompile {
    Write-Host "==== py_compile ===="
    & $Python -m py_compile `
        "train.py" `
        "aspnet_lt\classifiers.py" `
        "aspnet_lt\data.py" `
        "aspnet_lt\losses.py" `
        "aspnet_lt\resnet_cifar.py" `
        "aspnet_lt\utils.py" `
        "analyze_structure.py"
    if ($LASTEXITCODE -ne 0) {
        throw "py_compile failed"
    }
}

function Invoke-CudaCheck {
    Write-Host "==== CUDA check ===="
    & $Python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA check failed"
    }
}

function Add-CeRun {
    param([string]$IfTag)
    $runName = if ($IfTag -eq "100") { "ce_e$Epochs" } else { "ce_if${IfTag}_e$Epochs" }
    Invoke-Experiment "IF=$IfTag CE e$Epochs" $runName @(
        "train.py", "--dataset", $Dataset, "--data-root", $DataRoot, "--imb-factor", $IfTag,
        "--model", "ce", "--loss", "ce", "--epochs", "$Epochs",
        "--batch-size", "$BatchSize", "--lr", "$Lr", "--workers", "$Workers",
        "--device", $Device, "--run-name", $runName
    )
}

function Add-BalancedSoftmaxRun {
    param([string]$IfTag)
    $runName = "balanced_softmax_if${IfTag}_e$Epochs"
    Invoke-Experiment "IF=$IfTag Balanced Softmax e$Epochs" $runName @(
        "train.py", "--dataset", $Dataset, "--data-root", $DataRoot, "--imb-factor", $IfTag,
        "--model", "ce", "--loss", "balanced_softmax", "--epochs", "$Epochs",
        "--batch-size", "$BatchSize", "--lr", "$Lr", "--workers", "$Workers",
        "--device", $Device, "--run-name", $runName
    )
}

function Add-LdamRun {
    param([string]$IfTag)
    $runName = "ldam_drw_if${IfTag}_e${Epochs}_s1"
    Invoke-Experiment "IF=$IfTag LDAM-DRW e$Epochs scale=1" $runName @(
        "train.py", "--dataset", $Dataset, "--data-root", $DataRoot, "--imb-factor", $IfTag,
        "--model", "ce", "--loss", "ldam", "--ldam-scale", "1", "--ldam-max-m", "0.5",
        "--ldam-reweight-beta", "0.9999", "--drw-epoch", "$DrwEpoch",
        "--epochs", "$Epochs", "--batch-size", "$BatchSize", "--lr", "$Lr",
        "--workers", "$Workers", "--device", $Device, "--run-name", $runName
    )
}

function Add-FixedRun {
    param(
        [string]$IfTag,
        [int]$K
    )
    $runName = if ($IfTag -eq "100") { "fixed_k${K}_t01_tau025_e$Epochs" } else { "fixed_k${K}_if${IfTag}_e$Epochs" }
    Invoke-Experiment "IF=$IfTag Fixed K=$K e$Epochs" $runName @(
        "train.py", "--dataset", $Dataset, "--data-root", $DataRoot, "--imb-factor", $IfTag,
        "--model", "proto", "--proto-mode", "fixed", "--fixed-k", "$K",
        "--epochs", "$Epochs", "--batch-size", "$BatchSize", "--lr", "$Lr",
        "--workers", "$Workers", "--device", $Device, "--temperature", "0.1",
        "--pool-tau", "0.25", "--run-name", $runName
    )
}

function Add-SqrtRun {
    param(
        [string]$IfTag,
        [int]$Budget
    )
    $runName = if ($IfTag -eq "100") { "adaptive_sqrt_budget${Budget}_e$Epochs" } else { "adaptive_sqrt_budget${Budget}_if${IfTag}_e$Epochs" }
    Invoke-Experiment "IF=$IfTag Sqrt B=$Budget e$Epochs" $runName @(
        "train.py", "--dataset", $Dataset, "--data-root", $DataRoot, "--imb-factor", $IfTag,
        "--model", "adaptive_proto", "--allocation", "sqrt", "--proto-budget", "$Budget",
        "--k-min", "1", "--k-max", "8", "--loss", "ce",
        "--epochs", "$Epochs", "--batch-size", "$BatchSize", "--lr", "$Lr",
        "--workers", "$Workers", "--device", $Device, "--temperature", "0.1",
        "--pool-tau", "0.25", "--run-name", $runName
    )
}

function Add-SqrtLdamRun {
    param(
        [string]$IfTag,
        [int]$Budget
    )
    $runName = "adaptive_sqrt_budget${Budget}_if${IfTag}_e${Epochs}_ldam_s1"
    Invoke-Experiment "IF=$IfTag Sqrt B=$Budget + LDAM e$Epochs" $runName @(
        "train.py", "--dataset", $Dataset, "--data-root", $DataRoot, "--imb-factor", $IfTag,
        "--model", "adaptive_proto", "--allocation", "sqrt", "--proto-budget", "$Budget",
        "--k-min", "1", "--k-max", "8", "--loss", "ldam",
        "--ldam-scale", "1", "--ldam-max-m", "0.5", "--ldam-reweight-beta", "0.9999",
        "--drw-epoch", "$DrwEpoch", "--epochs", "$Epochs",
        "--batch-size", "$BatchSize", "--lr", "$Lr", "--workers", "$Workers",
        "--device", $Device, "--temperature", "0.1", "--pool-tau", "0.25",
        "--run-name", $runName
    )
}

Invoke-PyCompile
Invoke-CudaCheck

Add-CeRun "100"
Add-BalancedSoftmaxRun "100"
Add-LdamRun "100"
Add-FixedRun "100" 4
Add-SqrtRun "100" 200
Add-SqrtLdamRun "100" 200

Add-CeRun "50"
Add-BalancedSoftmaxRun "50"
Add-LdamRun "50"
Add-FixedRun "50" 3
Add-FixedRun "50" 4
Add-SqrtRun "50" 300
Add-SqrtLdamRun "50" 300

$summaryNames = @(
    "ce_e$Epochs",
    "balanced_softmax_if100_e$Epochs",
    "ldam_drw_if100_e${Epochs}_s1",
    "fixed_k4_t01_tau025_e$Epochs",
    "adaptive_sqrt_budget200_e$Epochs",
    "adaptive_sqrt_budget200_if100_e${Epochs}_ldam_s1",
    "ce_if50_e$Epochs",
    "balanced_softmax_if50_e$Epochs",
    "ldam_drw_if50_e${Epochs}_s1",
    "fixed_k3_if50_e$Epochs",
    "fixed_k4_if50_e$Epochs",
    "adaptive_sqrt_budget300_if50_e$Epochs",
    "adaptive_sqrt_budget300_if50_e${Epochs}_ldam_s1"
)

Write-Host "==== E$Epochs Summary ===="
Add-Content -Path $log -Value "==== E$Epochs Summary ===="
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
