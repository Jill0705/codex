param(
    [string]$Python = "C:\Users\x2472\miniconda3\envs\torch\python.exe",
    [string]$Device = "cuda",
    [int]$Epochs = 50,
    [int]$BatchSize = 128,
    [double]$Lr = 0.1,
    [double[]]$ImbFactors = @(100, 50),
    [int[]]$SqrtBudgets = @(200, 250, 300, 350),
    [int[]]$FixedKs = @(2, 3, 4),
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
$log = "logs/sqrt_budget_curve_e50_$stamp.log"

function Format-ImbFactor {
    param([double]$Value)
    return ("{0:g}" -f $Value)
}

function Test-IsIf100 {
    param([double]$Value)
    return ([math]::Abs($Value - 100.0) -lt 0.000001)
}

function Get-CeRunName {
    param([double]$ImbFactor)
    if (Test-IsIf100 $ImbFactor) {
        return "ce_e$Epochs"
    }
    $ifTag = Format-ImbFactor $ImbFactor
    return "ce_if${ifTag}_e$Epochs"
}

function Get-FixedRunName {
    param(
        [double]$ImbFactor,
        [int]$K
    )
    if (Test-IsIf100 $ImbFactor) {
        return "fixed_k${K}_t01_tau025_e$Epochs"
    }
    $ifTag = Format-ImbFactor $ImbFactor
    return "fixed_k${K}_if${ifTag}_e$Epochs"
}

function Get-SqrtRunName {
    param(
        [double]$ImbFactor,
        [int]$Budget
    )
    if (Test-IsIf100 $ImbFactor) {
        return "adaptive_sqrt_budget${Budget}_e$Epochs"
    }
    $ifTag = Format-ImbFactor $ImbFactor
    return "adaptive_sqrt_budget${Budget}_if${ifTag}_e$Epochs"
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
    param([double]$ImbFactor)

    $ifTag = Format-ImbFactor $ImbFactor
    $runName = Get-CeRunName $ImbFactor

    Run-Step "CE IF=$ifTag e$Epochs" $runName @(
        "train.py",
        "--dataset", $Dataset,
        "--data-root", $DataRoot,
        "--imb-factor", "$ifTag",
        "--model", "ce",
        "--epochs", "$Epochs",
        "--batch-size", "$BatchSize",
        "--lr", "$Lr",
        "--workers", "$Workers",
        "--device", $Device,
        "--run-name", $runName
    )
}

function Run-Fixed {
    param(
        [double]$ImbFactor,
        [int]$K
    )

    $ifTag = Format-ImbFactor $ImbFactor
    $runName = Get-FixedRunName $ImbFactor $K

    Run-Step "Fixed K=$K IF=$ifTag t=0.1 tau=0.25 e$Epochs" $runName @(
        "train.py",
        "--dataset", $Dataset,
        "--data-root", $DataRoot,
        "--imb-factor", "$ifTag",
        "--model", "proto",
        "--proto-mode", "fixed",
        "--fixed-k", "$K",
        "--epochs", "$Epochs",
        "--batch-size", "$BatchSize",
        "--lr", "$Lr",
        "--workers", "$Workers",
        "--device", $Device,
        "--temperature", "0.1",
        "--pool-tau", "0.25",
        "--run-name", $runName
    )
}

function Run-Sqrt {
    param(
        [double]$ImbFactor,
        [int]$Budget
    )

    $ifTag = Format-ImbFactor $ImbFactor
    $runName = Get-SqrtRunName $ImbFactor $Budget

    Run-Step "Adaptive sqrt budget=$Budget IF=$ifTag e$Epochs" $runName @(
        "train.py",
        "--dataset", $Dataset,
        "--data-root", $DataRoot,
        "--imb-factor", "$ifTag",
        "--model", "adaptive_proto",
        "--allocation", "sqrt",
        "--proto-budget", "$Budget",
        "--k-min", "$KMin",
        "--k-max", "$KMax",
        "--epochs", "$Epochs",
        "--batch-size", "$BatchSize",
        "--lr", "$Lr",
        "--workers", "$Workers",
        "--device", $Device,
        "--temperature", "0.1",
        "--pool-tau", "0.25",
        "--run-name", $runName
    )
}

$summaryRows = @()

foreach ($imb in $ImbFactors) {
    $ifTag = Format-ImbFactor $imb

    Run-Ce $imb
    $summaryRows += [pscustomobject]@{ IF = $ifTag; Name = Get-CeRunName $imb }

    foreach ($k in $FixedKs) {
        Run-Fixed $imb $k
        $summaryRows += [pscustomobject]@{ IF = $ifTag; Name = Get-FixedRunName $imb $k }
    }

    foreach ($budget in $SqrtBudgets) {
        Run-Sqrt $imb $budget
        $summaryRows += [pscustomobject]@{ IF = $ifTag; Name = Get-SqrtRunName $imb $budget }
    }
}

Write-Host "==== Summary ===="
Add-Content -Path $log -Value "==== Summary ===="

foreach ($group in ($summaryRows | Group-Object IF)) {
    $header = "---- IF=$($group.Name) ----"
    Write-Host $header
    Add-Content -Path $log -Value $header

    foreach ($row in $group.Group) {
        $n = $row.Name
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
}

Write-Host "Log saved to $log"
