param(
    [string]$Python = "C:\Users\x2472\miniconda3\envs\torch\python.exe",
    [string]$Device = "cuda",
    [int]$Epochs = 50,
    [int]$BatchSize = 128,
    [double]$Lr = 0.1,
    [int[]]$Seeds = @(1, 2, 3),
    [int]$ProtoBudget = 300,
    [int]$KMin = 1,
    [int]$KMax = 8,
    [double]$ImbFactor = 100,
    [string]$Dataset = "cifar100lt",
    [string]$DataRoot = "data",
    [int]$Workers = 4
)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = "logs/linear_severe_stability_e50_$stamp.log"

function Format-ImbFactor {
    param([double]$Value)
    return ("{0:g}" -f $Value)
}

function Get-LinearRunName {
    param([int]$Seed)
    if ($Seed -eq 1) {
        return "adaptive_linear_budget${ProtoBudget}_e$Epochs"
    }
    return "adaptive_linear_budget${ProtoBudget}_e${Epochs}_seed$Seed"
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

function Run-Linear {
    param([int]$Seed)

    $ifTag = Format-ImbFactor $ImbFactor
    $runName = Get-LinearRunName $Seed

    Run-Step "Adaptive linear budget=$ProtoBudget IF=$ifTag seed=$Seed e$Epochs" $runName @(
        "train.py",
        "--dataset", $Dataset,
        "--data-root", $DataRoot,
        "--imb-factor", "$ifTag",
        "--model", "adaptive_proto",
        "--allocation", "linear",
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
        "--seed", "$Seed",
        "--run-name", $runName
    )
}

function Get-BestMetrics {
    param([string]$RunName)

    $metrics = Join-Path "runs" "$RunName\metrics.csv"
    if (-not (Test-Path $metrics)) {
        return $null
    }

    $rows = @(Import-Csv $metrics)
    if ($rows.Count -eq 0) {
        return $null
    }

    return $rows | Sort-Object {[double]$_.val_acc} -Descending | Select-Object -First 1
}

function Get-PrototypeCount {
    param([string]$RunName)

    $protoPath = Join-Path "runs" "$RunName\prototype_counts.json"
    if (-not (Test-Path $protoPath)) {
        return "N/A"
    }

    $protoJson = Get-Content $protoPath -Raw | ConvertFrom-Json
    return "$($protoJson.total_prototypes)"
}

function Get-Mean {
    param([double[]]$Values)
    if ($Values.Count -eq 0) {
        return [double]::NaN
    }
    return (($Values | Measure-Object -Average).Average)
}

function Get-Std {
    param([double[]]$Values)
    if ($Values.Count -le 1) {
        return 0.0
    }
    $mean = Get-Mean $Values
    $sumSq = 0.0
    foreach ($v in $Values) {
        $sumSq += [math]::Pow($v - $mean, 2)
    }
    return [math]::Sqrt($sumSq / ($Values.Count - 1))
}

foreach ($seed in $Seeds) {
    Run-Linear $seed
}

$comparison = @(
    [pscustomobject]@{ Label = "IF100 Fixed K3"; Runs = @("fixed_k3_t01_tau025_e50", "fixed_k3_t01_tau025_e50_seed2", "fixed_k3_t01_tau025_e50_seed3") },
    [pscustomobject]@{ Label = "IF100 Fixed K4"; Runs = @("fixed_k4_t01_tau025_e50", "fixed_k4_t01_tau025_e50_seed2", "fixed_k4_t01_tau025_e50_seed3") },
    [pscustomobject]@{ Label = "IF100 Sqrt B200"; Runs = @("adaptive_sqrt_budget200_e50", "adaptive_sqrt_budget200_e50_seed2", "adaptive_sqrt_budget200_e50_seed3") },
    [pscustomobject]@{ Label = "IF100 Linear B300"; Runs = @("adaptive_linear_budget300_e50", "adaptive_linear_budget300_e50_seed2", "adaptive_linear_budget300_e50_seed3") }
)

Write-Host "==== Per-seed Summary ===="
Add-Content -Path $log -Value "==== Per-seed Summary ===="

$aggregateRows = @()

foreach ($exp in $comparison) {
    $seed = 1
    foreach ($runName in $exp.Runs) {
        $best = Get-BestMetrics $runName
        if ($null -eq $best) {
            $msg = "$($exp.Label) seed=$seed ${runName}: missing metrics.csv"
            Write-Host $msg
            Add-Content -Path $log -Value $msg
            $seed += 1
            continue
        }

        $proto = Get-PrototypeCount $runName
        $aggregateRows += [pscustomobject]@{
            Label = $exp.Label
            Seed = $seed
            RunName = $runName
            Val = [double]$best.val_acc
            Many = [double]$best.many_acc
            Medium = [double]$best.medium_acc
            Few = [double]$best.few_acc
            BestEpoch = [int]$best.epoch
            Prototypes = $proto
        }

        $msg = "{0} seed={1}: run={2}, best_epoch={3}, val={4}, many={5}, medium={6}, few={7}, prototypes={8}" -f $exp.Label, $seed, $runName, $best.epoch, $best.val_acc, $best.many_acc, $best.medium_acc, $best.few_acc, $proto
        Write-Host $msg
        Add-Content -Path $log -Value $msg
        $seed += 1
    }
}

Write-Host "==== Mean/Std Summary ===="
Add-Content -Path $log -Value "==== Mean/Std Summary ===="

foreach ($group in ($aggregateRows | Group-Object Label)) {
    $vals = @($group.Group | ForEach-Object { [double]$_.Val })
    $many = @($group.Group | ForEach-Object { [double]$_.Many })
    $medium = @($group.Group | ForEach-Object { [double]$_.Medium })
    $few = @($group.Group | ForEach-Object { [double]$_.Few })
    $proto = ($group.Group | Select-Object -First 1).Prototypes

    $msg = "{0}: n={1}, val={2:N2}+/-{3:N2}, many={4:N2}+/-{5:N2}, medium={6:N2}+/-{7:N2}, few={8:N2}+/-{9:N2}, prototypes={10}" -f $group.Name, $vals.Count, (Get-Mean $vals), (Get-Std $vals), (Get-Mean $many), (Get-Std $many), (Get-Mean $medium), (Get-Std $medium), (Get-Mean $few), (Get-Std $few), $proto
    Write-Host $msg
    Add-Content -Path $log -Value $msg
}

Write-Host "Log saved to $log"
