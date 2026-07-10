param(
    [string]$Python = "C:\Users\x2472\miniconda3\envs\torch\python.exe",
    [string]$Device = "cuda",
    [int]$Epochs = 50,
    [int]$BatchSize = 128,
    [double]$Lr = 0.1,
    [int[]]$Seeds = @(1, 2, 3),
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
$log = "logs/key_stability_seeds_e50_$stamp.log"

function Format-ImbFactor {
    param([double]$Value)
    return ("{0:g}" -f $Value)
}

function Test-IsIf100 {
    param([double]$Value)
    return ([math]::Abs($Value - 100.0) -lt 0.000001)
}

function Get-BaseRunName {
    param(
        [double]$ImbFactor,
        [string]$Kind,
        [int]$Value
    )

    $ifTag = Format-ImbFactor $ImbFactor
    if ($Kind -eq "fixed") {
        if (Test-IsIf100 $ImbFactor) {
            return "fixed_k${Value}_t01_tau025_e$Epochs"
        }
        return "fixed_k${Value}_if${ifTag}_e$Epochs"
    }

    if ($Kind -eq "sqrt") {
        if (Test-IsIf100 $ImbFactor) {
            return "adaptive_sqrt_budget${Value}_e$Epochs"
        }
        return "adaptive_sqrt_budget${Value}_if${ifTag}_e$Epochs"
    }

    throw "Unknown kind: $Kind"
}

function Get-SeedRunName {
    param(
        [double]$ImbFactor,
        [string]$Kind,
        [int]$Value,
        [int]$Seed
    )

    if ($Seed -eq 1) {
        return Get-BaseRunName $ImbFactor $Kind $Value
    }

    $baseName = Get-BaseRunName $ImbFactor $Kind $Value
    return "${baseName}_seed$Seed"
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

function Run-Fixed {
    param(
        [double]$ImbFactor,
        [int]$K,
        [int]$Seed
    )

    $ifTag = Format-ImbFactor $ImbFactor
    $runName = Get-SeedRunName $ImbFactor "fixed" $K $Seed

    Run-Step "Fixed K=$K IF=$ifTag seed=$Seed e$Epochs" $runName @(
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
        "--seed", "$Seed",
        "--run-name", $runName
    )
}

function Run-Sqrt {
    param(
        [double]$ImbFactor,
        [int]$Budget,
        [int]$Seed
    )

    $ifTag = Format-ImbFactor $ImbFactor
    $runName = Get-SeedRunName $ImbFactor "sqrt" $Budget $Seed

    Run-Step "Adaptive sqrt budget=$Budget IF=$ifTag seed=$Seed e$Epochs" $runName @(
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

$experiments = @(
    [pscustomobject]@{ IF = 100.0; Kind = "fixed"; Value = 3; Label = "IF100 Fixed K3" },
    [pscustomobject]@{ IF = 100.0; Kind = "fixed"; Value = 4; Label = "IF100 Fixed K4" },
    [pscustomobject]@{ IF = 100.0; Kind = "sqrt"; Value = 200; Label = "IF100 Sqrt B200" },
    [pscustomobject]@{ IF = 50.0; Kind = "fixed"; Value = 3; Label = "IF50 Fixed K3" },
    [pscustomobject]@{ IF = 50.0; Kind = "fixed"; Value = 4; Label = "IF50 Fixed K4" },
    [pscustomobject]@{ IF = 50.0; Kind = "sqrt"; Value = 300; Label = "IF50 Sqrt B300" },
    [pscustomobject]@{ IF = 50.0; Kind = "sqrt"; Value = 350; Label = "IF50 Sqrt B350" }
)

foreach ($exp in $experiments) {
    foreach ($seed in $Seeds) {
        if ($exp.Kind -eq "fixed") {
            Run-Fixed $exp.IF $exp.Value $seed
        } elseif ($exp.Kind -eq "sqrt") {
            Run-Sqrt $exp.IF $exp.Value $seed
        } else {
            throw "Unknown kind: $($exp.Kind)"
        }
    }
}

Write-Host "==== Per-seed Summary ===="
Add-Content -Path $log -Value "==== Per-seed Summary ===="

$aggregateRows = @()

foreach ($exp in $experiments) {
    foreach ($seed in $Seeds) {
        $runName = Get-SeedRunName $exp.IF $exp.Kind $exp.Value $seed
        $best = Get-BestMetrics $runName
        if ($null -eq $best) {
            $msg = "$($exp.Label) seed=$seed ${runName}: missing metrics.csv"
            Write-Host $msg
            Add-Content -Path $log -Value $msg
            continue
        }

        $protoPath = Join-Path "runs" "$runName\prototype_counts.json"
        $protoInfo = "N/A"
        if (Test-Path $protoPath) {
            $protoJson = Get-Content $protoPath -Raw | ConvertFrom-Json
            $protoInfo = "$($protoJson.total_prototypes)"
        }

        $aggregateRows += [pscustomobject]@{
            Label = $exp.Label
            RunName = $runName
            Seed = $seed
            Val = [double]$best.val_acc
            Many = [double]$best.many_acc
            Medium = [double]$best.medium_acc
            Few = [double]$best.few_acc
            BestEpoch = [int]$best.epoch
            Prototypes = $protoInfo
        }

        $msg = "{0} seed={1}: run={2}, best_epoch={3}, val={4}, many={5}, medium={6}, few={7}, prototypes={8}" -f $exp.Label, $seed, $runName, $best.epoch, $best.val_acc, $best.many_acc, $best.medium_acc, $best.few_acc, $protoInfo
        Write-Host $msg
        Add-Content -Path $log -Value $msg
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

    $msg = "{0}: n={1}, val={2:N2}±{3:N2}, many={4:N2}±{5:N2}, medium={6:N2}±{7:N2}, few={8:N2}±{9:N2}, prototypes={10}" -f $group.Name, $vals.Count, (Get-Mean $vals), (Get-Std $vals), (Get-Mean $many), (Get-Std $many), (Get-Mean $medium), (Get-Std $medium), (Get-Mean $few), (Get-Std $few), $proto
    Write-Host $msg
    Add-Content -Path $log -Value $msg
}

Write-Host "Log saved to $log"
