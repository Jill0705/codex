param(
    [string]$TablesDir = "tables"
)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
New-Item -ItemType Directory -Force -Path $TablesDir | Out-Null

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

function Get-PrototypeInfo {
    param([string]$RunName)
    $protoPath = Join-Path "runs" "$RunName\prototype_counts.json"
    if (-not (Test-Path $protoPath)) {
        return [pscustomobject]@{ Prototypes = "N/A"; KDist = "N/A" }
    }
    $protoJson = Get-Content $protoPath -Raw | ConvertFrom-Json
    $dist = @($protoJson.proto_counts) | Group-Object | Sort-Object {[int]$_.Name}
    return [pscustomobject]@{
        Prototypes = "$($protoJson.total_prototypes)"
        KDist = (($dist | ForEach-Object { "K=$($_.Name):$($_.Count)" }) -join " ")
    }
}

function New-ResultRow {
    param(
        [string]$Method,
        [string]$IF,
        [string]$Seed,
        [string]$Budget,
        [string]$RunName,
        [string]$Role = "main"
    )

    $best = Get-BestMetrics $RunName
    $proto = Get-PrototypeInfo $RunName
    if ($null -eq $best) {
        return [pscustomobject]@{
            method = $Method
            IF = $IF
            seed = $Seed
            budget = $Budget
            role = $Role
            run_name = $RunName
            status = "missing"
            prototypes = $proto.Prototypes
            k_distribution = $proto.KDist
            best_epoch = ""
            val = ""
            many = ""
            medium = ""
            few = ""
        }
    }

    return [pscustomobject]@{
        method = $Method
        IF = $IF
        seed = $Seed
        budget = $Budget
        role = $Role
        run_name = $RunName
        status = "ok"
        prototypes = $proto.Prototypes
        k_distribution = $proto.KDist
        best_epoch = $best.epoch
        val = $best.val_acc
        many = $best.many_acc
        medium = $best.medium_acc
        few = $best.few_acc
    }
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

function New-AggregateRows {
    param([object[]]$Rows)

    $out = @()
    foreach ($group in ($Rows | Where-Object { $_.status -eq "ok" } | Group-Object method, IF, budget, role)) {
        $items = @($group.Group)
        $first = $items | Select-Object -First 1
        $vals = @($items | ForEach-Object { [double]$_.val })
        $many = @($items | ForEach-Object { [double]$_.many })
        $medium = @($items | ForEach-Object { [double]$_.medium })
        $few = @($items | ForEach-Object { [double]$_.few })
        $out += [pscustomobject]@{
            method = $first.method
            IF = $first.IF
            budget = $first.budget
            role = $first.role
            n = $vals.Count
            prototypes = $first.prototypes
            val_mean = "{0:N2}" -f (Get-Mean $vals)
            val_std = "{0:N2}" -f (Get-Std $vals)
            many_mean = "{0:N2}" -f (Get-Mean $many)
            many_std = "{0:N2}" -f (Get-Std $many)
            medium_mean = "{0:N2}" -f (Get-Mean $medium)
            medium_std = "{0:N2}" -f (Get-Std $medium)
            few_mean = "{0:N2}" -f (Get-Mean $few)
            few_std = "{0:N2}" -f (Get-Std $few)
        }
    }
    return $out
}

function Write-Table {
    param(
        [string]$FileName,
        [object[]]$Rows
    )
    $path = Join-Path $TablesDir $FileName
    $Rows | Export-Csv -Path $path -NoTypeInformation -Encoding UTF8
    Write-Host "Wrote $path ($($Rows.Count) rows)"
}

$ccfbIf100 = @(
    New-ResultRow "CE" "100" "1" "N/A" "ce_e50"
    New-ResultRow "Balanced Softmax" "100" "1" "N/A" "balanced_softmax_if100_e50"
    New-ResultRow "LDAM-DRW" "100" "1" "N/A" "ldam_drw_if100_e50"
    New-ResultRow "Fixed K4" "100" "1" "400" "fixed_k4_t01_tau025_e50"
    New-ResultRow "Sqrt B200" "100" "1" "200" "adaptive_sqrt_budget200_e50"
    New-ResultRow "Sqrt B200 + BS" "100" "1" "200" "adaptive_sqrt_budget200_if100_e50_bs" "optional"
)

$ccfbIf50 = @(
    New-ResultRow "CE" "50" "1" "N/A" "ce_if50_e50"
    New-ResultRow "Balanced Softmax" "50" "1" "N/A" "balanced_softmax_if50_e50"
    New-ResultRow "LDAM-DRW" "50" "1" "N/A" "ldam_drw_if50_e50"
    New-ResultRow "Fixed K3" "50" "1" "300" "fixed_k3_if50_e50"
    New-ResultRow "Fixed K4" "50" "1" "400" "fixed_k4_if50_e50"
    New-ResultRow "Sqrt B300" "50" "1" "300" "adaptive_sqrt_budget300_if50_e50"
    New-ResultRow "Sqrt B300 + BS" "50" "1" "300" "adaptive_sqrt_budget300_if50_e50_bs" "optional"
)

$stabilityRows = @(
    New-ResultRow "IF100 Fixed K4" "100" "1" "400" "fixed_k4_t01_tau025_e50"
    New-ResultRow "IF100 Fixed K4" "100" "2" "400" "fixed_k4_t01_tau025_e50_seed2"
    New-ResultRow "IF100 Fixed K4" "100" "3" "400" "fixed_k4_t01_tau025_e50_seed3"
    New-ResultRow "IF100 Sqrt B200" "100" "1" "200" "adaptive_sqrt_budget200_e50"
    New-ResultRow "IF100 Sqrt B200" "100" "2" "200" "adaptive_sqrt_budget200_e50_seed2"
    New-ResultRow "IF100 Sqrt B200" "100" "3" "200" "adaptive_sqrt_budget200_e50_seed3"
    New-ResultRow "IF50 Fixed K3" "50" "1" "300" "fixed_k3_if50_e50"
    New-ResultRow "IF50 Fixed K3" "50" "2" "300" "fixed_k3_if50_e50_seed2"
    New-ResultRow "IF50 Fixed K3" "50" "3" "300" "fixed_k3_if50_e50_seed3"
    New-ResultRow "IF50 Fixed K4" "50" "1" "400" "fixed_k4_if50_e50"
    New-ResultRow "IF50 Fixed K4" "50" "2" "400" "fixed_k4_if50_e50_seed2"
    New-ResultRow "IF50 Fixed K4" "50" "3" "400" "fixed_k4_if50_e50_seed3"
    New-ResultRow "IF50 Sqrt B300" "50" "1" "300" "adaptive_sqrt_budget300_if50_e50"
    New-ResultRow "IF50 Sqrt B300" "50" "2" "300" "adaptive_sqrt_budget300_if50_e50_seed2"
    New-ResultRow "IF50 Sqrt B300" "50" "3" "300" "adaptive_sqrt_budget300_if50_e50_seed3"
)

$negativeRows = @(
    New-ResultRow "Linear B300 unstable" "100" "1" "300" "adaptive_linear_budget300_e50" "negative"
    New-ResultRow "Linear B300 unstable" "100" "2" "300" "adaptive_linear_budget300_e50_seed2" "negative"
    New-ResultRow "Linear B300 unstable" "100" "3" "300" "adaptive_linear_budget300_e50_seed3" "negative"
    New-ResultRow "Adaptive K4 + EMA" "100" "1" "250" "adaptive_k4_t01_tau025_e50_ema" "negative"
    New-ResultRow "Adaptive K4 + conf-EMA" "100" "1" "250" "adaptive_k4_t01_tau025_e50_confema" "negative"
    New-ResultRow "Adaptive K8 + EMA" "100" "1" "449" "adaptive_k8_t01_tau025_e50_ema" "negative"
    New-ResultRow "Adaptive K8 + conf-EMA" "100" "1" "449" "adaptive_k8_t01_tau025_e50_confema" "negative"
    New-ResultRow "CE + LA" "100" "1" "N/A" "ce_e50_la" "negative"
    New-ResultRow "Fixed K4 + LA" "100" "1" "400" "fixed_k4_t01_tau025_e50_la" "negative"
    New-ResultRow "Adaptive K4 + LA" "100" "1" "250" "adaptive_k4_t01_tau025_e50_la" "negative"
    New-ResultRow "Adaptive K8 + LA" "100" "1" "449" "adaptive_k8_t01_tau025_e50_la" "negative"
)

Write-Table "ccfb_main_if100.csv" $ccfbIf100
Write-Table "ccfb_main_if50.csv" $ccfbIf50
Write-Table "ccfb_stability_seeds.csv" $stabilityRows
Write-Table "ccfb_stability_summary.csv" (New-AggregateRows $stabilityRows)
Write-Table "negative_modules.csv" $negativeRows
