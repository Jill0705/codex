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
        return [pscustomobject]@{
            Prototypes = "N/A"
            KDist = "N/A"
        }
    }

    $protoJson = Get-Content $protoPath -Raw | ConvertFrom-Json
    $dist = @($protoJson.proto_counts) | Group-Object | Sort-Object {[int]$_.Name}
    $distText = (($dist | ForEach-Object { "K=$($_.Name):$($_.Count)" }) -join " ")

    return [pscustomobject]@{
        Prototypes = "$($protoJson.total_prototypes)"
        KDist = $distText
    }
}

function New-ResultRow {
    param(
        [string]$Method,
        [string]$IF,
        [string]$Seed,
        [string]$Budget,
        [string]$RunName
    )

    $best = Get-BestMetrics $RunName
    $proto = Get-PrototypeInfo $RunName

    if ($null -eq $best) {
        return [pscustomobject]@{
            method = $Method
            IF = $IF
            seed = $Seed
            budget = $Budget
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

function Write-Table {
    param(
        [string]$FileName,
        [object[]]$Rows
    )

    $path = Join-Path $TablesDir $FileName
    $Rows | Export-Csv -Path $path -NoTypeInformation -Encoding UTF8
    Write-Host "Wrote $path ($($Rows.Count) rows)"
}

$if100Rows = @(
    New-ResultRow "CE" "100" "1" "N/A" "ce_e50"
    New-ResultRow "Fixed K2" "100" "1" "200" "fixed_k2_t01_tau025_e50"
    New-ResultRow "Fixed K3" "100" "1" "300" "fixed_k3_t01_tau025_e50"
    New-ResultRow "Fixed K4" "100" "1" "400" "fixed_k4_t01_tau025_e50"
    New-ResultRow "Sqrt B200" "100" "1" "200" "adaptive_sqrt_budget200_e50"
    New-ResultRow "Sqrt B250" "100" "1" "250" "adaptive_sqrt_budget250_e50"
    New-ResultRow "Sqrt B300" "100" "1" "300" "adaptive_sqrt_budget300_e50"
    New-ResultRow "Sqrt B350" "100" "1" "350" "adaptive_sqrt_budget350_e50"
    New-ResultRow "Linear B300" "100" "1" "300" "adaptive_linear_budget300_e50"
)

$if50Rows = @(
    New-ResultRow "CE" "50" "1" "N/A" "ce_if50_e50"
    New-ResultRow "Fixed K2" "50" "1" "200" "fixed_k2_if50_e50"
    New-ResultRow "Fixed K3" "50" "1" "300" "fixed_k3_if50_e50"
    New-ResultRow "Fixed K4" "50" "1" "400" "fixed_k4_if50_e50"
    New-ResultRow "Sqrt B200" "50" "1" "200" "adaptive_sqrt_budget200_if50_e50"
    New-ResultRow "Sqrt B250" "50" "1" "250" "adaptive_sqrt_budget250_if50_e50"
    New-ResultRow "Sqrt B300" "50" "1" "300" "adaptive_sqrt_budget300_if50_e50"
    New-ResultRow "Sqrt B350" "50" "1" "350" "adaptive_sqrt_budget350_if50_e50"
)

$stabilityRows = @(
    New-ResultRow "IF100 Fixed K3" "100" "1" "300" "fixed_k3_t01_tau025_e50"
    New-ResultRow "IF100 Fixed K3" "100" "2" "300" "fixed_k3_t01_tau025_e50_seed2"
    New-ResultRow "IF100 Fixed K3" "100" "3" "300" "fixed_k3_t01_tau025_e50_seed3"
    New-ResultRow "IF100 Fixed K4" "100" "1" "400" "fixed_k4_t01_tau025_e50"
    New-ResultRow "IF100 Fixed K4" "100" "2" "400" "fixed_k4_t01_tau025_e50_seed2"
    New-ResultRow "IF100 Fixed K4" "100" "3" "400" "fixed_k4_t01_tau025_e50_seed3"
    New-ResultRow "IF100 Sqrt B200" "100" "1" "200" "adaptive_sqrt_budget200_e50"
    New-ResultRow "IF100 Sqrt B200" "100" "2" "200" "adaptive_sqrt_budget200_e50_seed2"
    New-ResultRow "IF100 Sqrt B200" "100" "3" "200" "adaptive_sqrt_budget200_e50_seed3"
    New-ResultRow "IF100 Linear B300" "100" "1" "300" "adaptive_linear_budget300_e50"
    New-ResultRow "IF100 Linear B300" "100" "2" "300" "adaptive_linear_budget300_e50_seed2"
    New-ResultRow "IF100 Linear B300" "100" "3" "300" "adaptive_linear_budget300_e50_seed3"
    New-ResultRow "IF50 Fixed K3" "50" "1" "300" "fixed_k3_if50_e50"
    New-ResultRow "IF50 Fixed K3" "50" "2" "300" "fixed_k3_if50_e50_seed2"
    New-ResultRow "IF50 Fixed K3" "50" "3" "300" "fixed_k3_if50_e50_seed3"
    New-ResultRow "IF50 Fixed K4" "50" "1" "400" "fixed_k4_if50_e50"
    New-ResultRow "IF50 Fixed K4" "50" "2" "400" "fixed_k4_if50_e50_seed2"
    New-ResultRow "IF50 Fixed K4" "50" "3" "400" "fixed_k4_if50_e50_seed3"
    New-ResultRow "IF50 Sqrt B300" "50" "1" "300" "adaptive_sqrt_budget300_if50_e50"
    New-ResultRow "IF50 Sqrt B300" "50" "2" "300" "adaptive_sqrt_budget300_if50_e50_seed2"
    New-ResultRow "IF50 Sqrt B300" "50" "3" "300" "adaptive_sqrt_budget300_if50_e50_seed3"
    New-ResultRow "IF50 Sqrt B350" "50" "1" "350" "adaptive_sqrt_budget350_if50_e50"
    New-ResultRow "IF50 Sqrt B350" "50" "2" "350" "adaptive_sqrt_budget350_if50_e50_seed2"
    New-ResultRow "IF50 Sqrt B350" "50" "3" "350" "adaptive_sqrt_budget350_if50_e50_seed3"
)

$allocationIf100Rows = @(
    New-ResultRow "Fixed K3" "100" "1" "300" "fixed_k3_t01_tau025_e50"
    New-ResultRow "Log B300" "100" "1" "300" "adaptive_log_budget300_e50"
    New-ResultRow "Sqrt B300" "100" "1" "300" "adaptive_sqrt_budget300_e50"
    New-ResultRow "Linear B300" "100" "1" "300" "adaptive_linear_budget300_e50"
    New-ResultRow "Effective B300" "100" "1" "300" "adaptive_effective_budget300_e50"
)

$ifSweepRows = @(
    New-ResultRow "CE" "100" "1" "N/A" "ce_e50"
    New-ResultRow "Fixed K3" "100" "1" "300" "fixed_k3_t01_tau025_e50"
    New-ResultRow "Sqrt B300" "100" "1" "300" "adaptive_sqrt_budget300_e50"
    New-ResultRow "Linear B300" "100" "1" "300" "adaptive_linear_budget300_e50"
    New-ResultRow "CE" "50" "1" "N/A" "ce_if50_e50"
    New-ResultRow "Fixed K3" "50" "1" "300" "fixed_k3_if50_e50"
    New-ResultRow "Sqrt B300" "50" "1" "300" "adaptive_sqrt_budget300_if50_e50"
    New-ResultRow "Linear B300" "50" "1" "300" "adaptive_linear_budget300_if50_e50"
    New-ResultRow "CE" "10" "1" "N/A" "ce_if10_e50"
    New-ResultRow "Fixed K3" "10" "1" "300" "fixed_k3_if10_e50"
    New-ResultRow "Sqrt B300" "10" "1" "300" "adaptive_sqrt_budget300_if10_e50"
    New-ResultRow "Linear B300" "10" "1" "300" "adaptive_linear_budget300_if10_e50"
)

Write-Table "main_budget_efficiency_if100.csv" $if100Rows
Write-Table "main_budget_efficiency_if50.csv" $if50Rows
Write-Table "stability_seeds.csv" $stabilityRows
Write-Table "allocation_function_if100.csv" $allocationIf100Rows
Write-Table "if_sweep_summary.csv" $ifSweepRows
