param(
  [switch]$AllTest,
  [int]$LastN = 101,
  [double]$Threshold = 0.5,
  [string]$CsvPath = "masters_csvs/matches.csv",
  [string]$ModelPath = "mvp_model/artifacts/model.pkl",
  [string]$MetricsPath = "mvp_model/artifacts/metrics.json",
  [string]$TrainInfoPath = "mvp_model/artifacts/train_info.json",
  [string]$PlotsDir = "mvp_model/artifacts/plots",
  [string]$TailCsv = "mvp_model/artifacts/test_tail_preds.csv"
)

$ErrorActionPreference = 'Stop'

# Change to repo root (parent of this script)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Resolve-Path (Join-Path $ScriptDir '..')
Push-Location $Root
Write-Host "Repo root: $Root" -ForegroundColor Yellow

# Try to activate Windows venv if present
$VenvActivate = Join-Path $Root ".venv\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
  Write-Host "Activando .venv..." -ForegroundColor Green
  . $VenvActivate
}

function Assert-LastExit {
  param([string]$Step)
  if ($LASTEXITCODE -ne 0) { throw "Fallo en: $Step (exit $LASTEXITCODE)" }
}

Write-Host "[1/5] Unificando maestros" -ForegroundColor Cyan
python scripts/merge_tournaments_to_masters.py
Assert-LastExit "merge_tournaments_to_masters"

Write-Host "[2/5] Generando join por match_id" -ForegroundColor Cyan
python scripts/join_matches_by_match_id.py
Assert-LastExit "join_matches_by_match_id"

Write-Host "[3/5] Entrenando modelo" -ForegroundColor Cyan
python -m mvp_model.train_mvp --csv-path $CsvPath --model-out $ModelPath --metrics-out $MetricsPath --train-info-out $TrainInfoPath
Assert-LastExit "train_mvp"

Write-Host "[4/5] Exportando bloque de test a CSV" -ForegroundColor Cyan
$tailArgs = @('--csv-path', $CsvPath, '--model', $ModelPath, '--out', $TailCsv, '--threshold', "$Threshold")
if ($UseAllTest) { $tailArgs += '--all-test' } else { $tailArgs += @('--last-n', "$LastN") }
python -m mvp_model.print_test_tail @tailArgs
Assert-LastExit "print_test_tail"

Write-Host "[5/5] Generando gráficas" -ForegroundColor Cyan
# Por defecto usamos TODO el test, a menos que el usuario pase -LastN explícitamente
$UseAllTest = $true
if ($PSBoundParameters.ContainsKey('LastN')) { $UseAllTest = $false }
if ($PSBoundParameters.ContainsKey('AllTest')) { $UseAllTest = [bool]$AllTest }

$plotArgs = @('--csv-path', $CsvPath, '--model', $ModelPath, '--out-dir', $PlotsDir, '--test-size', '0.2')
if ($UseAllTest) { $plotArgs += '--all-test' } else { $plotArgs += @('--last-n', "$LastN") }
python -m mvp_model.plot_test_predictions @plotArgs
Assert-LastExit "plot_test_predictions"

Write-Host "\nListo. Salidas principales:" -ForegroundColor Yellow
Write-Host " - Modelo: $ModelPath"
Write-Host " - Metricas (test completo): $MetricsPath"
Write-Host " - Tail CSV (ultimos $LastN): $TailCsv"
Write-Host " - Graficas: $PlotsDir\test_predictions_timeseries.png, $PlotsDir\test_calibration_curve.png"

Pop-Location
