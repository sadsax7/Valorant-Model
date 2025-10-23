#!/usr/bin/env bash
set -euo pipefail

# Defaults
ALL_TEST=true
LAST_N=10
THRESH=0.5
CSV_PATH="masters_csvs/matches.csv"
MODEL_PATH="mvp_model/artifacts/model.pkl"
METRICS_PATH="mvp_model/artifacts/metrics.json"
TRAIN_INFO_PATH="mvp_model/artifacts/train_info.json"
PLOTS_DIR="mvp_model/artifacts/plots"
TAIL_CSV="mvp_model/artifacts/test_tail_preds.csv"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --all-test) ALL_TEST=true; shift ;;
    --last-n) LAST_N="$2"; ALL_TEST=false; shift 2 ;;
    --threshold) THRESH="$2"; shift 2 ;;
    --csv-path) CSV_PATH="$2"; shift 2 ;;
    --model) MODEL_PATH="$2"; shift 2 ;;
    --out-dir) PLOTS_DIR="$2"; shift 2 ;;
    --tail-out) TAIL_CSV="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# cd to repo root (parent of this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
echo "Repo root: $ROOT"

# Activate Linux venv if present
if [[ -f .venv_cli/bin/activate ]]; then
  echo "Activando .venv_cli..."
  # shellcheck disable=SC1091
  source .venv_cli/bin/activate
fi

echo "[1/5] Unificando maestros"
python3 scripts/merge_tournaments_to_masters.py

echo "[2/5] Generando join por match_id"
python3 scripts/join_matches_by_match_id.py

echo "[3/5] Entrenando modelo"
python3 -m mvp_model.train_mvp --csv-path "$CSV_PATH" --model-out "$MODEL_PATH" --metrics-out "$METRICS_PATH" --train-info-out "$TRAIN_INFO_PATH"

echo "[4/5] Exportando bloque de test a CSV"
TAIL_ARGS=(--csv-path "$CSV_PATH" --model "$MODEL_PATH" --out "$TAIL_CSV" --threshold "$THRESH")
if [[ "$ALL_TEST" == true ]]; then
  TAIL_ARGS+=(--all-test)
else
  TAIL_ARGS+=(--last-n "$LAST_N")
fi
python3 -m mvp_model.print_test_tail "${TAIL_ARGS[@]}"

echo "[5/5] Generando gráficas"
PLOT_ARGS=(--csv-path "$CSV_PATH" --model "$MODEL_PATH" --out-dir "$PLOTS_DIR" --test-size 0.2)
if [[ "$ALL_TEST" == true ]]; then
  PLOT_ARGS+=(--all-test)
else
  PLOT_ARGS+=(--last-n "$LAST_N")
fi
python3 -m mvp_model.plot_test_predictions "${PLOT_ARGS[@]}"

echo
echo "Listo. Salidas principales:"
echo " - Modelo: $MODEL_PATH"
echo " - Métricas (test completo): $METRICS_PATH"
echo " - Tail CSV (últimos $LAST_N): $TAIL_CSV"
echo " - Gráficas: $PLOTS_DIR/test_predictions_timeseries.png, $PLOTS_DIR/test_calibration_curve.png"
