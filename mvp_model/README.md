## MVP de predicción de resultados (matches.csv)

Resumen
- Entrena un modelo de clasificación (XGBoost si está disponible; fallback a Regresión Logística) para predecir si `team1` gana un partido.
- Usa exclusivamente `masters_csvs/matches.csv` y crea la etiqueta binaria `team1_win` a partir de `winner`.
- Features: diferencia de Elo prepartido (`elo_diff`) calculado de manera cronológica.

Requisitos
- Python 3.9+.
- Paquetes: `pandas`, `numpy`, `scikit-learn`, `joblib`, `matplotlib`. Opcional: `xgboost`.

Instalación de dependencias (incluye matplotlib; XGBoost es opcional)
```bash
pip install -r mvp_model/requirements.txt
```

Entrenamiento
```bash
python mvp_model/train_mvp.py \
  --csv-path masters_csvs/matches.csv \
  --model-out mvp_model/artifacts/model.pkl \
  --metrics-out mvp_model/artifacts/metrics.json \
  --train-info-out mvp_model/artifacts/train_info.json \
  # Opcional (si instalaste xgboost):
  # --use-xgb \
  # Opcional (debe coincidir con predicción si lo cambias):
  # --elo-k 32 --elo-base 1500
```

Salidas
- `mvp_model/artifacts/model.pkl`: Pipeline entrenado (XGBoost o Regresión Logística).
- `mvp_model/artifacts/metrics.json`: Métricas en el split temporal (LogLoss, ROC-AUC, Brier).
- `mvp_model/artifacts/train_info.json`: Metadatos (fecha de entrenamiento, n muestras, parámetros Elo, columnas).

Predicción (opcional)
```bash
python mvp_model/predict_mvp.py \
  --model mvp_model/artifacts/model.pkl \
  --csv masters_csvs/matches.csv \
  --out mvp_model/artifacts/preds_sample.csv \
  # Si cambiaste parámetros de Elo al entrenar, usa los mismos aquí:
  # --elo-k 32 --elo-base 1500
```

Gráficas (test)
```bash
# Requiere matplotlib (incluido en requirements)
# Todo el bloque de test (recomendado para evaluación estable):
python -m mvp_model.plot_test_predictions \
  --csv-path masters_csvs/matches.csv \
  --model mvp_model/artifacts/model.pkl \
  --out-dir mvp_model/artifacts/plots \
  --test-size 0.2 \
  --all-test \
  --threshold 0.5

# Solo los últimos N del test (para ver tendencia reciente):
python -m mvp_model.plot_test_predictions \
  --csv-path masters_csvs/matches.csv \
  --model mvp_model/artifacts/model.pkl \
  --out-dir mvp_model/artifacts/plots \
  --test-size 0.2 \
  --last-n 100 \
  --threshold 0.5
```
Salidas:
- `mvp_model/artifacts/plots/test_predictions_timeseries.png`: serie temporal con probabilidad predicha y resultado real (0/1).
- `mvp_model/artifacts/plots/test_calibration_curve.png`: curva de calibración (con 10 bins por cuantiles).
- `mvp_model/artifacts/plots/test_metrics.json`: resumen de métricas del bloque de test, incluyendo métricas discretas (accuracy, precision, recall, F1) y TP/TN/FP/FN al umbral indicado.

Formato del CSV de test (test_tail_preds.csv)
```text
parsed_date,match_id,team1,team2,p_team1_win,pred_team1_win,team1_win,correct
2025-09-29,542277,Paper Rex,Team Heretics,0.6290,1,1,True
...
```
- `p_team1_win`: probabilidad predicha para que `team1` gane.
- `pred_team1_win`: etiqueta 0/1 usando el umbral (`--threshold`, por defecto 0.5).
- `team1_win`: resultado real 0/1.
- `correct`: True si `pred_team1_win == team1_win`.

Ejemplos de verificación (PowerShell)
```powershell
$rows = Import-Csv "mvp_model/artifacts/test_tail_preds.csv"
$rows.Count
$ok = ($rows | ? { $_.correct -eq 'True' }).Count
"Aciertos: $ok de $($rows.Count) = $([math]::Round($ok/$rows.Count,4))"
```

Ejemplos de verificación (Linux/WSL)
```bash
wc -l mvp_model/artifacts/test_tail_preds.csv   # incluye cabecera
head -n 5 mvp_model/artifacts/test_tail_preds.csv
```

Últimos N del test (Windows/PowerShell y Linux)
```bash
# Ejecutar desde la raíz del proyecto
python -m mvp_model.print_test_tail \
  --csv-path masters_csvs/matches.csv \
  --model mvp_model/artifacts/model.pkl \
  --all-test \
  # o limitar a N recientes: \
  # --last-n 10 \
  --out mvp_model/artifacts/test_tail_preds.csv \
  --threshold 0.5
# Para exportar TODO el bloque de test en este formato:
# python -m mvp_model.print_test_tail --csv-path masters_csvs/matches.csv --model mvp_model/artifacts/model.pkl --all-test --out mvp_model/artifacts/test_tail_preds.csv --threshold 0.5
```

Todo el bloque de test a CSV
```bash
python -m mvp_model.print_test_all \
  --csv-path masters_csvs/matches.csv \
  --model mvp_model/artifacts/model.pkl \
  --out mvp_model/artifacts/test_preds.csv
```

Notas
- Este MVP usa solo `elo_diff`. Es intencional para evitar fugas de información usando campos post-partido.
- Próximos pasos: añadir más features prepartido (ratings por mapa, forma reciente por jugador/agente, contexto de patch/torneo) manteniendo splits temporales.
- Requisitos de columnas mínimas en `masters_csvs/matches.csv`: `date` (recomendado, para ordenar), `team1`, `team2`, `winner`, `status` (para filtrar a `Completed`).
 - Si `parsed_date` no se puede parsear para alguna fila, aparecerá `NaT` (no afecta el cálculo de Elo ni la predicción).
