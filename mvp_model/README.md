## MVP de predicción de resultados (matches.csv)

Resumen
- Entrena un modelo de clasificación (XGBoost si está disponible; fallback a Regresión Logística) para predecir si `team1` gana un partido.
- Usa exclusivamente `masters_csvs/matches.csv` y crea la etiqueta binaria `team1_win` a partir de `winner`.
- Features: diferencia de Elo prepartido (`elo_diff`) calculado de manera cronológica.

Requisitos
- Python 3.9+.
- Paquetes: `pandas`, `numpy`, `scikit-learn`. Opcional: `xgboost`.

Instalación de dependencias (opcional XGBoost)
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

Últimos N del test (Windows/PowerShell y Linux)
```bash
# Ejecutar desde la raíz del proyecto
python -m mvp_model.print_test_tail \
  --csv-path masters_csvs/matches.csv \
  --model mvp_model/artifacts/model.pkl \
  --last-n 10 \
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
