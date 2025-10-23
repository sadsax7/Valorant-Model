MVP de predicción de resultados (matches.csv)

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
  --metrics-out mvp_model/artifacts/metrics.json
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
  --out mvp_model/artifacts/preds_sample.csv
```

Últimos 10 del test (Windows/PowerShell y Linux)
```bash
# Ejecutar desde la raíz del proyecto
python -m mvp_model.print_test_tail \
  --csv-path masters_csvs/matches.csv \
  --model mvp_model/artifacts/model.pkl
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
