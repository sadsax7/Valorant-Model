# MVP Model - Predicción de Resultados de Partidos de Valorant

Este módulo contiene el modelo de predicción de resultados de partidos profesionales de Valorant.

## 🎯 Descripción

El modelo predice la probabilidad de que el equipo 1 gane un partido utilizando:
- **Sistema Elo** para ratings de equipos
- **Métricas de jugadores** agregadas por equipo (ACS, KAST, ADR, HS%)
- **Regresión Logística** como algoritmo de clasificación

## 📊 Features del Modelo (7 total)

1. **`elo1_before`** - Rating Elo del equipo 1 antes del partido
2. **`elo2_before`** - Rating Elo del equipo 2 antes del partido
3. **`elo_diff`** - Diferencia de Elo (team1 - team2)
4. **`diff_acs_mean`** - Diferencia de ACS promedio
5. **`diff_kast_mean`** - Diferencia de KAST promedio
6. **`diff_adr_mean`** - Diferencia de ADR promedio (Average Damage per Round)
7. **`diff_hs_percent_mean`** - Diferencia de HS% promedio (Headshot Percentage)

## 📁 Estructura

```
mvp_model/
├── train_mvp.py           # Entrenamiento del modelo
├── predict_mvp.py         # Predicciones con modelo entrenado
├── analyze_test.py        # Análisis unificado (CSV + gráficas + métricas)
├── utils/                 # Módulos de utilidades
│   ├── __init__.py       # Exports del módulo
│   ├── elo.py            # Sistema de rating Elo
│   ├── features.py       # Construcción de features
│   ├── data_loader.py    # Carga y preparación de datos
│   ├── model_utils.py    # Utilidades de modelo
│   └── cli_args.py       # Argumentos CLI reutilizables
├── artifacts/            # Modelos y resultados
│   ├── model.pkl         # Modelo entrenado
│   ├── metrics.json      # Métricas del modelo
│   ├── train_info.json   # Información de entrenamiento
│   └── plots/            # Gráficas de análisis
├── MODEL_WORKFLOW.md     # Documentación completa del flujo
└── README.md             # Este archivo
```

## 🚀 Uso

### Instalación de Dependencias

```bash
pip install -r mvp_model/requirements.txt
```

Paquetes: `pandas`, `numpy`, `scikit-learn`, `joblib`, `matplotlib`. Opcional: `xgboost`.

### 1. Entrenamiento

```bash
python -m mvp_model.train_mvp \
  --csv-path masters_csvs/matches.csv \
  --players-stats-path masters_csvs/detailed_matches_player_stats.csv \
  --model-out mvp_model/artifacts/model.pkl \
  --metrics-out mvp_model/artifacts/metrics.json \
  --train-info-out mvp_model/artifacts/train_info.json
```

**Opciones adicionales:**
- `--use-xgb`: Usar XGBoost en lugar de Regresión Logística
- `--elo-k 32`: K-factor del sistema Elo (default: 32)
- `--elo-base 1500`: Rating base del sistema Elo (default: 1500)
- `--test-size 0.2`: Fracción del dataset para test (default: 0.2)

**Salidas:**
- `model.pkl`: Pipeline entrenado (Imputer + Scaler + Modelo)
- `metrics.json`: Métricas del test (Log Loss, ROC-AUC, Brier)
- `train_info.json`: Metadatos del entrenamiento

### 2. Predicción

```bash
python -m mvp_model.predict_mvp \
  --model mvp_model/artifacts/model.pkl \
  --csv masters_csvs/matches.csv \
  --out predictions.csv
```

**Salida:** CSV con columnas `match_id`, `team1`, `team2`, `p_team1_win`

### 3. Análisis del Test (Nuevo Script Unificado)

El script `analyze_test.py` reemplaza los antiguos `print_test_all.py`, `print_test_tail.py` y `plot_test_predictions.py`.

#### Exportar predicciones a CSV

```bash
# Todo el test
python -m mvp_model.analyze_test --all-test --out test_results.csv

# Últimos 10 partidos
python -m mvp_model.analyze_test --last-n 10 --out last_10.csv
```

#### Imprimir resultados en consola

```bash
python -m mvp_model.analyze_test --all-test --print
```

#### Generar gráficas

```bash
python -m mvp_model.analyze_test --all-test --plot --plot-dir mvp_model/artifacts/plots
```

#### Todo junto

```bash
python -m mvp_model.analyze_test \
  --all-test \
  --print \
  --plot \
  --out test_results.csv \
  --threshold 0.5
```

**Gráficas generadas:**
- `test_predictions_timeseries.png`: Serie temporal de predicciones
- `test_calibration_curve.png`: Curva de calibración
- `test_metrics.json`: Métricas completas (probabilísticas + discretas)

## 📈 Resultados Actuales

**Modelo con 7 features (Elo + ACS + KAST + ADR + HS%)**:

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Accuracy** | 85.11% | 160 de 188 partidos correctos |
| **ROC-AUC** | 0.9469 | Excelente discriminación |
| **Log Loss** | 0.2855 | Excelente calibración |
| **Brier Score** | 0.0921 | Predicciones precisas |
| **Precision** | 85.19% | Pocas falsas alarmas |
| **Recall** | 88.46% | Detecta mayoría de victorias |
| **F1 Score** | 86.79% | Balance excelente |

## 📝 Formato de Salida

### CSV de Predicciones

```csv
match_id,team1,team2,p_team1_win,pred_team1_win,team1_win,correct
530935,Team Liquid,GIANTX,0.971562,1,1,True
542279,FNATIC,DRX,0.975082,1,1,True
```

**Columnas:**
- `p_team1_win`: Probabilidad predicha (0-1)
- `pred_team1_win`: Predicción discreta (0 o 1) con threshold
- `team1_win`: Resultado real (0 o 1)
- `correct`: Si la predicción fue correcta

## 🔧 Notas Técnicas

- **Split temporal**: 80% entrenamiento, 20% test (orden cronológico)
- **Sin fuga de información**: Solo usa datos disponibles antes del partido
- **Sistema Elo**: Actualizado cronológicamente, ratings previos al partido
- **Métricas de jugadores**: Agregadas por equipo (promedio de 5 jugadores)
- **Pipeline**: Imputer (mediana) → StandardScaler → LogisticRegression

## 📚 Documentación Adicional

- **[MODEL_WORKFLOW.md](MODEL_WORKFLOW.md)**: Guía completa del flujo del modelo
- **[../README.md](../README.md)**: Documentación principal del proyecto

## 🎯 Próximos Pasos

Posibles mejoras futuras:
- Agregar más métricas de jugadores (FK, FD, K/D ratio)
- Ratings por mapa específico
- Forma reciente del equipo (últimos N partidos)
- Contexto de torneo/patch
- Validación cruzada temporal
