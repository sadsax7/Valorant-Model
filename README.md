# Data Valorant – Consolidación de CSVs y MVP de Modelo
**Autoría del Proyecto: Alejandro Arango Mejía y Thomas Rivera Fernandez.**

Proyecto para consolidar datos de torneos de Valorant (dump por torneo en carpetas `*_csvs/`) hacia un conjunto maestro (`masters_csvs/`) y entrenar un MVP de modelo para predecir ganadores usando Elo (pre‑partido). Incluye scripts para visualización, export de predicciones y orquestadores para ejecutar todo con un solo comando.

## Estructura del Repo
- `scripts/`
  - `merge_tournaments_to_masters.py`: une CSVs por torneo en `masters_csvs/` agregando `tournament_name` y unificando columnas.
  - `join_matches_by_match_id.py`: hace join por `match_id` entre `matches.csv`, `detailed_matches_overview.csv`, `detailed_matches_player_stats.csv` y `detailed_matches_maps.csv`, produciendo `masters_csvs/matches_joined.csv` con columnas `ov_*` y dos columnas JSON (`players_json`, `maps_json`).
- `tournaments/` (recomendado): carpeta donde viven todas las carpetas crudas `*_csvs/` de torneos.
- `masters_csvs/`: CSVs maestros consolidados (salida de los scripts, se mantienen en la raíz del repo).
- `mvp_model/`: MVP del modelo (entrenamiento, predicción, utilidades Elo y artifacts).
  - `train_mvp.py`, `predict_mvp.py`, `plot_test_predictions.py`, `print_test_tail.py`, `print_test_all.py`, `utils/elo.py`, `artifacts/`, `README.md`.
- `.venv/` (Windows) o `.venv_cli/` (Linux/WSL, opcional): entornos virtuales.
- `.gitignore`: ignora caches, entornos, artefactos y temporales.

## Requisitos
- Python 3.9+ (recomendado 3.11 o 3.12).
- Paquetes del modelo (si vas a entrenar/predicción): `pandas`, `numpy`, `scikit-learn`, `joblib`, `matplotlib` (opcional `xgboost`).
  - Ya listados en `mvp_model/requirements.txt`.

## Flujo de Datos
1) Entrada: carpetas `*_csvs/` por torneo con archivos como `matches.csv`, `detailed_matches_overview.csv`, etc.
2) Consolidación: `scripts/merge_tournaments_to_masters.py` crea `masters_csvs/*.csv` unificando columnas y añadiendo `tournament_name`.
3) Join por partido: `scripts/join_matches_by_match_id.py` crea `masters_csvs/matches_joined.csv` con overview y listas JSON de jugadores y mapas por `match_id`.
4) Modelo: `mvp_model/train_mvp.py` entrena un clasificador para `team1_win` usando `elo_diff` generado cronológicamente.

## Uso Rápido
Desde la raíz del repo (este folder):

0. Entorno y dependencias
```powershell
py -3.12 -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r mvp_model/requirements.txt
```

1. Generar maestros
```bash
python scripts/merge_tournaments_to_masters.py
# recomendado: si cambiaste la ubicación de los dumps
# python scripts/merge_tournaments_to_masters.py --data-root /ruta/a/mis/tournaments --output-dir /ruta/a/masters_csvs
```

2. Generar join de partidos
```bash
python scripts/join_matches_by_match_id.py
# recomendado: especificar ubicación de masters distinta
# python scripts/join_matches_by_match_id.py --masters-dir /ruta/a/mis/masters_csvs
```

3. Entrenar modelo y generar métricas/artefactos

Windows (PowerShell)
```powershell
# Crear/activar entorno (si no existe)
py -3.12 -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r mvp_model/requirements.txt

# Entrenar
python -m mvp_model.train_mvp \
  --csv-path masters_csvs/matches.csv \
  --model-out mvp_model/artifacts/model.pkl \
  --metrics-out mvp_model/artifacts/metrics.json \
  --train-info-out mvp_model/artifacts/train_info.json

# Predecir sobre un CSV
python -m mvp_model.predict_mvp \
  --model mvp_model/artifacts/model.pkl \
  --csv masters_csvs/matches.csv \
  --out mvp_model/artifacts/preds_sample.csv
```

Linux / WSL
```bash
# Crear/activar entorno (si lo prefieres separado del de Windows)
python3 -m venv .venv_cli
source .venv_cli/bin/activate
pip install -r mvp_model/requirements.txt

# Entrenar
python -m mvp_model.train_mvp \
  --csv-path masters_csvs/matches.csv \
  --model-out mvp_model/artifacts/model.pkl \
  --metrics-out mvp_model/artifacts/metrics.json \
  --train-info-out mvp_model/artifacts/train_info.json

# Predecir
python -m mvp_model.predict_mvp \
  --model mvp_model/artifacts/model.pkl \
  --csv masters_csvs/matches.csv \
  --out mvp_model/artifacts/preds_sample.csv
```

4. Utilidades de test
```bash
# CSV del bloque de test (todo el test recomendado)
python -m mvp_model.print_test_tail \
  --csv-path masters_csvs/matches.csv \
  --model mvp_model/artifacts/model.pkl \
  --all-test \
  --out mvp_model/artifacts/test_tail_preds.csv \
  --threshold 0.5

# Todo el bloque de test a CSV
python -m mvp_model.print_test_all \
  --csv-path masters_csvs/matches.csv \
  --model mvp_model/artifacts/model.pkl \
  --out mvp_model/artifacts/test_preds.csv
```

5. Gráficas del test (serie temporal + calibración)
```bash
python -m mvp_model.plot_test_predictions \
  --csv-path masters_csvs/matches.csv \
  --model mvp_model/artifacts/model.pkl \
  --out-dir mvp_model/artifacts/plots \
  --test-size 0.2 \
  --all-test \
  --threshold 0.5
```

## Detalles de Implementación
- `merge_tournaments_to_masters.py`
  - Detecta la raíz del proyecto automáticamente y busca todas las carpetas `*_csvs/` (excepto `masters_csvs`).
  - Para cada base (p. ej. `matches`, `player_stats`, …) genera una cabecera unión para no perder columnas cuando los torneos difieren.
  - Escribe en `masters_csvs/{base}.csv` con reemplazo atómico (`.tmp_*.csv`).

- `join_matches_by_match_id.py`
  - Une por `match_id` y crea columnas `ov_*` del overview y dos columnas JSON: `players_json` y `maps_json` (sin `match_id` para no duplicar).
  - Salida: `masters_csvs/matches_joined.csv`.

- `mvp_model/train_mvp.py`
  - Crea etiquetas `team1_win` a partir de `winner`.
  - Genera Elo cronológico (`elo_diff`) con K y base configurables.
  - Entrena un pipeline: `StandardScaler` + `LogisticRegression` (o `XGBClassifier` si activado y disponible).
  - Métricas: LogLoss, ROC-AUC, Brier en split temporal (cola como test).
  - Artefactos: `model.pkl`, `metrics.json`, `train_info.json` en `mvp_model/artifacts/`.

## Buenas Prácticas y Notas
- Ejecuta los scripts desde la raíz o desde `scripts/` indistintamente: detectan la raíz del proyecto.
- Codificación CSV: se usa `utf-8-sig` para tolerar BOM.
- `.gitignore` ya ignora `mvp_model/artifacts/` y temporales `.tmp_*.csv`.
  - Si no quieres subir los dumps crudos, puedes ignorar `*_csvs/` manteniendo `masters_csvs/` (ver bloque comentado en `.gitignore`).
- En Windows y WSL, los entornos virtuales no son intercambiables (Windows usa `Scripts/`, Linux `bin/`). Crea un entorno por sistema si alternas.
- Si tu Python en Linux no tiene `venv` (error `ensurepip is not available`), instala el paquete de sistema (p. ej. `apt install python3.12-venv`) y recrea el venv.

## Problemas Comunes (Troubleshooting)
- `ModuleNotFoundError` (p. ej. `numpy`): activa el entorno correcto antes de ejecutar (`.venv/Scripts/Activate.ps1` en Windows; `source .venv_cli/bin/activate` en Linux/WSL) o instala dependencias.
- `PermissionError` al activar en PowerShell: permite scripts con `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y vuelve a activar.
- CSVs vacíos o faltantes en algún torneo: el merge los omite; revisa el resumen por archivo para detectar ausentes.
- PowerShell partiendo líneas: pega los comandos en una sola línea o usa backticks (`) al final de cada línea. Evita partir rutas en medio (p. ej., `mvp_model/artifacts/` + `model.pkl`).

## Siguientes Pasos
- Añadir más features pre-partido (ratings por mapa, forma reciente por jugador/agente, contexto de patch/torneo) manteniendo splits temporales.
- Agregar validación cruzada temporal y tracking de experimentos.

---
Para más detalles del modelo, ver `mvp_model/README.md`.

## Ejecución de todo el pipeline (one‑liner)
- Windows (PowerShell):
  - `powershell -ExecutionPolicy Bypass -File scripts/run_all.ps1`
  - Por defecto usa TODO el bloque de test para las gráficas; para limitar a N: `-LastN 10`
  - Otros parámetros: `-Threshold 0.5 -CsvPath masters_csvs/matches.csv -ModelPath mvp_model/artifacts/model.pkl`

- Linux/WSL:
  - `bash scripts/run_all.sh`
  - Por defecto usa TODO el bloque de test; para limitar a N: `bash scripts/run_all.sh --last-n 10`
  - Otros parámetros: `--threshold 0.5 --csv-path masters_csvs/matches.csv --model mvp_model/artifacts/model.pkl`

## Salidas esperadas y verificación rápida
- Salidas clave (rutas por defecto):
  - Modelo: `mvp_model/artifacts/model.pkl`
  - Métricas test (probabilísticas): `mvp_model/artifacts/metrics.json`
  - CSV de test (predicciones + aciertos): `mvp_model/artifacts/test_tail_preds.csv`
  - Gráficas: `mvp_model/artifacts/plots/test_predictions_timeseries.png` y `.../test_calibration_curve.png`
  - Métricas de gráficas (incluye discretas y matriz de confusión): `mvp_model/artifacts/plots/test_metrics.json`

- Verificar en PowerShell:
  - `Test-Path "mvp_model/artifacts/model.pkl"` (debe ser True)
  - `Get-Content "mvp_model/artifacts/metrics.json"`
  - `Get-Content "mvp_model/artifacts/plots/test_metrics.json"`
  - Abrir imágenes: `Start-Process "mvp_model/artifacts/plots/test_predictions_timeseries.png"`

- Verificar en Linux/WSL:
  - `ls -lh mvp_model/artifacts/model.pkl`
  - `sed -n '1,200p' mvp_model/artifacts/metrics.json`
  - `sed -n '1,200p' mvp_model/artifacts/plots/test_metrics.json`

## Cómo interpretar resultados
- `roc_auc` (≈0.5 azar, >0.6 aceptable): mide discriminación (qué tanto ordena ganadores por encima de perdedores).
- `log_loss` y `brier` (más bajos son mejores): calidad/calibración de las probabilidades.
- Gráficas:
  - Serie temporal: la línea de probabilidad debe estar alta cuando el punto real es 1 y baja cuando es 0.
  - Calibración: la curva ideal se pega a la diagonal; por debajo = sobreconfianza, por encima = subconfianza.
- Métricas discretas en `test_metrics.json.discrete` (umbral configurable `--threshold`): TP/TN/FP/FN, accuracy, precision, recall, F1.

## Contactos:
- **aarangom1@eafit.edu.co**
- **triveraf@eafit.edu.co**
