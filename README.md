# Data Valorant – Consolidación de CSVs y MVP de Modelo
**Autoria del Proyecto: Alejandro Arango Mejía y Thomas Rivera Fernandez.**

Proyecto para consolidar datos de torneos de Valorant (dump por torneo en carpetas `*_csvs/`) hacia un conjunto maestro (`masters_csvs/`) y, opcionalmente, entrenar un MVP de modelo de predicción de resultados de partidos (basado en Elo y un clasificador sencillo).

## Estructura del Repo
- `scripts/`
  - `merge_tournaments_to_masters.py`: une CSVs por torneo en `masters_csvs/` agregando `tournament_name` y unificando columnas.
  - `join_matches_by_match_id.py`: hace join por `match_id` entre `matches.csv`, `detailed_matches_overview.csv`, `detailed_matches_player_stats.csv` y `detailed_matches_maps.csv`, produciendo `masters_csvs/matches_joined.csv` con columnas `ov_*` y dos columnas JSON (`players_json`, `maps_json`).
- `masters_csvs/`: CSVs maestros consolidados (salida de los scripts de `scripts/`).
- `VCT 2025 ... _csvs/`, `Valorant ... _csvs/`: dumps crudos por torneo (entrada).
- `mvp_model/`: MVP del modelo (entrenamiento, predicción, utilidades Elo y artifacts).
  - `train_mvp.py`, `predict_mvp.py`, `utils/elo.py`, `artifacts/`, `README.md`.
- `.venv/` (Windows) o `.venv_cli/` (Linux/WSL, opcional): entornos virtuales.
- `.gitignore`: ignora caches, entornos, artefactos y temporales.

## Requisitos
- Python 3.9+ (recomendado 3.11 o 3.12).
- Paquetes del modelo (si vas a entrenar/predicción): `pandas`, `numpy`, `scikit-learn`, `joblib` (opcional `xgboost`).
  - Ya listados en `mvp_model/requirements.txt`.

## Flujo de Datos
1) Entrada: carpetas `*_csvs/` por torneo con archivos como `matches.csv`, `detailed_matches_overview.csv`, etc.
2) Consolidación: `scripts/merge_tournaments_to_masters.py` crea `masters_csvs/*.csv` unificando columnas y añadiendo `tournament_name`.
3) Join por partido: `scripts/join_matches_by_match_id.py` crea `masters_csvs/matches_joined.csv` con overview y listas JSON de jugadores y mapas por `match_id`.
4) Modelo (opcional): `mvp_model/train_mvp.py` entrena un clasificador para `team1_win` usando `elo_diff` generado cronológicamente.

## Uso Rápido
Desde la raíz del repo (este folder):

1. Generar maestros
```bash
python scripts/merge_tournaments_to_masters.py
```

2. Generar join de partidos
```bash
python scripts/join_matches_by_match_id.py
```

3. (Opcional) Entrenar modelo y generar métricas/artefactos

Windows (PowerShell)
```powershell
# Crear/activar entorno (si no existe)
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r mvp_model/requirements.txt

# Entrenar
python -m mvp_model.train_mvp \
  --csv-path masters_csvs/matches.csv \
  --model-out mvp_model/artifacts/model.pkl \
  --metrics-out mvp_model/artifacts/metrics.json \
  --train-info-out mvp_model/artifacts/train_info.json

# (Opcional) Predecir sobre un CSV
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

# (Opcional) Predecir
python -m mvp_model.predict_mvp \
  --model mvp_model/artifacts/model.pkl \
  --csv masters_csvs/matches.csv \
  --out mvp_model/artifacts/preds_sample.csv
```

4. (Opcional) Utilidades de test
```bash
# Últimos N del test (ver README de mvp_model)
python -m mvp_model.print_test_tail \
  --csv-path masters_csvs/matches.csv \
  --model mvp_model/artifacts/model.pkl

# Todo el bloque de test a CSV
python -m mvp_model.print_test_all \
  --csv-path masters_csvs/matches.csv \
  --model mvp_model/artifacts/model.pkl \
  --out mvp_model/artifacts/test_preds.csv
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
  - Si no quieres subir los dumps crudos, puedes ignorar `*_csvs/` manteniendo `masters_csvs/` (ver bloque opcional comentado en `.gitignore`).
- En Windows y WSL, los entornos virtuales no son intercambiables (Windows usa `Scripts/`, Linux `bin/`). Crea un entorno por sistema si alternas.
- Si tu Python en Linux no tiene `venv` (error `ensurepip is not available`), instala el paquete de sistema (p. ej. `apt install python3.12-venv`) y recrea el venv.

## Problemas Comunes (Troubleshooting)
- `ModuleNotFoundError` (p. ej. `numpy`): activa el entorno correcto antes de ejecutar (`.venv/Scripts/Activate.ps1` en Windows; `source .venv_cli/bin/activate` en Linux/WSL) o instala dependencias.
- `PermissionError` al activar en PowerShell: permite scripts con `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y vuelve a activar.
- CSVs vacíos o faltantes en algún torneo: el merge los omite; revisa el resumen por archivo para detectar ausentes.

## Siguientes Pasos
- Añadir más features pre-partido (ratings por mapa, forma reciente por jugador/agente, contexto de patch/torneo) manteniendo splits temporales.
- Agregar validación cruzada temporal y tracking de experimentos.

---
Para más detalles del modelo, ver `mvp_model/README.md`.
