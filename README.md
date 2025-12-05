# Modelo Predictivo de Valorant Competitivo
**Autoría del Proyecto: Alejandro Arango Mejía y Thomas Rivera Fernandez.**

Modelo de machine learning para predecir resultados de partidos profesionales de Valorant usando **Regresión Logística** con sistema de rating **Elo** y métricas de rendimiento de jugadores. El modelo alcanza un **86% de precisión** en el conjunto de prueba.

## 📊 Sobre los Datos

Recolectamos información detallada de **torneos competitivos de Valorant de 2024 y 2025**. La decisión de enfocarnos en estas temporadas es estratégica: queremos evitar sesgar el modelo con datos históricos que reflejen cómo jugaban los equipos en metas anteriores del juego, cuando las estrategias, composiciones y el nivel competitivo eran diferentes.

**Planeamos agregar datos de temporadas anteriores** en el futuro, pero actualmente priorizamos la precisión predictiva sobre equipos y jugadores en su forma actual.

Los datos incluyen:
- Información de partidos (equipos, ganador, fecha, torneo)
- Estadísticas detalladas de jugadores por partido
- Métricas de rendimiento por mapa

## 🎯 Metodología del Modelo

### Sistema de Rating Elo

Implementamos un **sistema de rating Elo** adaptado para Valorant competitivo. El Elo es un método de clasificación que:

- **Asigna un rating inicial** de 1500 puntos a cada equipo
- **Actualiza ratings después de cada partido** basándose en el resultado esperado vs. el resultado real
- **Usa un factor K de 32** para controlar la velocidad de ajuste de ratings
- **Calcula probabilidades pre-partido** usando la diferencia de Elo entre equipos

La fórmula de probabilidad esperada es:
```
P(A gana) = 1 / (1 + 10^((Elo_B - Elo_A) / 400))
```

**Importante:** El Elo se calcula de forma **cronológica** (ordenando partidos por fecha) para evitar fugas de información. Solo usamos el rating de cada equipo **antes** del partido para hacer predicciones.

### Métricas de Jugadores

Además del Elo de equipo, el modelo incorpora métricas agregadas de rendimiento de jugadores:

#### **ACS (Average Combat Score)**
- Métrica compuesta que mide el impacto general de un jugador en el combate
- Considera daño infligido, kills, multi-kills y otros factores
- **Calculamos el promedio de ACS** de los 5 jugadores de cada equipo por partido
- Usamos la **diferencia de ACS promedio** entre equipos como feature

#### **KAST (Kill, Assist, Survive, Trade)**
- Porcentaje de rounds donde el jugador:
  - Consiguió un kill, O
  - Dio una asistencia, O
  - Sobrevivió el round, O
  - Fue tradeado (su muerte fue vengada inmediatamente)
- Mide la **consistencia y participación** del jugador en cada round
- **Calculamos el promedio de KAST** de los 5 jugadores de cada equipo
- Usamos la **diferencia de KAST promedio** entre equipos como feature

#### **ADR (Average Damage per Round)**
- Daño promedio infligido por el jugador en cada round
- Mide el **impacto ofensivo** del jugador
- **Calculamos el promedio de ADR** de los 5 jugadores de cada equipo por partido
- Usamos la **diferencia de ADR promedio** entre equipos como feature
- Complementa ACS al enfocarse específicamente en el daño

#### **HS% (Headshot Percentage)**
- Porcentaje de kills que son headshots
- Mide la **precisión y habilidad mecánica** del jugador
- Headshots hacen más daño y son más letales
- **Calculamos el promedio de HS%** de los 5 jugadores de cada equipo
- Usamos la **diferencia de HS% promedio** entre equipos como feature

### Features del Modelo

El modelo usa las siguientes características para cada partido:

1. **`elo1_before`**: Rating Elo del equipo 1 antes del partido
2. **`elo2_before`**: Rating Elo del equipo 2 antes del partido
3. **`elo_diff`**: Diferencia de Elo (equipo1 - equipo2)
4. **`diff_acs_mean`**: Diferencia de ACS promedio entre equipos
5. **`diff_kast_mean`**: Diferencia de KAST promedio entre equipos
6. **`diff_adr_mean`**: Diferencia de ADR (Average Damage per Round) promedio entre equipos
7. **`diff_hs_percent_mean`**: Diferencia de porcentaje de headshots promedio entre equipos

### Algoritmo: Regresión Logística

Usamos **Regresión Logística** con las siguientes características:

- **Variable objetivo dicotómica**: 1 si gana el equipo 1, 0 si gana el equipo 2
- **Pipeline de preprocesamiento**:
  - Imputación de valores faltantes (mediana)
  - Estandarización de features (StandardScaler)
- **Split temporal**: 80% entrenamiento, 20% prueba (respetando orden cronológico)

## 📊 Resultados del Modelo

### Evolución de Entrenamientos

A continuación se muestra la evolución del modelo a través de diferentes experimentos, agregando más datos y features:

| Entrenamiento | Datos Usados | Features Principales | Modelo | Log Loss | ROC-AUC | Brier | Accuracy | Precision | Recall | F1 |
|---------------|--------------|---------------------|---------|----------|---------|-------|----------|-----------|--------|-----|
| **1** | Torneos 2025<br>(504 partidos) | Solo Elo<br>(`elo1_before`, `elo2_before`, `elo_diff`) | Logistic Regression | 0.6763 | 0.5980 | 0.2456 | 54.46% | 55.17% | 58.25% | 56.60% |
| **2** | Torneos 2024+2025<br>(940 partidos) | Solo Elo<br>(`elo1_before`, `elo2_before`, `elo_diff`) | Logistic Regression | 0.6581 | 0.6510 | 0.2398 | 61.17% | 62.50% | 65.00% | 63.68% |
| **3** | Torneos 2024+2025<br>(940 partidos) | Elo + ACS/KAST<br>(`elo_diff`, `diff_acs_mean`, `diff_kast_mean`) | Logistic Regression | 0.2922 | 0.9453 | 0.0935 | 86.17% | 87.10% | 87.63% | 87.32% |
| **4** | Torneos 2024+2025<br>(940 partidos) | Elo + ACS/KAST/ADR/HS%<br>(`elo_diff`, `diff_acs_mean`, `diff_kast_mean`, `diff_adr_mean`, `diff_hs_percent_mean`) | Logistic Regression | **0.2855** | **0.9469** | **0.0921** | **85.11%** | **85.19%** | **88.46%** | **86.79%** |

### Análisis de Resultados

**Del Entrenamiento 1 al 2:**
- Al casi **duplicar los datos** (de 504 a 940 partidos), el modelo mejoró significativamente
- ROC-AUC aumentó **+8.9%** (0.598 → 0.651)
- Accuracy mejoró **+6.7 puntos porcentuales** (54.46% → 61.17%)
- Esto demuestra que el sistema Elo se beneficia enormemente de tener más historial de partidos

**Del Entrenamiento 2 al 3:**
- Agregar métricas de jugadores (ACS y KAST) produjo una **mejora dramática**
- Log Loss se redujo **-0.37** (0.658 → 0.292) - mucho mejor calibración
- ROC-AUC saltó a **0.945** - discriminación casi perfecta
- Accuracy alcanzó **86.17%** - el modelo predice correctamente 86 de cada 100 partidos
- Esto confirma que las estadísticas individuales de jugadores agregadas por equipo aportan señal predictiva real

**Del Entrenamiento 3 al 4:**
- Agregar **ADR** (Average Damage per Round) y **HS%** (Headshot Percentage) mejoró la calibración
- Log Loss mejoró **-2.3%** (0.292 → 0.286) - mejor calibración de probabilidades
- ROC-AUC mejoró a **0.947** - mejor discriminación
- Brier Score mejoró **-1.5%** (0.0935 → 0.0921) - predicciones más precisas
- El modelo ahora considera el daño por round y la precisión de headshots, métricas clave del rendimiento individual

### Explicación de Métricas

#### **Accuracy (Exactitud)**
- **Qué mide**: Porcentaje de predicciones correctas sobre el total
- **Fórmula**: (Predicciones Correctas) / (Total de Predicciones)
- **Interpretación**:
  - **< 60%**: Rendimiento pobre, apenas mejor que el azar
  - **60-70%**: Aceptable, el modelo tiene cierta capacidad predictiva
  - **70-80%**: Bueno, el modelo es confiable
  - **80-90%**: Muy bueno, alto nivel de precisión
  - **> 90%**: Excelente (pero cuidado con overfitting)
- **Nuestro resultado**: **85.64%** ✅ Muy bueno

#### **ROC-AUC (Area Under the ROC Curve)**
- **Qué mide**: Capacidad del modelo para discriminar entre clases (victorias vs derrotas)
- **Rango**: 0.0 a 1.0
- **Interpretación**:
  - **0.5**: Modelo aleatorio, sin capacidad predictiva
  - **0.6-0.7**: Discriminación pobre
  - **0.7-0.8**: Discriminación aceptable
  - **0.8-0.9**: Discriminación buena
  - **0.9-1.0**: Discriminación excelente
- **Nuestro resultado**: **0.9453** ✅ Excelente

#### **Log Loss (Logarithmic Loss)**
- **Qué mide**: Penaliza predicciones con alta confianza que resultan incorrectas
- **Rango**: 0.0 a infinito (más bajo es mejor)
- **Interpretación**:
  - **< 0.3**: Excelente calibración de probabilidades
  - **0.3-0.5**: Buena calibración
  - **0.5-0.7**: Calibración aceptable
  - **> 0.7**: Calibración pobre
- **Nuestro resultado**: **0.2922** ✅ Excelente

#### **Brier Score**
- **Qué mide**: Error cuadrático medio de las probabilidades predichas
- **Rango**: 0.0 a 1.0 (más bajo es mejor)
- **Interpretación**:
  - **< 0.1**: Excelente calibración
  - **0.1-0.2**: Buena calibración
  - **0.2-0.25**: Calibración aceptable
  - **> 0.25**: Calibración pobre
- **Nuestro resultado**: **0.0856** ✅ Excelente

#### **Precision (Precisión)**
- **Qué mide**: De todas las predicciones positivas, cuántas fueron correctas
- **Fórmula**: Verdaderos Positivos / (Verdaderos Positivos + Falsos Positivos)
- **Interpretación**:
  - **< 70%**: Muchos falsos positivos
  - **70-80%**: Aceptable
  - **80-90%**: Buena
  - **> 90%**: Excelente
- **Nuestro resultado**: **87.10%** ✅ Buena

#### **Recall (Sensibilidad)**
- **Qué mide**: De todos los casos positivos reales, cuántos detectó el modelo
- **Fórmula**: Verdaderos Positivos / (Verdaderos Positivos + Falsos Negativos)
- **Interpretación**:
  - **< 70%**: El modelo pierde muchos casos positivos
  - **70-80%**: Aceptable
  - **80-90%**: Buena
  - **> 90%**: Excelente
- **Nuestro resultado**: **87.63%** ✅ Buena

#### **F1 Score**
- **Qué mide**: Media armónica entre Precision y Recall (balance entre ambas)
- **Fórmula**: 2 × (Precision × Recall) / (Precision + Recall)
- **Interpretación**:
  - **< 70%**: Rendimiento pobre
  - **70-80%**: Aceptable
  - **80-90%**: Bueno
  - **> 90%**: Excelente
- **Nuestro resultado**: **87.32%** ✅ Bueno

### Resumen de Rendimiento

Nuestro modelo actual (Entrenamiento 4) alcanza:
- ✅ **85.11% de accuracy** - Predice correctamente 85 de cada 100 partidos
- ✅ **ROC-AUC de 0.947** - Discriminación casi perfecta entre victorias y derrotas
- ✅ **Log Loss de 0.286** - Excelente calibración de probabilidades
- ✅ **Brier Score de 0.092** - Predicciones probabilísticas muy precisas
- ✅ **F1 Score de 86.79%** - Excelente balance entre precision y recall
- ✅ **7 features** - Elo + ACS + KAST + ADR + HS%

Todas las métricas están en rangos **excelentes**, lo que indica que el modelo es altamente confiable para predecir resultados de partidos profesionales de Valorant.

## 📁 Estructura del Proyecto

```
Valorant-Model/
├── masters_csvs/              # Datos consolidados de todos los torneos
│   ├── matches.csv           # Información de partidos
│   ├── detailed_matches_player_stats.csv  # Estadísticas de jugadores
│   └── ...
├── mvp_model/                # Código del modelo
│   ├── train_mvp.py         # Script de entrenamiento
│   ├── predict_mvp.py       # Script de predicción
│   ├── utils/
│   │   ├── elo.py          # Implementación del sistema Elo
│   │   └── features.py     # Construcción de features
│   ├── artifacts/          # Modelos y resultados guardados
│   └── requirements.txt    # Dependencias Python
├── scripts/                 # Scripts auxiliares
└── tournaments/            # Datos crudos por torneo
```

## 🚀 Cómo Ejecutar el Modelo

### 0. Configuración del Entorno

**Windows (PowerShell):**
```powershell
py -3.12 -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r mvp_model/requirements.txt
```

**Linux / WSL:**
```bash
python3 -m venv .venv_cli
source .venv_cli/bin/activate
pip install -r mvp_model/requirements.txt
```

### 1. Entrenar el Modelo

**Windows:**
```powershell
python -m mvp_model.train_mvp `
  --csv-path masters_csvs/matches.csv `
  --players-stats-path masters_csvs/detailed_matches_player_stats.csv `
  --model-out mvp_model/artifacts/model.pkl `
  --metrics-out mvp_model/artifacts/metrics.json `
  --train-info-out mvp_model/artifacts/train_info.json
```

**Linux / WSL:**
```bash
python -m mvp_model.train_mvp \
  --csv-path masters_csvs/matches.csv \
  --players-stats-path masters_csvs/detailed_matches_player_stats.csv \
  --model-out mvp_model/artifacts/model.pkl \
  --metrics-out mvp_model/artifacts/metrics.json \
  --train-info-out mvp_model/artifacts/train_info.json
```

### 2. Hacer Predicciones

```bash
python -m mvp_model.predict_mvp \
  --model mvp_model/artifacts/model.pkl \
  --csv masters_csvs/matches.csv \
  --out mvp_model/artifacts/preds_sample.csv
```

### 3. Visualizar Resultados del Test

```bash
python -m mvp_model.plot_test_predictions \
  --csv-path masters_csvs/matches.csv \
  --model mvp_model/artifacts/model.pkl \
  --out-dir mvp_model/artifacts/plots \
  --test-size 0.2 \
  --all-test \
  --threshold 0.5
```

**Salidas generadas:**
- `test_predictions_timeseries.png`: Serie temporal de probabilidades predichas vs. resultados reales
- `test_calibration_curve.png`: Curva de calibración del modelo
- `test_metrics.json`: Métricas detalladas (accuracy, precision, recall, F1, matriz de confusión)

### 4. Exportar Predicciones del Conjunto de Test

```bash
python -m mvp_model.print_test_tail \
  --csv-path masters_csvs/matches.csv \
  --model mvp_model/artifacts/model.pkl \
  --all-test \
  --out mvp_model/artifacts/test_tail_preds.csv \
  --threshold 0.5
```

El CSV resultante incluye:
- `p_team1_win`: Probabilidad predicha de victoria del equipo 1
- `pred_team1_win`: Predicción binaria (0 o 1)
- `team1_win`: Resultado real
- `correct`: Si la predicción fue correcta

## 📈 Interpretación de Resultados

### Métricas Clave

- **ROC-AUC** (≈0.5 azar, >0.6 aceptable, >0.8 excelente): Mide qué tan bien el modelo discrimina entre victorias y derrotas
- **Log Loss** (más bajo es mejor): Penaliza predicciones con alta confianza que son incorrectas
- **Brier Score** (más bajo es mejor): Mide la precisión de las probabilidades predichas
- **Accuracy**: Porcentaje de predicciones correctas (~86% en nuestro modelo)

### Gráficas

**Serie Temporal:**
- La línea de probabilidad debe estar **alta** (cerca de 1.0) cuando el resultado real es 1
- La línea debe estar **baja** (cerca de 0.0) cuando el resultado real es 0
- Muestra la evolución del rendimiento del modelo a lo largo del tiempo

**Curva de Calibración:**
- La curva ideal se pega a la **diagonal** (predicciones perfectamente calibradas)
- **Por debajo de la diagonal**: El modelo está sobreconfiado
- **Por encima de la diagonal**: El modelo está subconfiado

## 🛠️ Requisitos Técnicos

- **Python**: 3.9+ (recomendado 3.11 o 3.12)
- **Dependencias principales**:
  - `pandas`: Manipulación de datos
  - `numpy`: Operaciones numéricas
  - `scikit-learn`: Algoritmos de ML y métricas
  - `joblib`: Serialización de modelos
  - `matplotlib`: Visualizaciones
  - `xgboost` (opcional): Algoritmo alternativo más potente

## 🔧 Ejecución Completa (One-Liner)

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_all.ps1
```

**Linux/WSL:**
```bash
bash scripts/run_all.sh
```

Estos scripts ejecutan todo el pipeline: entrenamiento, predicción, generación de gráficas y exportación de resultados.

## 📝 Verificación Rápida

**PowerShell:**
```powershell
# Verificar que el modelo existe
Test-Path "mvp_model/artifacts/model.pkl"

# Ver métricas
Get-Content "mvp_model/artifacts/metrics.json"

# Abrir gráficas
Start-Process "mvp_model/artifacts/plots/test_predictions_timeseries.png"
```

**Linux/WSL:**
```bash
# Verificar modelo
ls -lh mvp_model/artifacts/model.pkl

# Ver métricas
cat mvp_model/artifacts/metrics.json

# Ver primeras líneas de predicciones
head mvp_model/artifacts/test_tail_preds.csv
```

## 🐛 Problemas Comunes

- **`ModuleNotFoundError`**: Asegúrate de activar el entorno virtual antes de ejecutar
  - Windows: `. .\.venv\Scripts\Activate.ps1`
  - Linux: `source .venv_cli/bin/activate`

- **`PermissionError` en PowerShell**: Ejecuta `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

- **Muy pocos partidos**: El modelo requiere al menos 20 partidos para entrenar

## 🔮 Próximos Pasos

- Incorporar ratings Elo **por mapa** (cada mapa tiene su propia meta)
- Agregar **forma reciente** de equipos (últimos 5-10 partidos)
- Incluir **estadísticas por agente** de los jugadores
- Considerar **contexto de patch** (cambios de balance del juego)
- Implementar **validación cruzada temporal** para evaluación más robusta

## 📧 Contacto

- **Alejandro Arango Mejía**: aarangom1@eafit.edu.co
- **Thomas Rivera Fernandez**: triveraf@eafit.edu.co

---

Para más detalles técnicos del modelo, consulta [`mvp_model/README.md`](mvp_model/README.md).
