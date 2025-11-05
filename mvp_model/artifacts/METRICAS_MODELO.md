# Métricas del Modelo MVP - Valorant Match Predictor

## Resumen Ejecutivo

Este documento contiene las métricas de rendimiento del modelo de predicción de resultados de partidos de Valorant. El modelo utiliza un sistema de clasificación basado en diferencias de Elo pre-partido para predecir si `team1` ganará el encuentro.

---

## Entrenamiento 1: Datos iniciales (Solo torneos 2025)

**Fecha de entrenamiento:** 2025-11-04T21:05:06.753195+00:00

### Datos de Entrenamiento

| Métrica | Valor |
|---------|-------|
| **Total de partidos** | 504 |
| **Partidos entrenamiento** | 403 (80%) |
| **Partidos test** | 101 (20%) |
| **Modelo utilizado** | LogisticRegression |
| **Features** | elo1_before, elo2_before, elo_diff |
| **Parámetros Elo** | K=32.0, Base=1500.0 |

### Métricas de Confiabilidad del Modelo (Probabilísticas)

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Log Loss** | 0.6763 | Cuanto más bajo mejor. Un valor < 0.7 es aceptable. |
| **ROC-AUC** | 0.5980 | Mide discriminación. >0.5 es mejor que azar. >0.6 es aceptable. |
| **Brier Score** | 0.2430 | Mide calibración. Más bajo es mejor. <0.25 es bueno. |

### Métricas Discretas (Umbral = 0.5)

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Accuracy** | 0.5446 (54.46%) | Precisión general del modelo |
| **Precision** | 0.6000 (60.00%) | De los predichos como ganadores, 60% realmente ganaron |
| **Recall** | 0.5357 (53.57%) | De los que realmente ganaron, el modelo capturó 53.57% |
| **F1-Score** | 0.5660 (56.60%) | Media armónica de Precision y Recall |

### Matriz de Confusión

| | Predicho: No Gana | Predicho: Gana | Total Real |
|---|-------------------|----------------|------------|
| **Real: No Gana** | 25 (TN) | 20 (FP) | 45 |
| **Real: Gana** | 26 (FN) | 30 (TP) | 56 |
| **Total Predicho** | 51 | 50 | **101** |

- **TP (True Positives):** 30 - Predijo ganador y acertó
- **TN (True Negatives):** 25 - Predijo no ganador y acertó
- **FP (False Positives):** 20 - Predijo ganador pero no ganó
- **FN (False Negatives):** 26 - No predijo ganador pero sí ganó

### Análisis de Predicciones del Test

| Métrica | Valor |
|---------|-------|
| **Total de predicciones en test** | 101 |
| **Aciertos totales** | 55 |
| **Tasa de aciertos** | 54.46% |

### Interpretación de Resultados

**Fortalezas:**
- El modelo tiene una discriminación aceptable (ROC-AUC > 0.59) que supera el azar
- La precisión es razonable (60%) cuando predice que un equipo ganará
- El Brier Score indica una calibración moderadamente buena

**Áreas de mejora:**
- El accuracy general es cercano al azar (54.46%), lo que sugiere que el modelo necesita más datos o features adicionales
- El recall es bajo (53.57%), indicando que el modelo se pierde casi la mitad de los casos donde team1 gana
- El Log Loss es alto, sugiriendo que las probabilidades no están bien calibradas

**Conclusión inicial:**
El modelo muestra un rendimiento básico pero funcional. Con solo 504 partidos y usando únicamente Elo pre-partido, el modelo logra superar ligeramente el azar, lo cual es un buen punto de partida para un MVP.

---

## Entrenamiento 2: Con datos adicionales (Torneos 2024 + 2025)

**Fecha de entrenamiento:** 2025-11-05T06:11:28.794465+00:00

### Datos de Entrenamiento

| Métrica | Valor |
|---------|-------|
| **Total de partidos** | 940 |
| **Partidos entrenamiento** | 752 (80%) |
| **Partidos test** | 188 (20%) |
| **Modelo utilizado** | LogisticRegression |
| **Features** | elo1_before, elo2_before, elo_diff |
| **Parámetros Elo** | K=32.0, Base=1500.0 |
| **Torneos incluidos** | 30 (15 de 2024 + 15 de 2025) |

### Métricas de Confiabilidad del Modelo (Probabilísticas)

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Log Loss** | 0.6581 | Mejoró desde 0.6763. Valor más bajo indica mejor calibración. |
| **ROC-AUC** | 0.6510 | Mejoró significativamente desde 0.5980. Excelente discriminación. |
| **Brier Score** | 0.2340 | Mejoró desde 0.2430. Calibración mejorada. |

### Métricas Discretas (Umbral = 0.5)

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Accuracy** | 0.6117 (61.17%) | Mejoró desde 54.46%. Supera el 60% de precisión. |
| **Precision** | 0.6598 (65.98%) | Mejoró desde 60.00%. Mayor confiabilidad en predicciones positivas. |
| **Recall** | 0.6154 (61.54%) | Mejoró desde 53.57%. Captura más casos reales de victorias. |
| **F1-Score** | 0.6368 (63.68%) | Mejoró desde 56.60%. Mejor balance entre precision y recall. |

### Matriz de Confusión

| | Predicho: No Gana | Predicho: Gana | Total Real |
|---|-------------------|----------------|------------|
| **Real: No Gana** | 51 (TN) | 33 (FP) | 84 |
| **Real: Gana** | 40 (FN) | 64 (TP) | 104 |
| **Total Predicho** | 91 | 97 | **188** |

- **TP (True Positives):** 64 - Predijo ganador y acertó
- **TN (True Negatives):** 51 - Predijo no ganador y acertó
- **FP (False Positives):** 33 - Predijo ganador pero no ganó
- **FN (False Negatives):** 40 - No predijo ganador pero sí ganó

### Análisis de Predicciones del Test

| Métrica | Valor |
|---------|-------|
| **Total de predicciones en test** | 188 |
| **Aciertos totales** | 115 |
| **Tasa de aciertos** | 61.17% |

### Interpretación de Resultados

**Mejoras significativas:**
- **ROC-AUC mejoró +8.9%** (de 0.598 a 0.651): El modelo ahora discrimina mucho mejor entre ganadores y perdedores
- **Accuracy mejoró +6.7%** (de 54.46% a 61.17%): El modelo acierta en más del 60% de los casos
- **Recall mejoró +8.0%** (de 53.57% a 61.54%): El modelo captura más victorias reales
- **F1-Score mejoró +7.1%** (de 56.60% a 63.68%): Mejor balance general entre precision y recall
- **Log Loss mejoró -2.7%**: Las probabilidades están mejor calibradas
- **Brier Score mejoró -3.7%**: Mejor calibración de probabilidades

**Conclusión:**
El modelo mejoró sustancialmente con más datos. Al duplicar el dataset (de 504 a 940 partidos), el modelo aprendió patrones más robustos y generalizó mejor. El ROC-AUC de 0.65 indica que el modelo tiene una discriminación aceptable y el accuracy del 61% supera significativamente el azar.

---

## Comparación de Modelos

### Tabla Comparativa de Métricas

| Métrica | Entrenamiento 1 (Solo 2025) | Entrenamiento 2 (2024 + 2025) | Mejora | Cambio % |
|---------|----------------------------|-------------------------------|--------|----------|
| **Total de partidos** | 504 | 940 | +436 | +86.5% |
| **Partidos entrenamiento** | 403 | 752 | +349 | +86.6% |
| **Partidos test** | 101 | 188 | +87 | +86.1% |
| **Log Loss** | 0.6763 | 0.6581 | -0.0182 | -2.7% |
| **ROC-AUC** | 0.5980 | 0.6510 | +0.0530 | +8.9% |
| **Brier Score** | 0.2430 | 0.2340 | -0.0090 | -3.7% |
| **Accuracy** | 54.46% | 61.17% | +6.71% | +12.3% |
| **Precision** | 60.00% | 65.98% | +5.98% | +10.0% |
| **Recall** | 53.57% | 61.54% | +7.97% | +14.9% |
| **F1-Score** | 56.60% | 63.68% | +7.08% | +12.5% |

### Análisis de Mejora

#### Métricas Probabilísticas
- ✅ **Log Loss**: Mejoró (menor es mejor), indicando mejor calibración de probabilidades
- ✅ **ROC-AUC**: Mejora significativa (+8.9%), ahora el modelo tiene discriminación aceptable
- ✅ **Brier Score**: Mejoró (menor es mejor), confirmando mejor calibración

#### Métricas Discretas
- ✅ **Accuracy**: Mejoró +6.7 puntos porcentuales, ahora supera el 60%
- ✅ **Precision**: Mejoró +6.0 puntos porcentuales, menos falsos positivos
- ✅ **Recall**: Mejoró +8.0 puntos porcentuales, captura más victorias reales
- ✅ **F1-Score**: Mejoró +7.1 puntos porcentuales, mejor balance general

### Impacto del Aumento de Datos

**Antes (Solo 2025):**
- 504 partidos totales
- Modelo cercano al azar (ROC-AUC ≈ 0.60)
- Accuracy del 54.46%

**Después (2024 + 2025):**
- 940 partidos totales (+86.5% más datos)
- Modelo con discriminación aceptable (ROC-AUC = 0.65)
- Accuracy del 61.17% (+12.3% relativo)

### Conclusión de la Comparación

**El modelo mejoró significativamente con más datos:**

1. **Más datos = mejor aprendizaje**: El aumento del 86.5% en datos de entrenamiento permitió al modelo aprender patrones más robustos y generalizables.

2. **Mejor discriminación**: El ROC-AUC mejoró de 0.598 a 0.651, lo que indica que el modelo ahora separa mejor los casos de victoria y derrota.

3. **Mejor precisión práctica**: El accuracy mejoró de 54.46% a 61.17%, lo que significa que el modelo acierta en más del 60% de los casos, superando significativamente el azar (50%).

4. **Mejor balance**: El F1-Score mejoró de 56.60% a 63.68%, indicando un mejor equilibrio entre precision y recall.

5. **Mejor calibración**: Tanto Log Loss como Brier Score mejoraron, lo que significa que las probabilidades predichas son más confiables.

**Recomendación**: El modelo muestra que con más datos históricos, el rendimiento mejora sustancialmente. Continuar añadiendo datos de nuevos torneos debería mejorar aún más el modelo.

---

## Notas Técnicas

- El split de datos es **temporal** (últimos 20% como test) para simular predicciones reales
- Todas las features son **pre-partido** para evitar data leakage
- El modelo se entrena cronológicamente, actualizando Elo secuencialmente
- El umbral de decisión por defecto es 0.5 (puede optimizarse)

---

*Última actualización: 2025-11-05*

