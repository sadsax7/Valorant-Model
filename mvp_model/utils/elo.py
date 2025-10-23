from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class EloConfig:
    base: float = 1500.0
    k: float = 32.0


def expected_score(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def build_elo_features(
    df: pd.DataFrame,
    team1_col: str,
    team2_col: str,
    label_col: str,
    elo_k: float = 32.0,
    elo_base: float = 1500.0,
) -> pd.DataFrame:
    """
    Recorre el DataFrame en orden (se recomienda orden temporal) y construye
    features de Elo previas al partido. Actualiza Elo tras el resultado.

    Devuelve un DataFrame con columnas: elo1_before, elo2_before, elo_diff.
    """
    ratings: Dict[str, float] = {}
    elo1_before = np.zeros(len(df), dtype=float)
    elo2_before = np.zeros(len(df), dtype=float)

    # Iterar Secuencialmente; asume df ya está ordenado cronológicamente
    for i, row in df.iterrows():
        t1 = str(row[team1_col])
        t2 = str(row[team2_col])

        r1 = ratings.get(t1, elo_base)
        r2 = ratings.get(t2, elo_base)
        elo1_before[i] = r1
        elo2_before[i] = r2

        # Actualización tras el partido
        if label_col in df.columns:
            y = float(row[label_col])  # 1 si gana team1, 0 si no
            e1 = expected_score(r1, r2)
            e2 = 1.0 - e1
            r1_new = r1 + elo_k * (y - e1)
            r2_new = r2 + elo_k * ((1.0 - y) - e2)
            ratings[t1] = r1_new
            ratings[t2] = r2_new

    out = pd.DataFrame({
        "elo1_before": elo1_before,
        "elo2_before": elo2_before,
    })
    out["elo_diff"] = out["elo1_before"] - out["elo2_before"]
    return out

