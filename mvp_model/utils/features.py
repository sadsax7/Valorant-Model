from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from mvp_model.utils.elo import build_elo_features


def _safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_team_aggregates(player_stats: pd.DataFrame) -> pd.DataFrame:
    """
    A partir de `detailed_matches_player_stats.csv` calcula promedios por equipo dentro
    de cada partido. Columnas esperadas: match_id, player_team, acs, kast.

    Devuelve un DataFrame con columnas:
      match_id, player_team, acs_mean, kast_mean
    """
    dfp = player_stats.copy()
    if "acs" in dfp.columns:
        dfp["acs"] = _safe_to_numeric(dfp["acs"])
    if "kast" in dfp.columns:
        dfp["kast"] = _safe_to_numeric(dfp["kast"])

    agg = (
        dfp.groupby(["match_id", "player_team"], dropna=False)[["acs", "kast"]]
        .mean()
        .rename(columns={"acs": "acs_mean", "kast": "kast_mean"})
        .reset_index()
    )
    return agg


def attach_team_features(
    matches_df: pd.DataFrame,
    team_agg: pd.DataFrame,
) -> pd.DataFrame:
    """
    Une los promedios por equipo al nivel de partido. Añade:
      team1_acs_mean, team2_acs_mean, diff_acs_mean,
      team1_kast_mean, team2_kast_mean, diff_kast_mean
    """
    # Merge para team1
    t1 = team_agg.rename(
        columns={"player_team": "team1", "acs_mean": "team1_acs_mean", "kast_mean": "team1_kast_mean"}
    )
    out = matches_df.merge(t1[["match_id", "team1", "team1_acs_mean", "team1_kast_mean"]],
                           on=["match_id", "team1"], how="left")

    # Merge para team2
    t2 = team_agg.rename(
        columns={"player_team": "team2", "acs_mean": "team2_acs_mean", "kast_mean": "team2_kast_mean"}
    )
    out = out.merge(t2[["match_id", "team2", "team2_acs_mean", "team2_kast_mean"]],
                    on=["match_id", "team2"], how="left")

    # Diferencias (team1 - team2)
    out["diff_acs_mean"] = out["team1_acs_mean"] - out["team2_acs_mean"]
    out["diff_kast_mean"] = out["team1_kast_mean"] - out["team2_kast_mean"]
    return out


def build_full_features(
    matches_df: pd.DataFrame,
    player_stats: pd.DataFrame,
    elo_k: float,
    elo_base: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Devuelve (df_with_features, feats_only_df)
    - df_with_features: matches con columnas originales + elo_before/diff + agregados y diffs
    - feats_only_df: DataFrame con solo columnas de features numéricas sugeridas
    """
    # Elo features
    elo_feats = build_elo_features(
        df=matches_df,
        team1_col="team1",
        team2_col="team2",
        label_col="team1_win" if "team1_win" in matches_df.columns else "__none__",
        elo_k=elo_k,
        elo_base=elo_base,
    )
    with_elo = matches_df.copy()
    with_elo = with_elo.assign(
        elo1_before=elo_feats["elo1_before"].values,
        elo2_before=elo_feats["elo2_before"].values,
        elo_diff=elo_feats["elo_diff"].values,
    )

    # Team aggregates (ACS/KAST)
    team_agg = build_team_aggregates(player_stats)
    with_all = attach_team_features(with_elo, team_agg)

    # Features sugeridas para el modelo: usar diffs y elo_diff (evita duplicidad)
    feats = with_all[[
        "elo1_before", "elo2_before", "elo_diff",
        "team1_acs_mean", "team2_acs_mean", "diff_acs_mean",
        "team1_kast_mean", "team2_kast_mean", "diff_kast_mean",
    ]].copy()
    return with_all, feats




