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
    de cada partido. Columnas esperadas: match_id, player_team, acs, kast, adr, hs_percent.

    Devuelve un DataFrame con columnas:
      match_id, player_team, acs_mean, kast_mean, adr_mean, hs_percent_mean
    """
    dfp = player_stats.copy()
    if "acs" in dfp.columns:
        dfp["acs"] = _safe_to_numeric(dfp["acs"])
    if "kast" in dfp.columns:
        dfp["kast"] = _safe_to_numeric(dfp["kast"])
    if "adr" in dfp.columns:
        dfp["adr"] = _safe_to_numeric(dfp["adr"])
    if "hs_percent" in dfp.columns:
        dfp["hs_percent"] = _safe_to_numeric(dfp["hs_percent"])

    # Seleccionar columnas disponibles para agregación
    agg_cols = []
    rename_dict = {}
    if "acs" in dfp.columns:
        agg_cols.append("acs")
        rename_dict["acs"] = "acs_mean"
    if "kast" in dfp.columns:
        agg_cols.append("kast")
        rename_dict["kast"] = "kast_mean"
    if "adr" in dfp.columns:
        agg_cols.append("adr")
        rename_dict["adr"] = "adr_mean"
    if "hs_percent" in dfp.columns:
        agg_cols.append("hs_percent")
        rename_dict["hs_percent"] = "hs_percent_mean"

    agg = (
        dfp.groupby(["match_id", "player_team"], dropna=False)[agg_cols]
        .mean()
        .rename(columns=rename_dict)
        .reset_index()
    )
    return agg


def attach_team_features(
    matches_df: pd.DataFrame,
    team_agg: pd.DataFrame,
) -> pd.DataFrame:
    """
    Une los promedios por equipo al nivel de partido. Añade dinámicamente:
      team1_X_mean, team2_X_mean, diff_X_mean
    para cada métrica X disponible (acs, kast, adr, hs_percent)
    """
    out = matches_df.copy()
    
    # Detectar qué métricas están disponibles
    available_metrics = []
    for col in team_agg.columns:
        if col.endswith("_mean") and col not in ["match_id", "player_team"]:
            metric_name = col  # ej: "acs_mean", "adr_mean"
            available_metrics.append(metric_name)
    
    # Merge para team1
    t1_rename = {"player_team": "team1"}
    t1_cols = ["match_id", "team1"]
    for metric in available_metrics:
        t1_name = f"team1_{metric}"
        t1_rename[metric] = t1_name
        t1_cols.append(t1_name)
    
    t1 = team_agg.rename(columns=t1_rename)
    out = out.merge(t1[t1_cols], on=["match_id", "team1"], how="left")
    
    # Merge para team2
    t2_rename = {"player_team": "team2"}
    t2_cols = ["match_id", "team2"]
    for metric in available_metrics:
        t2_name = f"team2_{metric}"
        t2_rename[metric] = t2_name
        t2_cols.append(t2_name)
    
    t2 = team_agg.rename(columns=t2_rename)
    out = out.merge(t2[t2_cols], on=["match_id", "team2"], how="left")
    
    # Calcular diferencias (team1 - team2)
    for metric in available_metrics:
        diff_name = f"diff_{metric}"
        team1_col = f"team1_{metric}"
        team2_col = f"team2_{metric}"
        out[diff_name] = out[team1_col] - out[team2_col]
    
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

    # Team aggregates (ACS/KAST/ADR/HS%)
    team_agg = build_team_aggregates(player_stats)
    with_all = attach_team_features(with_elo, team_agg)

    # Detectar todas las columnas diff_* disponibles dinámicamente
    diff_cols = [col for col in with_all.columns if col.startswith("diff_") and col != "diff"]
    
    # Features sugeridas: elo + todas las diferencias disponibles
    feat_cols = ["elo1_before", "elo2_before", "elo_diff"] + diff_cols
    
    # Agregar también las columnas team1_* y team2_* para referencia
    team_cols = []
    for col in with_all.columns:
        if col.startswith("team1_") or col.startswith("team2_"):
            if col.endswith("_mean"):
                team_cols.append(col)
    
    feat_cols = feat_cols + team_cols
    
    # Seleccionar solo las columnas que existen
    available_feat_cols = [col for col in feat_cols if col in with_all.columns]
    
    feats = with_all[available_feat_cols].copy()
    return with_all, feats




