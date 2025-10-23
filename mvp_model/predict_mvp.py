import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

from mvp_model.utils.elo import build_elo_features


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict match win probability for team1 using trained MVP model")
    p.add_argument("--model", required=True, help="Path to trained model .pkl")
    p.add_argument("--csv", required=True, help="Path to matches.csv-like file")
    p.add_argument("--out", default=None, help="Optional output CSV for predictions")
    p.add_argument("--elo-k", type=float, default=32.0, help="Elo K-factor (must match training)")
    p.add_argument("--elo-base", type=float, default=1500.0, help="Elo base rating (must match training)")
    return p.parse_args()


def load_and_prepare(csv_path: str, elo_k: float, elo_base: float) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Keep only completed or upcoming; label may not exist, but features no leak.
    if "status" in df.columns:
        # No filtramos para predicción, pero mantenemos el orden por fecha si existe
        pass
    if "date" in df.columns:
        df["parsed_date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["parsed_date"] = pd.NaT
    if "match_id" in df.columns:
        df = df.sort_values(["parsed_date", "match_id"], kind="stable").reset_index(drop=True)
    else:
        df = df.sort_values(["parsed_date"], kind="stable").reset_index(drop=True)

    # Si existe winner, generamos label para referencia; no se usa para predecir
    if "winner" in df.columns and "team1" in df.columns:
        # Limpieza de espacios para evitar mismatches por espacios finales
        df["team1"] = df["team1"].astype(str).str.strip()
        df["team2"] = df["team2"].astype(str).str.strip()
        df["winner"] = df["winner"].astype(str).str.strip()
        df["team1_win"] = (df["winner"] == df["team1"]).astype(int)

    feats = build_elo_features(df, team1_col="team1", team2_col="team2", label_col="team1_win" if "team1_win" in df.columns else "__none__", elo_k=elo_k, elo_base=elo_base)
    feats["match_id"] = df["match_id"] if "match_id" in df.columns else np.arange(len(df))
    return df, feats


def main():
    args = parse_args()
    model = joblib.load(args.model)
    df, feats = load_and_prepare(args.csv, args.elo_k, args.elo_base)

    # Por compatibilidad con el MVP entrenado
    feature_names = ["elo1_before", "elo2_before", "elo_diff"]
    X = feats[feature_names]
    proba = model.predict_proba(X)[:, 1]

    out_df = pd.DataFrame({
        "match_id": feats["match_id"],
        "team1": df["team1"],
        "team2": df["team2"],
        "p_team1_win": proba,
    })
    if "team1_win" in df.columns:
        out_df["team1_win"] = df["team1_win"].values

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        out_df.to_csv(args.out, index=False)
        print(f"Predicciones guardadas en: {args.out}")
    else:
        print(out_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
