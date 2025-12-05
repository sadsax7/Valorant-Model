import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

from mvp_model.utils.data_loader import load_and_prepare_matches, load_player_stats_csv
from mvp_model.utils.model_utils import build_features_from_data, get_feature_names


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict match win probability for team1 using trained MVP model")
    p.add_argument("--model", required=True, help="Path to trained model .pkl")
    p.add_argument("--csv", required=True, help="Path to matches.csv-like file")
    p.add_argument("--out", default=None, help="Optional output CSV for predictions")
    p.add_argument("--players-stats-path", default="masters_csvs/detailed_matches_player_stats.csv", help="Path to detailed_matches_player_stats.csv")
    p.add_argument("--train-info", default="mvp_model/artifacts/train_info.json", help="Path to train_info.json to align feature columns")
    p.add_argument("--elo-k", type=float, default=32.0, help="Elo K-factor (must match training)")
    p.add_argument("--elo-base", type=float, default=1500.0, help="Elo base rating (must match training)")
    return p.parse_args()


def load_and_prepare(csv_path: str, players_path: str, elo_k: float, elo_base: float, feature_names: list):
    """Load and prepare data for prediction."""
    # Load matches (don't filter by status for predictions)
    df = load_and_prepare_matches(csv_path, filter_completed=False)
    
    # Load player stats
    df_players = load_player_stats_csv(players_path)
    
    # Build features
    df_with_all, X = build_features_from_data(df, df_players, elo_k, elo_base, feature_names)
    
    # Add match_id to features for output
    if "match_id" in df_with_all.columns:
        match_ids = df_with_all["match_id"].values
    else:
        match_ids = np.arange(len(df_with_all))
    
    return df_with_all, X, match_ids


def main():
    args = parse_args()
    model = joblib.load(args.model)
    
    # Get feature names from train_info
    feature_names = get_feature_names(args.train_info)
    
    # Load and prepare data
    df, X, match_ids = load_and_prepare(
        args.csv, args.players_stats_path, args.elo_k, args.elo_base, feature_names
    )
    
    proba = model.predict_proba(X)[:, 1]

    out_df = pd.DataFrame({
        "match_id": match_ids,
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
