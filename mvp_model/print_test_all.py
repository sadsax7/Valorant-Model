import argparse
import pandas as pd
import joblib

from mvp_model.utils.features import build_full_features


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exporta TODO el bloque de test (último 20%) con probabilidades")
    p.add_argument("--csv-path", default="masters_csvs/matches.csv", help="Ruta a matches.csv")
    p.add_argument("--model", default="mvp_model/artifacts/model.pkl", help="Ruta al modelo entrenado .pkl")
    p.add_argument("--out", default="mvp_model/artifacts/test_preds.csv", help="Ruta de salida CSV")
    p.add_argument("--elo-k", type=float, default=32.0)
    p.add_argument("--elo-base", type=float, default=1500.0)
    p.add_argument("--players-stats-path", default="masters_csvs/detailed_matches_player_stats.csv")
    p.add_argument("--train-info", default="mvp_model/artifacts/train_info.json")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.csv_path)
    df = df[df["status"].astype(str).str.lower() == "completed"].copy()
    df["parsed_date"] = pd.to_datetime(df["date"], errors="coerce")
    if "match_id" in df.columns:
        df = df.sort_values(["parsed_date", "match_id"], kind="stable").reset_index(drop=True)
    else:
        df = df.sort_values(["parsed_date"], kind="stable").reset_index(drop=True)
    # Limpieza y etiqueta
    for c in ["team1", "team2", "winner"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    df["team1_win"] = (df["winner"] == df["team1"]).astype(int)

    # Alinear columnas con el entrenamiento
    try:
        import json
        with open(args.train_info, "r", encoding="utf-8") as f:
            train_info = json.load(f)
        feature_names = train_info.get("features", ["elo1_before", "elo2_before", "elo_diff"])  # fallback
    except FileNotFoundError:
        feature_names = ["elo1_before", "elo2_before", "elo_diff"]

    df_players = pd.read_csv(args.players_stats_path)
    df_all, feats = build_full_features(df, df_players, elo_k=args.elo_k, elo_base=args.elo_base)
    X = feats[feature_names]
    n = len(df)
    n_test = int(max(1, round(n * 0.2)))
    start = n - n_test
    model = joblib.load(args.model)
    proba = model.predict_proba(X.iloc[start:])[:, 1]
    out = df_all.iloc[start:].copy()
    out = out.assign(
        elo1_before=feats["elo1_before"].iloc[start:].values,
        elo2_before=feats["elo2_before"].iloc[start:].values,
        elo_diff=feats["elo_diff"].iloc[start:].values,
        diff_acs_mean=feats["diff_acs_mean"].iloc[start:].values if "diff_acs_mean" in feats.columns else None,
        diff_kast_mean=feats["diff_kast_mean"].iloc[start:].values if "diff_kast_mean" in feats.columns else None,
        p_team1_win=proba,
    )
    cols = [
        "parsed_date",
        "match_id" if "match_id" in out.columns else None,
        "team1",
        "team2",
        "elo1_before",
        "elo2_before",
        "elo_diff",
        "diff_acs_mean" if "diff_acs_mean" in out.columns else None,
        "diff_kast_mean" if "diff_kast_mean" in out.columns else None,
        "p_team1_win",
        "team1_win",
    ]
    cols = [c for c in cols if c is not None]
    out[cols].to_csv(args.out, index=False)
    print(f"Test size: {n_test} matches. Saved to: {args.out}")


if __name__ == "__main__":
    main()

