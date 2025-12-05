import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False

from mvp_model.utils.data_loader import load_and_prepare_matches
from mvp_model.utils.features import build_full_features


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MVP match outcome model from matches.csv")
    p.add_argument("--csv-path", default="masters_csvs/matches.csv", help="Path to matches.csv")
    p.add_argument("--players-stats-path", default="masters_csvs/detailed_matches_player_stats.csv", help="Path to detailed_matches_player_stats.csv")
    p.add_argument("--model-out", default="mvp_model/artifacts/model.pkl", help="Output path for trained model")
    p.add_argument("--metrics-out", default="mvp_model/artifacts/metrics.json", help="Output path for metrics JSON")
    p.add_argument("--train-info-out", default="mvp_model/artifacts/train_info.json", help="Output path for training info JSON")
    p.add_argument("--test-size", type=float, default=0.2, help="Fraction of tail for test (time split)")
    p.add_argument("--elo-k", type=float, default=32.0, help="Elo K-factor")
    p.add_argument("--elo-base", type=float, default=1500.0, help="Elo base rating")
    p.add_argument("--use-xgb", action="store_true", help="Force use XGBoost if available")
    return p.parse_args()


def make_features(df_matches: pd.DataFrame, df_players: pd.DataFrame, elo_k: float, elo_base: float):
    df_all, feats = build_full_features(df_matches, df_players, elo_k=elo_k, elo_base=elo_base)
    # Usaremos diffs + elo_diff (evita duplicidad)
    feat_cols = [
        "elo_diff",
        "diff_acs_mean",
        "diff_kast_mean",
    ]
    # Mantener también los componentes base de Elo por si el modelo los requiere
    if "elo1_before" in feats.columns and "elo2_before" in feats.columns:
        feat_cols = ["elo1_before", "elo2_before"] + feat_cols
    X = feats[feat_cols].copy()
    y = df_all["team1_win"].astype(int).values
    meta = {"feature_names": feat_cols}
    return X, y, meta


def time_train_test_split(X: pd.DataFrame, y: np.ndarray, test_size: float):
    n = len(X)
    n_test = int(max(1, round(n * test_size)))
    split = n - n_test
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]
    return X_train, X_test, y_train, y_test


def build_model(use_xgb: bool) -> Pipeline:
    if use_xgb and HAS_XGB:
        model = XGBClassifier(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=1.0,
            reg_lambda=1.0,
            reg_alpha=0.0,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=4,
            tree_method="hist",
        )
        # Imputación simple por seguridad también con árboles
        pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("model", model)])
    else:
        # Simple and robust fallback
        model = LogisticRegression(max_iter=200, solver="lbfgs")
        pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", model)])
    return pipe


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "log_loss": float(log_loss(y_test, proba)),
        "roc_auc": float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else None,
        "brier": float(brier_score_loss(y_test, proba)),
        "n_test": int(len(y_test)),
    }
    return metrics


def main():
    args = parse_args()

    df = load_and_prepare_matches(args.csv_path, filter_completed=True)
    # Cargar player stats
    from mvp_model.utils.data_loader import load_player_stats_csv
    df_players = load_player_stats_csv(args.players_stats_path)
    if len(df) < 20:
        raise SystemExit("Muy pocos partidos para entrenar un modelo (se requieren > 20).")

    X, y, meta = make_features(df, df_players, elo_k=args.elo_k, elo_base=args.elo_base)

    X_train, X_test, y_train, y_test = time_train_test_split(X, y, test_size=args.test_size)

    use_xgb = args.use_xgb and HAS_XGB
    model = build_model(use_xgb=use_xgb)
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)

    # Persist artifacts
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump(model, args.model_out)

    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    train_info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_total": int(len(df)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "elo_k": args.elo_k,
        "elo_base": args.elo_base,
        "features": meta["feature_names"],
        "model_type": "XGBoost" if use_xgb else "LogisticRegression",
        "csv_path": args.csv_path,
        "players_stats_path": args.players_stats_path,
    }
    with open(args.train_info_out, "w", encoding="utf-8") as f:
        json.dump(train_info, f, indent=2)

    print("Entrenamiento completado.")
    print("Métricas (test temporal):", json.dumps(metrics, indent=2))
    print(f"Modelo guardado en: {args.model_out}")


if __name__ == "__main__":
    main()
