import argparse
import os
import json
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

from mvp_model.utils.elo import build_elo_features


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot test predictions vs actuals and calibration curve")
    p.add_argument("--csv-path", default="masters_csvs/matches.csv", help="Path to matches.csv")
    p.add_argument("--model", default="mvp_model/artifacts/model.pkl", help="Path to trained model .pkl")
    p.add_argument("--out-dir", default="mvp_model/artifacts/plots", help="Output directory for plots")
    p.add_argument("--test-size", type=float, default=0.2, help="Fraction of tail for test (time split)")
    p.add_argument("--last-n", type=int, default=None, help="Limit to last N test matches for the time plot")
    p.add_argument("--all-test", action="store_true", help="Use entire test block and ignore --last-n")
    p.add_argument("--elo-k", type=float, default=32.0, help="Elo K-factor (must match training)")
    p.add_argument("--elo-base", type=float, default=1500.0, help="Elo base rating (must match training)")
    p.add_argument("--style", default="seaborn-v0_8", help="Matplotlib style to use")
    p.add_argument("--dpi", type=int, default=140, help="Figure DPI for saved images")
    p.add_argument("--threshold", type=float, default=0.5, help="Threshold for discrete metrics (confusion matrix)")
    return p.parse_args()


def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Filter only completed matches for proper labels
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "completed"].copy()
    if "date" in df.columns:
        df["parsed_date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["parsed_date"] = pd.NaT
    # Clean strings and build label
    for c in ["team1", "team2", "winner"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    if "winner" in df.columns and "team1" in df.columns:
        df["team1_win"] = (df["winner"] == df["team1"]).astype(int)
    else:
        raise ValueError("CSV must contain columns: team1, winner")

    if "match_id" in df.columns:
        df = df.sort_values(["parsed_date", "match_id"], kind="stable").reset_index(drop=True)
    else:
        df = df.sort_values(["parsed_date"], kind="stable").reset_index(drop=True)
    return df


def compute_test_slice(n: int, test_size: float, last_n: Optional[int]) -> slice:
    n_test = int(max(1, round(n * test_size)))
    start = n - n_test
    if last_n is not None:
        start = max(start, n - last_n)
    return slice(start, n)


def main():
    args = parse_args()

    # Lazy import matplotlib to avoid hard dependency at import time
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    df = load_and_prepare(args.csv_path)
    feats = build_elo_features(
        df, team1_col="team1", team2_col="team2", label_col="team1_win", elo_k=args.elo_k, elo_base=args.elo_base
    )
    X = feats[["elo1_before", "elo2_before", "elo_diff"]]

    last_n = None if args.all_test else args.last_n
    idx = compute_test_slice(len(df), args.test_size, last_n)
    df_test = df.iloc[idx].copy()
    X_test = X.iloc[idx]
    y_test = df_test["team1_win"].astype(int).values

    model = joblib.load(args.model)
    proba = model.predict_proba(X_test)[:, 1]

    # Metrics summary (probabilistic)
    metrics = {
        "log_loss": float(log_loss(y_test, proba)),
        "roc_auc": float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else None,
        "brier": float(brier_score_loss(y_test, proba)),
        "n_test": int(len(y_test)),
    }

    # Discrete metrics at threshold
    pred = (proba >= args.threshold).astype(int)
    tp = int(((pred == 1) & (y_test == 1)).sum())
    tn = int(((pred == 0) & (y_test == 0)).sum())
    fp = int(((pred == 1) & (y_test == 0)).sum())
    fn = int(((pred == 0) & (y_test == 1)).sum())
    acc = (tp + tn) / max(1, len(y_test))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    metrics["discrete"] = {
        "threshold": args.threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }

    os.makedirs(args.out_dir, exist_ok=True)

    plt.style.use(args.style)

    # Time series plot: predicted probability vs time with actual outcomes at 0/1
    x = df_test["parsed_date"] if df_test["parsed_date"].notna().any() else np.arange(len(df_test))

    fig, ax = plt.subplots(figsize=(10, 4), dpi=args.dpi)
    ax.plot(x, proba, label="Predicción p(team1 gana)", color="#1f77b4")
    # Actual outcomes as scatter at 0/1
    ax.scatter(x, y_test, label="Resultado real (0/1)", color="#d62728", s=16, alpha=0.7)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Probabilidad / Resultado")
    ax.set_title("Predicciones vs resultados – bloque de test")
    ax.legend(loc="best")
    fig.autofmt_xdate()
    out_ts = os.path.join(args.out_dir, "test_predictions_timeseries.png")
    fig.savefig(out_ts, bbox_inches="tight")
    plt.close(fig)

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
    fig2, ax2 = plt.subplots(figsize=(5, 5), dpi=args.dpi)
    ax2.plot([0, 1], [0, 1], "--", color="gray", label="Calibración perfecta")
    ax2.plot(prob_pred, prob_true, marker="o", label="Modelo")
    ax2.set_xlabel("Predicción media por bin")
    ax2.set_ylabel("Fracción positiva por bin")
    ax2.set_title("Curva de calibración (test)")
    ax2.legend(loc="best")
    out_cal = os.path.join(args.out_dir, "test_calibration_curve.png")
    fig2.savefig(out_cal, bbox_inches="tight")
    plt.close(fig2)

    # Save metrics JSON alongside plots for quick inspection
    with open(os.path.join(args.out_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Plots guardados:")
    print(" - ", out_ts)
    print(" - ", out_cal)
    print("Métricas (test):", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
