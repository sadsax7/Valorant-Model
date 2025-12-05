"""Unified script for analyzing test predictions and generating reports."""

import argparse
import os
import json
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

from mvp_model.utils.data_loader import load_and_prepare_matches, load_player_stats_csv
from mvp_model.utils.model_utils import (
    get_feature_names,
    build_features_from_data,
    compute_test_slice
)
from mvp_model.utils.cli_args import (
    add_common_data_args,
    add_common_model_args,
    add_common_elo_args,
    add_test_args
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze test predictions: export CSV, print results, and/or generate plots"
    )
    
    # Add common arguments
    add_common_data_args(p)
    add_common_model_args(p)
    add_common_elo_args(p)
    add_test_args(p)
    
    # Output options
    p.add_argument(
        "--out",
        default=None,
        help="Output CSV path for predictions (default: mvp_model/artifacts/test_analysis.csv)"
    )
    p.add_argument(
        "--print",
        action="store_true",
        help="Print predictions to console"
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots (timeseries and calibration curve)"
    )
    p.add_argument(
        "--plot-dir",
        default="mvp_model/artifacts/plots",
        help="Output directory for plots"
    )
    
    # Analysis options
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for discrete predictions (0/1)"
    )
    p.add_argument(
        "--style",
        default="seaborn-v0_8",
        help="Matplotlib style for plots"
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=140,
        help="Figure DPI for saved images"
    )
    
    return p.parse_args()


def compute_metrics(y_test: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute all metrics (probabilistic and discrete)."""
    metrics = {
        "log_loss": float(log_loss(y_test, proba)),
        "roc_auc": float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else None,
        "brier": float(brier_score_loss(y_test, proba)),
        "n_test": int(len(y_test)),
    }
    
    # Discrete metrics
    pred = (proba >= threshold).astype(int)
    tp = int(((pred == 1) & (y_test == 1)).sum())
    tn = int(((pred == 0) & (y_test == 0)).sum())
    fp = int(((pred == 1) & (y_test == 0)).sum())
    fn = int(((pred == 0) & (y_test == 1)).sum())
    acc = (tp + tn) / max(1, len(y_test))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    
    metrics["discrete"] = {
        "threshold": threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }
    
    return metrics


def generate_plots(
    df_test: pd.DataFrame,
    proba: np.ndarray,
    y_test: np.ndarray,
    out_dir: str,
    style: str,
    dpi: int
) -> None:
    """Generate timeseries and calibration plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    os.makedirs(out_dir, exist_ok=True)
    plt.style.use(style)
    
    # Time series plot
    x = df_test["parsed_date"] if df_test["parsed_date"].notna().any() else np.arange(len(df_test))
    
    fig, ax = plt.subplots(figsize=(10, 4), dpi=dpi)
    ax.plot(x, proba, label="Predicción p(team1 gana)", color="#1f77b4")
    ax.scatter(x, y_test, label="Resultado real (0/1)", color="#d62728", s=16, alpha=0.7)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Probabilidad / Resultado")
    ax.set_title("Predicciones vs resultados – bloque de test")
    ax.legend(loc="best")
    fig.autofmt_xdate()
    out_ts = os.path.join(out_dir, "test_predictions_timeseries.png")
    fig.savefig(out_ts, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Timeseries plot: {out_ts}")
    
    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
    fig2, ax2 = plt.subplots(figsize=(5, 5), dpi=dpi)
    ax2.plot([0, 1], [0, 1], "--", color="gray", label="Calibración perfecta")
    ax2.plot(prob_pred, prob_true, marker="o", label="Modelo")
    ax2.set_xlabel("Predicción media por bin")
    ax2.set_ylabel("Fracción positiva por bin")
    ax2.set_title("Curva de calibración (test)")
    ax2.legend(loc="best")
    out_cal = os.path.join(out_dir, "test_calibration_curve.png")
    fig2.savefig(out_cal, bbox_inches="tight")
    plt.close(fig2)
    print(f"  ✓ Calibration plot: {out_cal}")


def main():
    args = parse_args()
    
    # Load data
    df = load_and_prepare_matches(args.csv_path, filter_completed=True)
    df_players = load_player_stats_csv(args.players_stats_path)
    
    # Get feature names
    feature_names = get_feature_names(args.train_info)
    
    # Build features
    df_all, X = build_features_from_data(
        df, df_players, args.elo_k, args.elo_base, feature_names
    )
    
    # Compute test slice
    last_n = None if args.all_test else args.last_n
    idx = compute_test_slice(len(df), args.test_size, last_n)
    df_test = df_all.iloc[idx].copy()
    X_test = X.iloc[idx]
    y_test = df_test["team1_win"].astype(int).values
    
    # Load model and predict
    model = joblib.load(args.model)
    proba = model.predict_proba(X_test)[:, 1]
    
    # Compute metrics
    metrics = compute_metrics(y_test, proba, args.threshold)
    
    # Prepare output DataFrame
    out_df = df_test.copy()
    out_df["p_team1_win"] = proba
    out_df["pred_team1_win"] = (proba >= args.threshold).astype(int)
    out_df["correct"] = out_df["pred_team1_win"] == out_df["team1_win"]
    
    # Select output columns
    cols = [
        "parsed_date",
        "match_id" if "match_id" in out_df.columns else None,
        "team1",
        "team2",
        "p_team1_win",
        "pred_team1_win",
        "team1_win",
        "correct",
    ]
    cols = [c for c in cols if c is not None]
    out_df = out_df[cols]
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Test Analysis Results")
    print(f"{'='*60}")
    print(f"Test size: {len(y_test)} matches")
    print(f"Test range: {idx.start} to {idx.stop}")
    print(f"\nMetrics (probabilistic):")
    print(f"  Log Loss:  {metrics['log_loss']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "  ROC-AUC:   N/A")
    print(f"  Brier:     {metrics['brier']:.4f}")
    print(f"\nMetrics (discrete, threshold={args.threshold}):")
    d = metrics['discrete']
    print(f"  Accuracy:  {d['accuracy']:.2%}")
    print(f"  Precision: {d['precision']:.2%}")
    print(f"  Recall:    {d['recall']:.2%}")
    print(f"  F1 Score:  {d['f1']:.2%}")
    print(f"  Confusion: TP={d['tp']}, TN={d['tn']}, FP={d['fp']}, FN={d['fn']}")
    
    # Print predictions if requested
    if args.print:
        print(f"\n{'='*60}")
        print("Predictions:")
        print(f"{'='*60}")
        print(out_df.to_string(index=False))
    
    # Save CSV
    out_path = args.out or "mvp_model/artifacts/test_analysis.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\n✓ Predictions saved to: {out_path}")
    
    # Generate plots if requested
    if args.plot:
        print(f"\nGenerating plots...")
        generate_plots(df_test, proba, y_test, args.plot_dir, args.style, args.dpi)
        
        # Save metrics JSON
        metrics_path = os.path.join(args.plot_dir, "test_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"  ✓ Metrics JSON: {metrics_path}")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
