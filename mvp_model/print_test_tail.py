import argparse
import os
import pandas as pd
import joblib

from mvp_model.utils.elo import build_elo_features


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Imprime y exporta los últimos N partidos del bloque de test con sus probabilidades")
    p.add_argument("--csv-path", default="masters_csvs/matches.csv", help="Ruta a matches.csv")
    p.add_argument("--model", default="mvp_model/artifacts/model.pkl", help="Ruta al modelo entrenado .pkl")
    p.add_argument("--out", default="mvp_model/artifacts/test_tail_preds.csv", help="Ruta de salida CSV (últimos N del test)")
    p.add_argument("--last-n", type=int, default=10, help="Número de partidos del final del test a mostrar/exportar")
    p.add_argument("--all-test", action="store_true", help="Exportar TODO el bloque de test (ignora --last-n)")
    p.add_argument("--threshold", type=float, default=0.5, help="Umbral para convertir probabilidad en predicción (0/1)")
    p.add_argument("--elo-k", type=float, default=32.0)
    p.add_argument("--elo-base", type=float, default=1500.0)
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

    feats = build_elo_features(df, "team1", "team2", "team1_win", elo_k=args.elo_k, elo_base=args.elo_base)
    X = feats[["elo1_before", "elo2_before", "elo_diff"]]
    n = len(df)
    n_test = int(max(1, round(n * 0.2)))
    start = n - n_test
    model = joblib.load(args.model)
    proba = model.predict_proba(X.iloc[start:])[:, 1]
    out = df.iloc[start:].copy()
    out["p_team1_win"] = proba

    # Selección: todo el test o últimos N
    tail = out.copy() if args.all_test else out.tail(args.last_n).copy()
    # Predicción discreta y acierto
    tail["pred_team1_win"] = (tail["p_team1_win"] >= args.threshold).astype(int)
    tail["correct"] = tail["pred_team1_win"] == tail["team1_win"]

    # Columnas de salida
    cols = [
        "parsed_date" if "parsed_date" in tail.columns else None,
        "match_id" if "match_id" in tail.columns else None,
        "team1",
        "team2",
        "p_team1_win",
        "pred_team1_win",
        "team1_win",
        "correct",
    ]
    cols = [c for c in cols if c is not None]

    # Imprimir a consola
    label = "ALL test" if args.all_test else f"Last {args.last_n}"
    print(f"Test size: {n_test} matches. {label}:")
    print(tail[cols].to_string(index=False))

    # Exportar a CSV
    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        tail[cols].to_csv(args.out, index=False)
        print(f"\nGuardado CSV: {args.out}")


if __name__ == "__main__":
    main()
