#!/usr/bin/env python3
import os
import csv
import json
from pathlib import Path
from typing import List, Dict, Any

def _detect_project_root() -> str:
    """Detect project root so the script works from any CWD.

    Preference order:
      1) Current working directory if it looks like repo root
      2) Parent of this script directory (assuming scripts/ layout)
      3) Script directory
      4) Fallback to current working directory
    """
    script_dir = Path(__file__).resolve().parent
    candidates = [Path.cwd(), script_dir.parent, script_dir]
    for cand in candidates:
        if (cand / "masters_csvs").exists() or any(cand.glob("*_csvs")) or (cand / ".git").exists():
            return str(cand)
    return str(Path.cwd())


ROOT = _detect_project_root()
IN_DIR = os.path.join(ROOT, "masters_csvs")
OUT_PATH = os.path.join(IN_DIR, "matches_joined.csv")
OUT_TMP = os.path.join(IN_DIR, ".tmp_matches_joined.csv")

BASE_FILE = os.path.join(IN_DIR, "matches.csv")
OV_FILE = os.path.join(IN_DIR, "detailed_matches_overview.csv")
PLAYERS_FILE = os.path.join(IN_DIR, "detailed_matches_player_stats.csv")
MAPS_FILE = os.path.join(IN_DIR, "detailed_matches_maps.csv")


def read_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def group_by(rows: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    g: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        k = r.get(key, "")
        g.setdefault(k, []).append(r)
    return g


def main() -> None:
    # Leer bases
    base_rows = read_csv(BASE_FILE)
    ov_rows = read_csv(OV_FILE)
    player_rows = read_csv(PLAYERS_FILE)
    map_rows = read_csv(MAPS_FILE)

    # Indexaciones por match_id
    ov_by_match = group_by(ov_rows, "match_id")
    players_by_match = group_by(player_rows, "match_id")
    maps_by_match = group_by(map_rows, "match_id")

    # Construir encabezados de salida
    base_header = []
    if base_rows:
        base_header = list(base_rows[0].keys())

    ov_header = []
    if ov_rows:
        ov_header = [c for c in ov_rows[0].keys() if c != "match_id"]

    # Prefijar columnas de overview para evitar colisiones
    ov_out_cols = [f"ov_{c}" for c in ov_header]

    out_header = base_header + ov_out_cols + ["players_json", "maps_json"]

    os.makedirs(IN_DIR, exist_ok=True)
    # Escritura atómica
    with open(OUT_TMP, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(out_header)

        for base in base_rows:
            mid = base.get("match_id", "")

            # Overview: tomar primera fila si hay múltiples
            ov = ov_by_match.get(mid, [])
            ov_first = ov[0] if ov else {}
            ov_values = [ov_first.get(c, "") for c in ov_header]

            # Agregados: listas JSON (sin match_id para evitar duplicación)
            players = players_by_match.get(mid, [])
            players_slim = [
                {k: v for k, v in p.items() if k != "match_id"}
                for p in players
            ]
            maps_ = maps_by_match.get(mid, [])
            maps_slim = [
                {k: v for k, v in m.items() if k != "match_id"}
                for m in maps_
            ]

            row_out = [base.get(col, "") for col in base_header] + [
                *ov_values,
                json.dumps(players_slim, ensure_ascii=False),
                json.dumps(maps_slim, ensure_ascii=False),
            ]
            writer.writerow(row_out)

    os.replace(OUT_TMP, OUT_PATH)

    # Resumen simple
    print("Join completado:")
    print(f" - matches: {len(base_rows)} filas")
    print(f" - overview: {len(ov_rows)} filas")
    print(f" - player_stats: {len(player_rows)} filas")
    print(f" - maps: {len(map_rows)} filas")
    print(f"Salida: {OUT_PATH}")


if __name__ == "__main__":
    main()
