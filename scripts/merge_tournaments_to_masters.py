#!/usr/bin/env python3
import os
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Optional

BASE_NAMES = [
    "agents_stats",
    "detailed_matches_maps",
    "detailed_matches_overview",
    "detailed_matches_player_stats",
    "economy_data",
    "event_info",
    "maps_stats",
    "matches",
    "performance_data",
    "player_stats",
]

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

def _detect_data_root(root: str, override: Optional[str] = None) -> str:
    """Detect where the raw tournament folders live.

    Preference order (unless overridden):
      1) ./tournaments
      2) ./datasets
      3) ./ (repo root)
    """
    if override:
        return override
    tournaments_dir = Path(root) / "tournaments"
    if tournaments_dir.exists() and tournaments_dir.is_dir():
        return str(tournaments_dir)
    datasets_dir = Path(root) / "datasets"
    if datasets_dir.exists() and datasets_dir.is_dir():
        return str(datasets_dir)
    return root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge tournament CSVs into masters_csvs")
    p.add_argument("--data-root", default=None, help="Folder containing *_csvs (default: ./datasets or ./)")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output folder for masters_csvs (default: <data-root>/masters_csvs)",
    )
    return p.parse_args()


ARGS = parse_args()
DATA_ROOT = _detect_data_root(ROOT, ARGS.data_root)
# By default, keep masters_csvs at repo root as requested
OUTPUT_DIR = ARGS.output_dir or os.path.join(ROOT, "masters_csvs")

def list_tournament_dirs(root: str, out_dir_name: str) -> List[str]:
    items: List[str] = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if (
            os.path.isdir(path)
            and not name.startswith('.')
            and name != out_dir_name
            and name.endswith('_csvs')  # solo dumps de torneos
        ):
            items.append(name)
    items.sort()
    return items


def consolidate_one(base_name: str, tournaments: List[str]) -> Dict[str, int]:
    """
    Consolidate all `{base_name}.csv` files across tournaments, adding a
    `tournament_name` column. Uses a union of all headers found to avoid
    dropping files when columns differ (fills missing cells with '').

    Returns summary dict: {"rows": int, "files": int, "skipped": int}
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{base_name}.csv")
    out_tmp = os.path.join(OUTPUT_DIR, f".tmp_{base_name}.csv")

    # 1) First pass: build union header in appearance order
    union_header: List[str] = []
    seen_cols = set()
    available_files: List[tuple[str, str, List[str]]] = []  # (tname, path, header)
    skipped_files = 0

    for tname in tournaments:
        in_path = os.path.join(DATA_ROOT, tname, f"{base_name}.csv")
        if not os.path.exists(in_path):
            skipped_files += 1
            continue
        with open(in_path, 'r', newline='', encoding='utf-8-sig') as fin:
            reader = csv.reader(fin)
            try:
                file_header = next(reader)
            except StopIteration:
                # empty file; skip
                continue
        available_files.append((tname, in_path, file_header))
        for col in file_header:
            if col not in seen_cols and col != "tournament_name":
                union_header.append(col)
                seen_cols.add(col)

    # 2) Write with union header
    total_rows = 0
    used_files = 0
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        with open(out_tmp, 'w', newline='', encoding='utf-8') as fout:
            writer = csv.writer(fout)
            writer.writerow(union_header + ["tournament_name"])  # keep new column at end

            for tname, in_path, file_header in available_files:
                with open(in_path, 'r', newline='', encoding='utf-8-sig') as fin:
                    dict_reader = csv.DictReader(fin)
                    # Normalize: if the input has the tournament_name already, ignore
                    for row in dict_reader:
                        out_row = [row.get(col, '') for col in union_header]
                        writer.writerow(out_row + [tname])
                        total_rows += 1
                used_files += 1
        # Reemplazo atómico del archivo final
        os.replace(out_tmp, out_path)
    finally:
        # Limpieza si quedara temporal por error
        if os.path.exists(out_tmp):
            try:
                os.remove(out_tmp)
            except OSError:
                pass

    return {"rows": total_rows, "files": used_files, "skipped": skipped_files}


def main():
    out_dir_name = os.path.basename(OUTPUT_DIR)
    tournaments = list_tournament_dirs(DATA_ROOT, out_dir_name)
    # Mantener solo directorios que tienen al menos un CSV esperado
    tournaments = [
        t for t in tournaments
        if any(os.path.exists(os.path.join(DATA_ROOT, t, f"{bn}.csv")) for bn in BASE_NAMES)
    ]

    if not tournaments:
        print("No se encontraron carpetas de torneos con CSVs esperados.")
        return

    print(f"Torneos detectados ({len(tournaments)}):")
    for t in tournaments:
        print(f" - {t}")

    print(f"\nGenerando maestros en: {OUTPUT_DIR}\n")

    totals: Dict[str, Dict[str, int]] = {}
    for bn in BASE_NAMES:
        summary = consolidate_one(bn, tournaments)
        totals[bn] = summary
        print(f"[OK] {bn}.csv -> filas: {summary['rows']}, archivos usados: {summary['files']}, omitidos: {summary['skipped']}")

    print("\nResumen total:")
    for bn, s in totals.items():
        print(f" - {bn}.csv: {s['rows']} filas de {s['files']} archivos (omitidos {s['skipped']})")

if __name__ == "__main__":
    main()
