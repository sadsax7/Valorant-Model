"""Path detection utilities for scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def detect_project_root() -> Path:
    """
    Detect project root so scripts work from any CWD.
    
    Preference order:
      1) Current working directory if it looks like repo root
      2) Parent of this script directory (assuming scripts/ layout)
      3) Script directory
      4) Fallback to current working directory
    
    Returns:
        Path to project root
    """
    script_dir = Path(__file__).resolve().parent.parent  # Go up from scripts/utils/
    candidates = [Path.cwd(), script_dir.parent, script_dir]
    
    for cand in candidates:
        if (cand / "masters_csvs").exists() or any(cand.glob("*_csvs")) or (cand / ".git").exists():
            return cand
    
    return Path.cwd()


def detect_data_root(root: Path, override: Optional[str] = None) -> Path:
    """
    Detect where the raw tournament folders live.
    
    Preference order (unless overridden):
      1) ./tournaments
      2) ./datasets
      3) ./ (repo root)
    
    Args:
        root: Project root path
        override: Optional override path
    
    Returns:
        Path to data root
    """
    if override:
        return Path(override)
    
    tournaments_dir = root / "tournaments"
    if tournaments_dir.exists() and tournaments_dir.is_dir():
        return tournaments_dir
    
    datasets_dir = root / "datasets"
    if datasets_dir.exists() and datasets_dir.is_dir():
        return datasets_dir
    
    return root


def detect_masters_dir(root: Path, override: Optional[str] = None) -> Path:
    """
    Detect masters_csvs directory location.
    
    Args:
        root: Project root path
        override: Optional override path
    
    Returns:
        Path to masters_csvs directory
    """
    if override:
        return Path(override)
    
    datasets_masters = root / "datasets" / "masters_csvs"
    if datasets_masters.exists():
        return datasets_masters
    
    return root / "masters_csvs"
