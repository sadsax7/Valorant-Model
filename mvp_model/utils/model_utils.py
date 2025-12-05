"""Model loading and feature building utilities."""

from __future__ import annotations

import json
from typing import List, Tuple, Optional

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from mvp_model.utils.features import build_full_features


def load_model_and_info(
    model_path: str,
    train_info_path: str
) -> Tuple[Pipeline, dict]:
    """
    Load trained model and training info.
    
    Args:
        model_path: Path to model .pkl file
        train_info_path: Path to train_info.json file
    
    Returns:
        Tuple of (model, train_info_dict)
    """
    model = joblib.load(model_path)
    
    try:
        with open(train_info_path, "r", encoding="utf-8") as f:
            train_info = json.load(f)
    except FileNotFoundError:
        train_info = {}
    
    return model, train_info


def get_feature_names(
    train_info_path: str,
    fallback: Optional[List[str]] = None
) -> List[str]:
    """
    Get feature names from train_info.json.
    
    Args:
        train_info_path: Path to train_info.json
        fallback: Fallback feature names if file not found
    
    Returns:
        List of feature names
    """
    if fallback is None:
        fallback = ["elo1_before", "elo2_before", "elo_diff"]
    
    try:
        with open(train_info_path, "r", encoding="utf-8") as f:
            train_info = json.load(f)
        return train_info.get("features", fallback)
    except FileNotFoundError:
        return fallback


def compute_test_split(n: int, test_size: float) -> Tuple[int, int]:
    """
    Compute train/test split indices for temporal split.
    
    Args:
        n: Total number of samples
        test_size: Fraction for test set (0.0 to 1.0)
    
    Returns:
        Tuple of (train_end_index, test_start_index)
    """
    n_test = int(max(1, round(n * test_size)))
    split_idx = n - n_test
    return split_idx, split_idx


def build_features_from_data(
    df_matches: pd.DataFrame,
    df_players: pd.DataFrame,
    elo_k: float,
    elo_base: float,
    feature_names: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build features from match and player data.
    
    Args:
        df_matches: Prepared matches DataFrame
        df_players: Player statistics DataFrame
        elo_k: Elo K-factor
        elo_base: Elo base rating
        feature_names: List of feature column names to extract
    
    Returns:
        Tuple of (df_with_all_features, X_features_only)
    """
    df_all, feats = build_full_features(
        df_matches,
        df_players,
        elo_k=elo_k,
        elo_base=elo_base
    )
    
    X = feats[feature_names].copy()
    return df_all, X


def compute_test_slice(
    n: int,
    test_size: float,
    last_n: Optional[int] = None
) -> slice:
    """
    Compute slice for test set, optionally limiting to last N samples.
    
    Args:
        n: Total number of samples
        test_size: Fraction for test set
        last_n: If provided, limit to last N samples
    
    Returns:
        Slice object for indexing
    """
    n_test = int(max(1, round(n * test_size)))
    start = n - n_test
    
    if last_n is not None:
        start = max(start, n - last_n)
    
    return slice(start, n)
