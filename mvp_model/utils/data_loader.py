"""Data loading and preparation utilities for MVP model."""

from __future__ import annotations

from typing import List

import pandas as pd


def load_matches_csv(csv_path: str, filter_completed: bool = True) -> pd.DataFrame:
    """
    Load matches CSV and optionally filter for completed matches.
    
    Args:
        csv_path: Path to matches.csv file
        filter_completed: If True, only keep matches with status='completed'
    
    Returns:
        DataFrame with matches data
    """
    df = pd.read_csv(csv_path)
    
    if filter_completed and "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "completed"].copy()
    
    return df


def load_player_stats_csv(csv_path: str) -> pd.DataFrame:
    """
    Load player statistics CSV.
    
    Args:
        csv_path: Path to detailed_matches_player_stats.csv file
    
    Returns:
        DataFrame with player statistics
    """
    try:
        return pd.read_csv(csv_path)
    except FileNotFoundError:
        # Return empty DataFrame with expected columns if file doesn't exist
        return pd.DataFrame(columns=["match_id", "player_team", "acs", "kast"])


def prepare_match_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare match data: parse dates, clean strings, create labels.
    
    Args:
        df: Raw matches DataFrame
    
    Returns:
        Prepared DataFrame with parsed_date and team1_win columns
    """
    df = df.copy()
    
    # Parse date column
    if "date" in df.columns:
        df["parsed_date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["parsed_date"] = pd.NaT
    
    # Clean team and winner columns
    for col in ["team1", "team2", "winner"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    # Create binary label: 1 if team1 wins, 0 otherwise
    if "winner" in df.columns and "team1" in df.columns:
        df["team1_win"] = (df["winner"] == df["team1"]).astype(int)
    else:
        raise ValueError("CSV must contain columns: team1, winner")
    
    # Sort chronologically
    if "match_id" in df.columns:
        df = df.sort_values(["parsed_date", "match_id"], kind="stable")
    else:
        df = df.sort_values(["parsed_date"], kind="stable")
    
    df = df.reset_index(drop=True)
    return df


def validate_csv_columns(df: pd.DataFrame, required_cols: List[str], name: str = "CSV") -> None:
    """
    Validate that DataFrame contains required columns.
    
    Args:
        df: DataFrame to validate
        required_cols: List of required column names
        name: Name of the CSV for error messages
    
    Raises:
        ValueError: If any required columns are missing
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_and_prepare_matches(
    csv_path: str,
    filter_completed: bool = True,
    validate: bool = True
) -> pd.DataFrame:
    """
    Load and prepare matches in one step.
    
    Args:
        csv_path: Path to matches.csv
        filter_completed: Whether to filter for completed matches
        validate: Whether to validate required columns
    
    Returns:
        Prepared matches DataFrame
    """
    df = load_matches_csv(csv_path, filter_completed=filter_completed)
    
    if validate:
        validate_csv_columns(df, ["team1", "team2", "winner"], "matches.csv")
    
    df = prepare_match_data(df)
    return df
