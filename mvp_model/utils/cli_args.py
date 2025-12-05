"""Common CLI argument definitions for MVP model scripts."""

from __future__ import annotations

import argparse


def add_common_data_args(parser: argparse.ArgumentParser) -> None:
    """
    Add common data-related arguments.
    
    Args:
        parser: ArgumentParser to add arguments to
    """
    parser.add_argument(
        "--csv-path",
        default="masters_csvs/matches.csv",
        help="Path to matches.csv"
    )
    parser.add_argument(
        "--players-stats-path",
        default="masters_csvs/detailed_matches_player_stats.csv",
        help="Path to detailed_matches_player_stats.csv"
    )


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    """
    Add common model-related arguments.
    
    Args:
        parser: ArgumentParser to add arguments to
    """
    parser.add_argument(
        "--model",
        default="mvp_model/artifacts/model.pkl",
        help="Path to trained model .pkl"
    )
    parser.add_argument(
        "--train-info",
        default="mvp_model/artifacts/train_info.json",
        help="Path to train_info.json to align feature columns"
    )


def add_common_elo_args(parser: argparse.ArgumentParser) -> None:
    """
    Add common Elo-related arguments.
    
    Args:
        parser: ArgumentParser to add arguments to
    """
    parser.add_argument(
        "--elo-k",
        type=float,
        default=32.0,
        help="Elo K-factor (must match training)"
    )
    parser.add_argument(
        "--elo-base",
        type=float,
        default=1500.0,
        help="Elo base rating (must match training)"
    )


def add_test_args(parser: argparse.ArgumentParser) -> None:
    """
    Add test set related arguments.
    
    Args:
        parser: ArgumentParser to add arguments to
    """
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of tail for test (time split)"
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=None,
        help="Limit to last N test matches"
    )
    parser.add_argument(
        "--all-test",
        action="store_true",
        help="Use entire test block (ignore --last-n)"
    )
