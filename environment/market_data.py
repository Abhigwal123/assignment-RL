"""
Market Data module for StockRL Mini.

Loads and pre-processes financial time-series data from CSV.
Guarantees strictly causal feature calculation (no lookahead bias).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class MarketData:
    """
    Encapsulates chronological market price sequences and provides
    causal observation features for the RL environment.
    """

    REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]

    def __init__(self, csv_path: Union[str, Path]):
        self.csv_path = Path(csv_path)
        self.df: pd.DataFrame = self._load_and_validate()
        self._precompute_features()

    def _load_and_validate(self) -> pd.DataFrame:
        """Loads CSV data, validates columns, data types, and chronological consistency."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Market data file not found at: {self.csv_path}")

        # Support comment lines to read synthetic metadata header
        df = pd.read_csv(self.csv_path, comment="#")

        # Validate column requirements
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        if len(df) < 30:
            raise ValueError(f"Dataset has {len(df)} rows, minimum required is 30 for indicators.")

        # Check for nulls
        if df[self.REQUIRED_COLUMNS].isnull().any().any():
            raise ValueError("Market data contains missing or NaN values.")

        # Validate price positivity
        for col in ["open", "high", "low", "close"]:
            if (df[col] <= 0).any():
                raise ValueError(f"Found non-positive price values in '{col}' column.")

        # Validate high/low bounds
        if (df["high"] < df["low"]).any():
            raise ValueError("Found high prices lower than low prices.")

        # Validate chronological order
        df["date"] = pd.to_datetime(df["date"])
        if not df["date"].is_monotonic_increasing:
            raise ValueError("Market data is not in strict chronological order.")

        return df.reset_index(drop=True)

    def _precompute_features(self) -> None:
        """
        Precompute causal features.
        All rolling windows use historical data only (no lookahead).
        """
        close = self.df["close"]
        self.initial_price: float = float(close.iloc[0])

        # Normalized price relative to initial price P0
        self.norm_price: np.ndarray = (close / self.initial_price).to_numpy(dtype=np.float32)

        # 1-step returns (first step return is 0.0)
        pct_change = close.pct_change().fillna(0.0)
        self.returns_1d: np.ndarray = pct_change.to_numpy(dtype=np.float32)

        # 5-step returns
        pct_change_5d = close.pct_change(periods=5).fillna(0.0)
        self.returns_5d: np.ndarray = pct_change_5d.to_numpy(dtype=np.float32)

        # 5-step Moving Average (relative difference: (P_t - SMA_5) / SMA_5)
        sma_5 = close.rolling(window=5, min_periods=1).mean()
        self.sma_5_ratio: np.ndarray = ((close - sma_5) / sma_5).to_numpy(dtype=np.float32)

        # 20-step Moving Average (relative difference: (P_t - SMA_20) / SMA_20)
        sma_20 = close.rolling(window=20, min_periods=1).mean()
        self.sma_20_ratio: np.ndarray = ((close - sma_20) / sma_20).to_numpy(dtype=np.float32)

        self.close_prices: np.ndarray = close.to_numpy(dtype=np.float64)

    @property
    def total_steps(self) -> int:
        """Total number of available time steps."""
        return len(self.df)

    def get_price(self, step: int) -> float:
        """
        Get the closing stock price at a specific step.
        Guarantees causal access within bounds.
        """
        if step < 0 or step >= self.total_steps:
            raise IndexError(f"Step {step} out of bounds (0 to {self.total_steps - 1})")
        return float(self.close_prices[step])

    def get_market_features(self, step: int) -> np.ndarray:
        """
        Get the vector of market observation features at step t:
        [
            normalized_price,   # P_t / P_0
            1d_return,          # (P_t - P_{t-1}) / P_{t-1}
            5d_return,          # (P_t - P_{t-5}) / P_{t-5}
            sma_5_ratio,        # (P_t - SMA_5) / SMA_5
            sma_20_ratio        # (P_t - SMA_20) / SMA_20
        ]
        Strictly causal: does not reference any step > t.
        """
        if step < 0 or step >= self.total_steps:
            raise IndexError(f"Step {step} out of bounds (0 to {self.total_steps - 1})")

        return np.array([
            self.norm_price[step],
            self.returns_1d[step],
            self.returns_5d[step],
            self.sma_5_ratio[step],
            self.sma_20_ratio[step],
        ], dtype=np.float32)
