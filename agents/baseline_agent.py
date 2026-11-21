"""
Baseline Trading Agents for StockRL Mini.

Implements three standard baseline strategies:
1. RandomAgent: Randomly selects actions (Buy, Hold, Sell).
2. BuyAndHoldAgent: Invests initial cash into stock and holds until termination.
3. SMAAgent: Simple Moving Average crossover strategy (SMA-5 vs SMA-20).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class BaseAgent(ABC):
    """Abstract base class for all trading agents."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def predict(self, observation: np.ndarray, info: Optional[Dict[str, Any]] = None) -> int:
        """
        Choose an action given the current observation.

        Actions:
            0 = Sell
            1 = Hold
            2 = Buy
        """
        pass

    def reset(self) -> None:
        """Reset agent internal state for a new episode."""
        pass


class RandomAgent(BaseAgent):
    """Selects actions uniformly at random with a reproducible seed."""

    def __init__(self, seed: Optional[int] = 42):
        super().__init__(name="Random Agent")
        self.rng = np.random.default_rng(seed)

    def predict(self, observation: np.ndarray, info: Optional[Dict[str, Any]] = None) -> int:
        return int(self.rng.integers(0, 3))

    def reset(self) -> None:
        pass


class BuyAndHoldAgent(BaseAgent):
    """
    Buys shares until cash is exhausted or position limit reached,
    then holds for the remainder of the episode.
    """

    def __init__(self):
        super().__init__(name="Buy and Hold Agent")
        self.fully_invested = False

    def reset(self) -> None:
        self.fully_invested = False

    def predict(self, observation: np.ndarray, info: Optional[Dict[str, Any]] = None) -> int:
        # observation[5] is normalized cash (cash / initial_cash)
        norm_cash = float(observation[5])

        # If we have cash remaining (>1% of initial capital), continue buying
        if not self.fully_invested and norm_cash > 0.01:
            if info and info.get("is_invalid_action", False):
                self.fully_invested = True
                return 1  # Hold
            if info and info.get("is_constraint_violation", False):
                self.fully_invested = True
                return 1  # Hold
            return 2  # Buy

        self.fully_invested = True
        return 1  # Hold


class SMAAgent(BaseAgent):
    """
    Moving Average Crossover Agent.

    Uses 5-step SMA (fast) and 20-step SMA (slow):
    - Golden Cross (Fast > Slow): Bullish signal -> BUY (2)
    - Death Cross (Fast < Slow): Bearish signal -> SELL (0)
    - Otherwise: HOLD (1)
    """

    def __init__(self):
        super().__init__(name="SMA Crossover Agent")

    def reset(self) -> None:
        pass

    def predict(self, observation: np.ndarray, info: Optional[Dict[str, Any]] = None) -> int:
        # observation[3] is sma_5_ratio: (Price - SMA5) / SMA5
        # observation[4] is sma_20_ratio: (Price - SMA20) / SMA20
        # When SMA5 > SMA20, (Price / SMA5) < (Price / SMA20) => sma_5_ratio < sma_20_ratio
        sma_5_ratio = float(observation[3])
        sma_20_ratio = float(observation[4])

        # Difference indicates relative positioning of SMA5 vs SMA20
        # Positive spread means SMA5 is above SMA20 (bullish)
        fast_minus_slow_spread = sma_20_ratio - sma_5_ratio

        if fast_minus_slow_spread > 0.005:
            return 2  # BUY
        elif fast_minus_slow_spread < -0.005:
            return 0  # SELL
        else:
            return 1  # HOLD
