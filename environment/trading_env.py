"""
StockRL Mini - Trading Gymnasium Environment.

Implements a single-stock trading simulator with three standardized tasks:
1. Basic Trading (zero fees)
2. Trading with Costs (fees + slippage)
3. Risk-Aware Trading (fees + slippage + max position + drawdown tracking)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .market_data import MarketData
from .reward import RewardCalculator


@dataclass
class TaskConfig:
    """Configuration for a specific trading task."""
    task_id: int
    name: str
    fee_rate: float = 0.0               # Brokerage / transaction fee (fraction of turnover)
    slippage: float = 0.0               # Execution slippage (fraction of stock price)
    max_position_shares: Optional[int] = None  # Max shares permitted (Task 3 constraint)
    drawdown_penalty_weight: float = 0.0       # Weight for portfolio drawdown penalty
    invalid_action_penalty: float = 0.0        # Penalty for invalid action (e.g. buying with ₹0)
    constraint_violation_penalty: float = 0.0  # Penalty for exceeding position limit
    trade_size: int = 10                       # Shares traded per buy/sell action


# Task Presets
TASK_CONFIGS: Dict[int, TaskConfig] = {
    1: TaskConfig(
        task_id=1,
        name="Task 1 — Basic Trading",
        fee_rate=0.0,
        slippage=0.0,
        max_position_shares=None,
        drawdown_penalty_weight=0.0,
        invalid_action_penalty=0.0,
        constraint_violation_penalty=0.0,
        trade_size=10,
    ),
    2: TaskConfig(
        task_id=2,
        name="Task 2 — Trading with Transaction Costs",
        fee_rate=0.001,             # 0.10% transaction fee
        slippage=0.0005,            # 0.05% slippage (5 bps)
        max_position_shares=None,
        drawdown_penalty_weight=0.0,
        invalid_action_penalty=0.001,
        constraint_violation_penalty=0.0,
        trade_size=10,
    ),
    3: TaskConfig(
        task_id=3,
        name="Task 3 — Risk-Aware Trading",
        fee_rate=0.001,             # 0.10% transaction fee
        slippage=0.0005,            # 0.05% slippage
        max_position_shares=50,     # Max position constraint (e.g., 50 shares)
        drawdown_penalty_weight=0.05,  # Penalize drawdowns
        invalid_action_penalty=0.001,
        constraint_violation_penalty=0.01,
        trade_size=10,
    ),
}


class StockTradingEnv(gym.Env):
    """
    Gymnasium Trading Environment for single-stock algorithmic trading.

    Actions:
        0: Sell
        1: Hold
        2: Buy

    Observations (8-dimensional float32 vector):
        [0]: normalized_stock_price  (P_t / P_0)
        [1]: return_1d               ((P_t - P_{t-1}) / P_{t-1})
        [2]: return_5d               ((P_t - P_{t-5}) / P_{t-5})
        [3]: sma_5_ratio             ((P_t - SMA_5) / SMA_5)
        [4]: sma_20_ratio            ((P_t - SMA_20) / SMA_20)
        [5]: normalized_cash         (cash / initial_cash)
        [6]: normalized_holdings     ((holdings * P_t) / initial_cash)
        [7]: normalized_portfolio    (portfolio_value / initial_cash)
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    # Action constants
    ACTION_SELL = 0
    ACTION_HOLD = 1
    ACTION_BUY = 2

    def __init__(
        self,
        task: Union[int, TaskConfig] = 1,
        csv_path: Optional[Union[str, Path]] = None,
        initial_cash: float = 100_000.0,
    ):
        super().__init__()

        # Resolve task configuration
        if isinstance(task, int):
            if task not in TASK_CONFIGS:
                raise ValueError(f"Unknown task {task}. Must be 1, 2, or 3.")
            self.config = TASK_CONFIGS[task]
        elif isinstance(task, TaskConfig):
            self.config = task
        else:
            raise TypeError("task must be an int (1, 2, 3) or TaskConfig instance.")

        # Resolve market data path
        if csv_path is None:
            csv_path = Path(__file__).parent.parent / "data" / "sample_prices.csv"
        self.market_data = MarketData(csv_path)

        self.initial_cash = float(initial_cash)
        self.trade_size = self.config.trade_size

        # Define Gym Spaces
        self.action_space = spaces.Discrete(3)
        # 8 normalized continuous observation features bounded in [-1000.0, 1000.0]
        self.observation_space = spaces.Box(
            low=-1000.0,
            high=1000.0,
            shape=(8,),
            dtype=np.float32,
        )

        # State variables
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares = 0
        self.portfolio_value = self.initial_cash
        self.prev_portfolio_value = self.initial_cash
        self.peak_portfolio_value = self.initial_cash
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0

        # Detailed episode audit trail for the independent grader
        self.episode_history: List[Dict[str, Any]] = []

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        self.current_step = 0
        self.cash = float(self.initial_cash)
        self.shares = 0
        self.portfolio_value = float(self.initial_cash)
        self.prev_portfolio_value = float(self.initial_cash)
        self.peak_portfolio_value = float(self.initial_cash)
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0

        current_price = self.market_data.get_price(0)
        self.episode_history = [
            {
                "step": 0,
                "action": self.ACTION_HOLD,
                "action_name": "HOLD",
                "price": current_price,
                "execution_price": current_price,
                "cash": self.cash,
                "shares": self.shares,
                "portfolio_value": self.portfolio_value,
                "step_cost": 0.0,
                "fee": 0.0,
                "slippage": 0.0,
                "shares_traded": 0,
                "is_invalid": False,
                "is_violation": False,
                "drawdown": 0.0,
                "reward": 0.0,
            }
        ]

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one trading step in the environment.

        Parameters:
        - action: 0 (Sell), 1 (Hold), 2 (Buy)
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be 0, 1, or 2.")

        # Advance to current timestep
        self.current_step += 1
        current_price = self.market_data.get_price(self.current_step)

        # Track pre-action portfolio value
        self.prev_portfolio_value = self.portfolio_value

        # Execution tracking variables
        shares_traded = 0
        step_fee = 0.0
        step_slippage = 0.0
        is_invalid_action = False
        is_constraint_violation = False
        action_name = "HOLD"
        exec_price = current_price

        if action == self.ACTION_BUY:
            action_name = "BUY"
            requested_shares = self.trade_size

            # Check position constraint for Task 3
            if self.config.max_position_shares is not None:
                if self.shares + requested_shares > self.config.max_position_shares:
                    is_constraint_violation = True
                    # Cap trade to maximum allowed position
                    requested_shares = max(0, self.config.max_position_shares - self.shares)

            if requested_shares <= 0:
                is_invalid_action = True
            else:
                # Slippage increases buy execution price
                exec_price = current_price * (1.0 + self.config.slippage)
                cost_per_share_with_fee = exec_price * (1.0 + self.config.fee_rate)

                # Maximum integer shares affordable with available cash
                max_affordable = int(self.cash // cost_per_share_with_fee)
                actual_shares = min(requested_shares, max_affordable)

                if actual_shares > 0:
                    gross_cost = actual_shares * exec_price
                    fee = gross_cost * self.config.fee_rate
                    total_cash_required = gross_cost + fee

                    # Safe cash deduction preventing negative balance
                    self.cash = max(0.0, self.cash - total_cash_required)
                    self.shares += actual_shares
                    shares_traded = actual_shares
                    step_fee = fee
                    step_slippage = actual_shares * current_price * self.config.slippage
                else:
                    is_invalid_action = True

        elif action == self.ACTION_SELL:
            action_name = "SELL"
            requested_shares = self.trade_size

            if self.shares <= 0:
                is_invalid_action = True
            else:
                actual_shares = min(requested_shares, self.shares)
                # Slippage decreases sell execution price
                exec_price = current_price * (1.0 - self.config.slippage)
                gross_proceeds = actual_shares * exec_price
                fee = gross_proceeds * self.config.fee_rate
                net_proceeds = gross_proceeds - fee

                self.cash += net_proceeds
                self.shares -= actual_shares
                shares_traded = actual_shares
                step_fee = fee
                step_slippage = actual_shares * current_price * self.config.slippage

        elif action == self.ACTION_HOLD:
            action_name = "HOLD"
            # No transactions

        total_step_cost = step_fee + step_slippage

        # Update portfolio valuation
        self.portfolio_value = self.cash + (self.shares * current_price)

        # Drawdown tracking
        if self.portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = self.portfolio_value

        self.current_drawdown = (
            (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
            if self.peak_portfolio_value > 0
            else 0.0
        )
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

        # Compute pure reward
        reward = RewardCalculator.calculate_reward(
            prev_portfolio_value=self.prev_portfolio_value,
            current_portfolio_value=self.portfolio_value,
            initial_portfolio_value=self.initial_cash,
            transaction_cost=total_step_cost,
            current_drawdown=self.current_drawdown,
            is_invalid_action=is_invalid_action,
            is_constraint_violation=is_constraint_violation,
            drawdown_penalty_weight=self.config.drawdown_penalty_weight,
            invalid_action_penalty=self.config.invalid_action_penalty,
            constraint_violation_penalty=self.config.constraint_violation_penalty,
        )

        # Audit trail logging
        self.episode_history.append(
            {
                "step": self.current_step,
                "action": int(action),
                "action_name": action_name,
                "price": current_price,
                "execution_price": exec_price,
                "cash": self.cash,
                "shares": self.shares,
                "portfolio_value": self.portfolio_value,
                "step_cost": total_step_cost,
                "fee": step_fee,
                "slippage": step_slippage,
                "shares_traded": shares_traded,
                "is_invalid": is_invalid_action,
                "is_violation": is_constraint_violation,
                "drawdown": self.current_drawdown,
                "reward": float(reward),
            }
        )

        # Episode termination: end of dataset or bankruptcy
        terminated = (self.current_step >= self.market_data.total_steps - 1) or (self.portfolio_value <= 0.0)
        truncated = False

        obs = self._get_observation()
        info = self._get_info()
        info.update({
            "step_reward": reward,
            "step_cost": total_step_cost,
            "shares_traded": shares_traded,
            "is_invalid_action": is_invalid_action,
            "is_constraint_violation": is_constraint_violation,
        })

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Construct causal observation vector at current step.
        Guaranteed to contain only current and past information.
        """
        market_feats = self.market_data.get_market_features(self.current_step)
        current_price = self.market_data.get_price(self.current_step)

        # Normalized account figures
        norm_cash = np.float32(self.cash / self.initial_cash)
        norm_holdings = np.float32((self.shares * current_price) / self.initial_cash)
        norm_portfolio = np.float32(self.portfolio_value / self.initial_cash)

        obs = np.concatenate(
            [market_feats, [norm_cash, norm_holdings, norm_portfolio]],
            dtype=np.float32,
        )
        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Provides human-readable diagnostic metadata."""
        current_price = self.market_data.get_price(self.current_step)
        net_return = (self.portfolio_value - self.initial_cash) / self.initial_cash
        return {
            "step": self.current_step,
            "price": current_price,
            "cash": self.cash,
            "shares": self.shares,
            "portfolio_value": self.portfolio_value,
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown,
            "net_return": net_return,
            "task_id": self.config.task_id,
            "task_name": self.config.name,
        }

    def get_episode_history(self) -> List[Dict[str, Any]]:
        """Return the immutable copy of the episode history for independent verification."""
        return list(self.episode_history)
