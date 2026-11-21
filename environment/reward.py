"""
Reward calculation module for StockRL Mini.

Provides pure, testable, and documented reward functions for:
- Task 1: Basic Trading (Step percentage return)
- Task 2: Trading with Costs (Net return after slippage and brokerage fees)
- Task 3: Risk-Aware Trading (Net return penalized by portfolio drawdown and position limits)
"""

from typing import Dict, Any


class RewardCalculator:
    """
    Computes step rewards strictly from portfolio state transitions and penalties.
    Guarantees:
    - No access to future market data.
    - Zero or negative reward for invalid actions, churn, or constraint violations.
    - Deterministic and testable computation.
    """

    @staticmethod
    def calculate_reward(
        prev_portfolio_value: float,
        current_portfolio_value: float,
        initial_portfolio_value: float,
        transaction_cost: float,
        current_drawdown: float,
        is_invalid_action: bool = False,
        is_constraint_violation: bool = False,
        drawdown_penalty_weight: float = 0.0,
        invalid_action_penalty: float = 0.0,
        constraint_violation_penalty: float = 0.0,
    ) -> float:
        """
        Calculate step reward based on net portfolio change and risk penalties.

        Parameters:
        - prev_portfolio_value: Portfolio value at step t-1
        - current_portfolio_value: Portfolio value at step t (already reflects costs)
        - initial_portfolio_value: Starting cash (e.g. ₹100,000)
        - transaction_cost: Total fees and slippage incurred during this step
        - current_drawdown: Fractional drawdown from peak portfolio value (0.0 to 1.0)
        - is_invalid_action: True if action was illegal (e.g., buying with ₹0 cash)
        - is_constraint_violation: True if action exceeded position limit
        - drawdown_penalty_weight: Multiplier for drawdown penalty
        - invalid_action_penalty: Fixed deduction for invalid action
        - constraint_violation_penalty: Fixed deduction for exceeding position limit

        Returns:
        - reward: float
        """
        if prev_portfolio_value <= 0.0:
            return 0.0

        # Step rate of return: (V_t - V_{t-1}) / V_{t-1}
        step_return = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value

        # Exploitation prevention: Invalid actions or constraint violations
        # cannot capture free positive market gains.
        if is_invalid_action or is_constraint_violation:
            step_return = min(0.0, step_return)

        reward = float(step_return)

        # Risk-aware drawdown penalty: encourages maintaining high-water mark
        if drawdown_penalty_weight > 0.0 and current_drawdown > 0.0:
            reward -= float(drawdown_penalty_weight * current_drawdown)

        # Penalty for invalid action (prevents free exploration on unexecutable trades)
        if is_invalid_action:
            reward -= float(invalid_action_penalty)

        # Penalty for position constraint violation
        if is_constraint_violation:
            reward -= float(constraint_violation_penalty)

        return float(reward)
