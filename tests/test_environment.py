"""
Unit tests for StockRL Mini Environment & Data layer.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of how pytest is invoked
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd
from gymnasium.utils.env_checker import check_env

from environment.market_data import MarketData
from environment.reward import RewardCalculator
from environment.trading_env import StockTradingEnv, TaskConfig, TASK_CONFIGS


@pytest.fixture
def data_path() -> Path:
    return Path(__file__).parent.parent / "data" / "sample_prices.csv"


# ==============================================================================
# 1. Market Data Tests
# ==============================================================================

def test_market_data_file_exists_and_labeled_synthetic(data_path: Path):
    """Ensure the CSV file exists and is clearly labeled as synthetic."""
    assert data_path.exists(), f"Sample prices file not found at {data_path}"
    with open(data_path, "r", encoding="utf-8") as f:
        content = f.read(500)
    assert "SYNTHETIC" in content.upper(), "Dataset must be clearly labeled as synthetic data."
    assert "NOT REAL MARKET DATA" in content.upper(), "Dataset must explicitly disclaim real market data."


def test_market_data_load_and_validation(data_path: Path):
    """Test loading and column schema validation."""
    md = MarketData(data_path)
    assert md.total_steps == 252
    assert md.initial_price > 0

    for col in MarketData.REQUIRED_COLUMNS:
        assert col in md.df.columns

    assert (md.df["close"] > 0).all()
    assert (md.df["open"] > 0).all()
    assert (md.df["high"] >= md.df["low"]).all()


def test_market_data_chronological_ordering(data_path: Path):
    """Verify that dates are monotonically increasing with no backwards jumps."""
    md = MarketData(data_path)
    dates = pd.to_datetime(md.df["date"])
    assert dates.is_monotonic_increasing, "Market data dates must be monotonically increasing."


def test_market_data_causality_and_no_lookahead(data_path: Path):
    """Verify that features at step t depend ONLY on information up to step t."""
    md = MarketData(data_path)
    t = 10
    features_t = md.get_market_features(t)

    expected_norm_price = float(md.df["close"].iloc[t] / md.df["close"].iloc[0])
    assert np.isclose(features_t[0], expected_norm_price, atol=1e-4)

    expected_1d_return = float((md.df["close"].iloc[t] - md.df["close"].iloc[t - 1]) / md.df["close"].iloc[t - 1])
    assert np.isclose(features_t[1], expected_1d_return, atol=1e-4)

    with pytest.raises(IndexError):
        md.get_price(md.total_steps)


# ==============================================================================
# 2. Gymnasium Compatibility & check_env
# ==============================================================================

@pytest.mark.parametrize("task_id", [1, 2, 3])
def test_gymnasium_env_checker(task_id: int):
    """Validate that environment conforms to Gymnasium specifications across all 3 tasks."""
    env = StockTradingEnv(task=task_id)
    # Gymnasium check_env raises exceptions or warnings if specifications are violated
    check_env(env.unwrapped)
    env.close()


# ==============================================================================
# 3. Environment State Transitions, Accounting, and Validation
# ==============================================================================

def test_environment_reset():
    """Verify that reset restores all initial state variables accurately."""
    env = StockTradingEnv(task=1)
    obs, info = env.reset(seed=42)

    assert env.current_step == 0
    assert env.cash == 100_000.0
    assert env.shares == 0
    assert env.portfolio_value == 100_000.0
    assert len(obs) == 8
    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32

    # Verification of initial observation
    assert np.isclose(obs[5], 1.0)  # norm_cash: 100,000 / 100,000 = 1.0
    assert np.isclose(obs[6], 0.0)  # norm_holdings: 0 shares = 0.0
    assert np.isclose(obs[7], 1.0)  # norm_portfolio_value = 1.0


def test_buy_and_sell_operations():
    """Verify proper execution of Buy and Sell orders."""
    env = StockTradingEnv(task=1)  # No fee task for exact math
    env.reset(seed=42)

    # Step 1: Execute BUY
    obs, reward, term, trunc, info = env.step(StockTradingEnv.ACTION_BUY)
    price_1 = env.market_data.get_price(1)

    assert env.shares == 10
    expected_cash = 100_000.0 - (10 * price_1)
    assert np.isclose(env.cash, expected_cash)
    assert np.isclose(env.portfolio_value, 100_000.0)  # No fees, portfolio value unchanged at execution

    # Step 2: Execute SELL
    obs, reward, term, trunc, info = env.step(StockTradingEnv.ACTION_SELL)
    price_2 = env.market_data.get_price(2)

    assert env.shares == 0
    expected_cash_after_sell = expected_cash + (10 * price_2)
    assert np.isclose(env.cash, expected_cash_after_sell)
    assert np.isclose(env.portfolio_value, expected_cash_after_sell)


def test_insufficient_cash_prevention():
    """Prevent negative cash balances when funds are exhausted."""
    env = StockTradingEnv(task=1, initial_cash=1500.0)  # Enough for ~1 share at ~₹1000
    env.reset()

    price_1 = env.market_data.get_price(1)
    # Attempting to buy 10 shares requires ~₹10,000, but we only have ₹1,500
    obs, reward, term, trunc, info = env.step(StockTradingEnv.ACTION_BUY)

    # Should buy only 1 share (max affordable), cash must remain >= 0
    assert env.shares == 1
    assert env.cash >= 0.0
    assert env.cash < price_1

    # Attempting another BUY with remaining cash (< price_2) must fail gracefully
    obs, reward, term, trunc, info = env.step(StockTradingEnv.ACTION_BUY)
    assert env.shares == 1
    assert info["is_invalid_action"] is True
    assert env.cash >= 0.0


def test_selling_with_zero_holdings():
    """Attempting to sell when holding 0 shares must be rejected without altering state."""
    env = StockTradingEnv(task=1)
    env.reset()

    initial_cash = env.cash
    obs, reward, term, trunc, info = env.step(StockTradingEnv.ACTION_SELL)

    assert env.shares == 0
    assert env.cash == initial_cash
    assert info["is_invalid_action"] is True
    assert reward <= 0.0  # Must not give free reward


def test_transaction_costs_and_slippage_comparison():
    """Verify that Task 2 deducts transaction fees and execution slippage compared to Task 1."""
    env_task1 = StockTradingEnv(task=1)
    env_task2 = StockTradingEnv(task=2)

    env_task1.reset(seed=123)
    env_task2.reset(seed=123)

    # Buy on step 1 for both
    _, _, _, _, info1 = env_task1.step(StockTradingEnv.ACTION_BUY)
    _, _, _, _, info2 = env_task2.step(StockTradingEnv.ACTION_BUY)

    # Task 2 cash should be strictly less due to slippage + brokerage fee
    assert env_task2.cash < env_task1.cash
    assert info2["step_cost"] > 0.0
    assert info1["step_cost"] == 0.0

    # Sell on step 2 for both
    _, _, _, _, info1_sell = env_task1.step(StockTradingEnv.ACTION_SELL)
    _, _, _, _, info2_sell = env_task2.step(StockTradingEnv.ACTION_SELL)

    # Net proceeds in Task 2 must be less than Task 1
    assert env_task2.portfolio_value < env_task1.portfolio_value


def test_task3_maximum_position_constraint():
    """Verify Task 3 enforces maximum position constraints."""
    env = StockTradingEnv(task=3)  # max_position_shares = 50, trade_size = 10
    env.reset()

    # Buy 5 times -> 50 shares
    for _ in range(5):
        env.step(StockTradingEnv.ACTION_BUY)

    assert env.shares == 50

    # 6th Buy attempt should exceed max position
    obs, reward, term, trunc, info = env.step(StockTradingEnv.ACTION_BUY)
    assert env.shares == 50  # Must not exceed 50
    assert info["is_constraint_violation"] is True
    assert reward < 0.0  # Position violation penalty applied


def test_portfolio_accounting_invariant():
    """Invariant test: Portfolio value = Cash + Shares * Price at every step."""
    env = StockTradingEnv(task=2)
    env.reset(seed=999)

    actions = [StockTradingEnv.ACTION_BUY, StockTradingEnv.ACTION_HOLD, StockTradingEnv.ACTION_SELL, StockTradingEnv.ACTION_BUY]
    for action in actions:
        env.step(action)
        expected_pv = env.cash + (env.shares * env.market_data.get_price(env.current_step))
        assert np.isclose(env.portfolio_value, expected_pv, atol=1e-5), "Accounting invariant failed!"


def test_reproducibility():
    """Identical seeds and action sequences must produce byte-identical results."""
    env1 = StockTradingEnv(task=2)
    env2 = StockTradingEnv(task=2)

    env1.reset(seed=777)
    env2.reset(seed=777)

    actions = [2, 2, 1, 0, 2, 1, 0, 1]
    for a in actions:
        _, r1, _, _, info1 = env1.step(a)
        _, r2, _, _, info2 = env2.step(a)
        assert np.isclose(r1, r2)
        assert np.isclose(info1["portfolio_value"], info2["portfolio_value"])
        assert env1.shares == env2.shares
        assert np.isclose(env1.cash, env2.cash)


def test_episode_termination():
    """Verify episode terminates properly at the end of the price series."""
    env = StockTradingEnv(task=1)
    env.reset()

    done = False
    step_count = 0
    while not done:
        _, _, terminated, truncated, _ = env.step(StockTradingEnv.ACTION_HOLD)
        done = terminated or truncated
        step_count += 1

    assert done is True
    assert step_count == env.market_data.total_steps - 1


# ==============================================================================
# 4. Pure Reward Calculator Tests
# ==============================================================================

def test_reward_pure_calculation():
    """Verify RewardCalculator logic in isolation."""
    # Positive return
    r_pos = RewardCalculator.calculate_reward(
        prev_portfolio_value=100_000.0,
        current_portfolio_value=102_000.0,
        initial_portfolio_value=100_000.0,
        transaction_cost=0.0,
        current_drawdown=0.0,
    )
    assert np.isclose(r_pos, 0.02)  # +2%

    # Negative return
    r_neg = RewardCalculator.calculate_reward(
        prev_portfolio_value=100_000.0,
        current_portfolio_value=98_000.0,
        initial_portfolio_value=100_000.0,
        transaction_cost=0.0,
        current_drawdown=0.02,
    )
    assert np.isclose(r_neg, -0.02)

    # Drawdown penalty
    r_dd = RewardCalculator.calculate_reward(
        prev_portfolio_value=100_000.0,
        current_portfolio_value=100_000.0,
        initial_portfolio_value=100_000.0,
        transaction_cost=0.0,
        current_drawdown=0.10,  # 10% drawdown
        drawdown_penalty_weight=0.05,
    )
    assert np.isclose(r_dd, -0.005)  # -0.05 * 0.10 = -0.005

    # Invalid action penalty
    r_inv = RewardCalculator.calculate_reward(
        prev_portfolio_value=100_000.0,
        current_portfolio_value=100_000.0,
        initial_portfolio_value=100_000.0,
        transaction_cost=0.0,
        current_drawdown=0.0,
        is_invalid_action=True,
        invalid_action_penalty=0.001,
    )
    assert np.isclose(r_inv, -0.001)
