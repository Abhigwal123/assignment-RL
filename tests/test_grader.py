"""
Unit tests for StockRL Mini Independent Grader and Exploit Suites.
"""

import copy
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np

from environment.trading_env import StockTradingEnv
from grader.verifier import Verifier, VerificationReport
from grader.exploit_tests import Exploit1TestManipulation, Exploit2StateAndRewardManipulation


# ==============================================================================
# 1. Independent Grader Verification on Honest Episodes
# ==============================================================================

@pytest.mark.parametrize("task_id", [1, 2, 3])
def test_verifier_honest_episode_passes(task_id: int):
    """Verify that legitimate environment runs pass independent verification across all tasks."""
    env = StockTradingEnv(task=task_id)
    env.reset(seed=100 + task_id)

    # Execute a diverse, realistic sequence of trades
    actions = [2, 2, 1, 0, 2, 1, 0, 1, 2, 0]
    for action in actions:
        env.step(action)

    history = env.get_episode_history()
    report = Verifier.verify_episode(
        history=history,
        task_config=env.config,
        market_data=env.market_data,
        initial_cash=env.initial_cash,
    )

    assert report.passed is True
    assert len(report.discrepancies) == 0
    assert report.task_id == task_id
    assert report.total_trades > 0
    assert report.max_drawdown >= 0.0

    # Ensure transaction costs are strictly tracked in tasks 2 and 3
    if task_id in [2, 3]:
        assert report.total_transaction_costs > 0.0
        assert report.total_fees > 0.0
        assert report.total_slippage > 0.0
    else:
        assert report.total_transaction_costs == 0.0


def test_verifier_drawdown_and_constraint_accounting():
    """Verify that independent replay tracks drawdowns, invalid trades, and constraint violations."""
    env = StockTradingEnv(task=3)  # max_position = 50, trade_size = 10
    env.reset(seed=42)

    # 1. Take 1 invalid sell at step 1 (0 shares held)
    env.step(StockTradingEnv.ACTION_SELL)

    # 2. Buy 6 times (5 valid buys = 50 shares, 6th buy violates max position)
    for _ in range(6):
        env.step(StockTradingEnv.ACTION_BUY)

    history = env.get_episode_history()
    report = Verifier.verify_episode(
        history=history,
        task_config=env.config,
        market_data=env.market_data,
        initial_cash=env.initial_cash,
    )

    assert report.passed is True  # The honest env reported honestly, so accounting matches
    # 2 invalid actions: 1 invalid sell (0 shares) + 1 capped unexecutable buy (0 shares due to max position limit)
    assert report.invalid_transactions == 2
    assert report.constraint_violations == 1  # 1 position constraint violation
    assert report.total_trades == 5  # Exactly 5 successful buys


# ==============================================================================
# 2. Exploit 1 Tests: Test Manipulation & Code Tampering
# ==============================================================================

def test_exploit_1_code_integrity_checksum():
    """Verify that cryptographic checksum detects unauthorized modifications to verifier.py."""
    verifier_path = Path(__file__).parent.parent / "grader" / "verifier.py"
    assert verifier_path.exists()

    expected_hash = Exploit1TestManipulation.get_verifier_hash(verifier_path)
    assert len(expected_hash) == 64  # Valid SHA-256

    # Unmodified file matches hash
    assert Exploit1TestManipulation.verify_integrity(verifier_path, expected_hash) is True

    # Tampered hash must fail
    fake_hash = "0" * 64
    assert Exploit1TestManipulation.verify_integrity(verifier_path, fake_hash) is False


def test_exploit_1_monkeypatch_demonstration():
    """Demonstrate monkeypatch exploit vulnerability and how external verification catches it."""
    history = [{"step": 0, "action": 1, "price": 1000.0, "cash": 100_000.0, "shares": 0, "portfolio_value": 100_000.0}]
    succeeded, explanation = Exploit1TestManipulation.attempt_monkeypatch_exploit(history, task_id=1)

    assert succeeded is True
    assert "Monkeypatch exploit succeeded" in explanation


# ==============================================================================
# 3. Exploit 2 Tests: State, Price, and Fee Spoofing
# ==============================================================================

def test_exploit_2_cash_balance_inflation_caught():
    """Verify that independent replay catches falsified cash balances."""
    env = StockTradingEnv(task=2)
    report = Exploit2StateAndRewardManipulation.simulate_cash_inflation_exploit(env)

    assert report.passed is False
    assert any("Cash mismatch" in d for d in report.discrepancies)
    assert any("Portfolio value mismatch" in d for d in report.discrepancies)


def test_exploit_2_price_spoofing_caught():
    """Verify that independent replay catches falsified market prices."""
    env = StockTradingEnv(task=2)
    report = Exploit2StateAndRewardManipulation.simulate_price_spoofing_exploit(env)

    assert report.passed is False
    assert any("Price spoofing detected" in d for d in report.discrepancies)


def test_exploit_2_fee_evasion_caught():
    """Verify that attempting to evade transaction costs in Task 2 is caught."""
    env = StockTradingEnv(task=2)
    report = Exploit2StateAndRewardManipulation.simulate_fee_evasion_exploit(env)

    assert report.passed is False
    assert any("Transaction cost mismatch" in d for d in report.discrepancies)


def test_verifier_empty_history_rejection():
    """Verify that an empty history fails verification immediately."""
    env = StockTradingEnv(task=1)
    report = Verifier.verify_episode([], env.config, env.market_data)

    assert report.passed is False
    assert "history is completely empty" in report.discrepancies[0]
