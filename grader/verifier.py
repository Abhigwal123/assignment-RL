"""
StockRL Mini - Independent Grader and Verifier.

Independently replays an episode's action history from ground-truth market data,
calculates accounting invariants from scratch, and verifies that the reported
environment trajectory has not been falsified or tampered with.
"""

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from environment.market_data import MarketData
from environment.trading_env import TaskConfig, TASK_CONFIGS, StockTradingEnv


@dataclass
class VerificationReport:
    """Detailed summary of independent verification."""
    passed: bool
    task_id: int
    initial_capital: float
    final_portfolio_value: float
    net_return: float
    total_transaction_costs: float
    total_fees: float
    total_slippage: float
    total_trades: int
    invalid_transactions: int
    constraint_violations: int
    max_drawdown: float
    discrepancies: List[str] = field(default_factory=list)
    integrity_hash: str = ""

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"=== VERIFICATION REPORT: {status} ===",
            f"Task ID: {self.task_id}",
            f"Initial Capital: ₹{self.initial_capital:,.2f}",
            f"Final Portfolio Value: ₹{self.final_portfolio_value:,.2f}",
            f"Net Return: {self.net_return * 100:.2f}%",
            f"Total Trades: {self.total_trades}",
            f"Total Transaction Costs: ₹{self.total_transaction_costs:,.2f} (Fees: ₹{self.total_fees:,.2f}, Slippage: ₹{self.total_slippage:,.2f})",
            f"Invalid Transactions: {self.invalid_transactions}",
            f"Constraint Violations: {self.constraint_violations}",
            f"Maximum Drawdown: {self.max_drawdown * 100:.2f}%",
            f"Integrity Checksum: {self.integrity_hash[:16]}...",
        ]
        if self.discrepancies:
            lines.append("Discrepancies Found:")
            for d in self.discrepancies:
                lines.append(f"  [!] {d}")
        return "\n".join(lines)


class Verifier:
    """
    Independent accounting and replay verifier.

    DOES NOT trust environment-reported portfolio values, cash, or rewards.
    Re-executes the actions against ground-truth market prices and asserts:
    1. Recorded prices match market dataset.
    2. Cash deductions and additions match execution formula.
    3. Share counts match valid buy/sell trades.
    4. Transaction costs match fee and slippage parameters.
    5. Portfolio values match Cash + Shares * Price.
    6. Position limits and invalid trades are strictly accounted for.
    """

    TOLERANCE: float = 1e-4

    @staticmethod
    def compute_file_hash(filepath: Path) -> str:
        """Compute SHA-256 hash of a file to check code integrity."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def verify_episode(
        cls,
        history: List[Dict[str, Any]],
        task_config: TaskConfig,
        market_data: MarketData,
        initial_cash: float = 100_000.0,
    ) -> VerificationReport:
        """
        Independently replay actions and cross-check the recorded trajectory.

        Parameters:
        - history: The raw recorded history from env.get_episode_history()
        - task_config: Configuration for the task being graded
        - market_data: Ground-truth market data instance
        - initial_cash: Starting capital (₹100,000)
        """
        discrepancies: List[str] = []

        if not history:
            return VerificationReport(
                passed=False,
                task_id=task_config.task_id,
                initial_capital=initial_cash,
                final_portfolio_value=0.0,
                net_return=-1.0,
                total_transaction_costs=0.0,
                total_fees=0.0,
                total_slippage=0.0,
                total_trades=0,
                invalid_transactions=0,
                constraint_violations=0,
                max_drawdown=0.0,
                discrepancies=["Episode history is completely empty."],
            )

        # Independent accounting state
        sim_cash = float(initial_cash)
        sim_shares = 0
        sim_peak_value = float(initial_cash)
        sim_max_drawdown = 0.0

        total_fees = 0.0
        total_slippage = 0.0
        total_trades = 0
        invalid_transactions = 0
        constraint_violations = 0

        # Step 0 validation
        step_0 = history[0]
        ground_truth_p0 = market_data.get_price(0)
        if not np.isclose(step_0["price"], ground_truth_p0, atol=cls.TOLERANCE):
            discrepancies.append(
                f"Step 0 price mismatch: reported {step_0['price']}, actual {ground_truth_p0}"
            )

        if not np.isclose(step_0["cash"], initial_cash, atol=cls.TOLERANCE):
            discrepancies.append(
                f"Step 0 initial cash mismatch: reported {step_0['cash']}, expected {initial_cash}"
            )

        if step_0["shares"] != 0:
            discrepancies.append(
                f"Step 0 shares mismatch: reported {step_0['shares']}, expected 0"
            )

        # Replay each step from step 1 onward
        for record in history[1:]:
            step = record["step"]
            action = record["action"]

            if step >= market_data.total_steps:
                discrepancies.append(f"Step {step} exceeds total market dataset length.")
                break

            # 1. Verify price integrity against ground truth
            ground_truth_price = market_data.get_price(step)
            if not np.isclose(record["price"], ground_truth_price, atol=cls.TOLERANCE):
                discrepancies.append(
                    f"Step {step}: Price spoofing detected! Reported {record['price']}, ground truth is {ground_truth_price}"
                )

            # 2. Independent execution replay
            trade_size = task_config.trade_size
            step_fee = 0.0
            step_slippage = 0.0
            step_shares_traded = 0
            is_invalid = False
            is_violation = False

            if action == StockTradingEnv.ACTION_BUY:
                requested_shares = trade_size

                # Position constraint check (Task 3)
                if task_config.max_position_shares is not None:
                    if sim_shares + requested_shares > task_config.max_position_shares:
                        is_violation = True
                        constraint_violations += 1
                        requested_shares = max(0, task_config.max_position_shares - sim_shares)

                if requested_shares <= 0:
                    is_invalid = True
                    invalid_transactions += 1
                else:
                    exec_price = ground_truth_price * (1.0 + task_config.slippage)
                    cost_per_share = exec_price * (1.0 + task_config.fee_rate)
                    max_affordable = int(sim_cash // cost_per_share)
                    actual_shares = min(requested_shares, max_affordable)

                    if actual_shares > 0:
                        gross_cost = actual_shares * exec_price
                        fee = gross_cost * task_config.fee_rate
                        total_cash_required = gross_cost + fee

                        sim_cash = max(0.0, sim_cash - total_cash_required)
                        sim_shares += actual_shares
                        step_shares_traded = actual_shares
                        step_fee = fee
                        step_slippage = actual_shares * ground_truth_price * task_config.slippage
                        total_trades += 1
                    else:
                        is_invalid = True
                        invalid_transactions += 1

            elif action == StockTradingEnv.ACTION_SELL:
                requested_shares = trade_size
                if sim_shares <= 0:
                    is_invalid = True
                    invalid_transactions += 1
                else:
                    actual_shares = min(requested_shares, sim_shares)
                    exec_price = ground_truth_price * (1.0 - task_config.slippage)
                    gross_proceeds = actual_shares * exec_price
                    fee = gross_proceeds * task_config.fee_rate
                    net_proceeds = gross_proceeds - fee

                    sim_cash += net_proceeds
                    sim_shares -= actual_shares
                    step_shares_traded = actual_shares
                    step_fee = fee
                    step_slippage = actual_shares * ground_truth_price * task_config.slippage
                    total_trades += 1

            elif action == StockTradingEnv.ACTION_HOLD:
                pass
            else:
                discrepancies.append(f"Step {step}: Illegal action code {action}")

            total_fees += step_fee
            total_slippage += step_slippage

            # 3. Independent valuation
            sim_portfolio_value = sim_cash + (sim_shares * ground_truth_price)

            # Drawdown update
            if sim_portfolio_value > sim_peak_value:
                sim_peak_value = sim_portfolio_value
            cur_dd = (sim_peak_value - sim_portfolio_value) / sim_peak_value if sim_peak_value > 0 else 0.0
            sim_max_drawdown = max(sim_max_drawdown, cur_dd)

            # 4. Cross-check against reported record
            if not np.isclose(record["cash"], sim_cash, atol=cls.TOLERANCE):
                discrepancies.append(
                    f"Step {step}: Cash mismatch! Reported {record['cash']:.2f}, replayed {sim_cash:.2f}"
                )

            if record["shares"] != sim_shares:
                discrepancies.append(
                    f"Step {step}: Holdings mismatch! Reported {record['shares']}, replayed {sim_shares}"
                )

            if not np.isclose(record["portfolio_value"], sim_portfolio_value, atol=cls.TOLERANCE):
                discrepancies.append(
                    f"Step {step}: Portfolio value mismatch! Reported {record['portfolio_value']:.2f}, replayed {sim_portfolio_value:.2f}"
                )

            reported_cost = record.get("step_cost", 0.0)
            replayed_cost = step_fee + step_slippage
            if not np.isclose(reported_cost, replayed_cost, atol=cls.TOLERANCE):
                discrepancies.append(
                    f"Step {step}: Transaction cost mismatch! Reported {reported_cost:.2f}, replayed {replayed_cost:.2f}"
                )

        final_portfolio_value = sim_cash + (sim_shares * market_data.get_price(len(history) - 1))
        net_return = (final_portfolio_value - initial_cash) / initial_cash
        total_costs = total_fees + total_slippage

        # Cryptographic checksum of verified trace
        trace_str = f"{task_config.task_id}_{final_portfolio_value:.4f}_{net_return:.6f}_{total_costs:.4f}_{sim_max_drawdown:.6f}"
        integrity_hash = hashlib.sha256(trace_str.encode("utf-8")).hexdigest()

        passed = len(discrepancies) == 0

        return VerificationReport(
            passed=passed,
            task_id=task_config.task_id,
            initial_capital=initial_cash,
            final_portfolio_value=final_portfolio_value,
            net_return=net_return,
            total_transaction_costs=total_costs,
            total_fees=total_fees,
            total_slippage=total_slippage,
            total_trades=total_trades,
            invalid_transactions=invalid_transactions,
            constraint_violations=constraint_violations,
            max_drawdown=sim_max_drawdown,
            discrepancies=discrepancies,
            integrity_hash=integrity_hash,
        )
