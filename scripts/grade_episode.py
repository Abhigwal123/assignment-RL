"""
Independent grading execution script for StockRL Mini.

Executes an agent episode and subjects the resulting action trajectory
to full independent verification and invariant checking.

Usage:
    python scripts/grade_episode.py --agent sma --task 2
    python scripts/grade_episode.py --agent random --task 3
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.trading_env import StockTradingEnv, TASK_CONFIGS
from grader.verifier import Verifier
from agents.baseline_agent import RandomAgent, BuyAndHoldAgent, SMAAgent


AGENT_REGISTRY = {
    "random": RandomAgent,
    "buy_and_hold": BuyAndHoldAgent,
    "sma": SMAAgent,
}


def main():
    parser = argparse.ArgumentParser(description="Grade an agent episode using the independent verifier.")
    parser.add_argument(
        "--agent",
        type=str,
        default="sma",
        choices=["random", "buy_and_hold", "sma"],
        help="Agent to evaluate (default: sma)",
    )
    parser.add_argument(
        "--task",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Task ID to grade (default: 2)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    # Step 1: Run Episode
    env = StockTradingEnv(task=args.task)
    agent_cls = AGENT_REGISTRY[args.agent]
    agent = agent_cls(seed=args.seed) if args.agent == "random" else agent_cls()

    obs, info = env.reset(seed=args.seed)
    agent.reset()

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = agent.predict(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)

    history = env.get_episode_history()

    # Step 2: Subject to Independent Verification
    print(f"\nSubmitting episode ({agent.name} on Task {args.task}) to Independent Verifier...")
    report = Verifier.verify_episode(
        history=history,
        task_config=env.config,
        market_data=env.market_data,
        initial_cash=env.initial_cash,
    )

    print("\n" + report.summary())

    if report.passed:
        print("\n[RESULT] Verification SUCCEEDED: Zero accounting discrepancies found.")
        sys.exit(0)
    else:
        print("\n[RESULT] Verification FAILED: Discrepancies detected!")
        sys.exit(1)


if __name__ == "__main__":
    main()
