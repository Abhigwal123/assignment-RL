"""
Execution script to run trading agents on StockRL Mini tasks.

Usage:
    python scripts/run_agent.py --agent all --task all
    python scripts/run_agent.py --agent sma --task 2
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Type

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.trading_env import StockTradingEnv, TASK_CONFIGS
from agents.baseline_agent import BaseAgent, RandomAgent, BuyAndHoldAgent, SMAAgent


AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {
    "random": RandomAgent,
    "buy_and_hold": BuyAndHoldAgent,
    "sma": SMAAgent,
}


def run_single_episode(agent: BaseAgent, task_id: int, seed: int = 42) -> Dict[str, Any]:
    """Execute one full episode and return key performance statistics."""
    env = StockTradingEnv(task=task_id)
    obs, info = env.reset(seed=seed)
    agent.reset()

    terminated = False
    truncated = False
    total_reward = 0.0
    step_count = 0

    while not (terminated or truncated):
        action = agent.predict(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

    history = env.get_episode_history()
    final_record = history[-1]
    initial_cash = env.initial_cash
    final_pv = final_record["portfolio_value"]
    net_return = (final_pv - initial_cash) / initial_cash

    total_costs = sum(h.get("step_cost", 0.0) for h in history)
    total_trades = sum(1 for h in history if h.get("shares_traded", 0) > 0)
    max_drawdown = env.max_drawdown

    return {
        "agent": agent.name,
        "task_id": task_id,
        "task_name": env.config.name,
        "initial_capital": initial_cash,
        "final_portfolio_value": final_pv,
        "net_return_pct": net_return * 100.0,
        "total_reward": total_reward,
        "total_trades": total_trades,
        "total_costs": total_costs,
        "max_drawdown_pct": max_drawdown * 100.0,
        "steps": step_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Run baseline agents on StockRL Mini tasks.")
    parser.add_argument(
        "--agent",
        type=str,
        default="all",
        choices=["all", "random", "buy_and_hold", "sma"],
        help="Agent to run (default: all)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["all", "1", "2", "3"],
        help="Task ID to evaluate on (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for environment and agent (default: 42)")
    args = parser.parse_args()

    agents_to_run = list(AGENT_REGISTRY.keys()) if args.agent == "all" else [args.agent]
    tasks_to_run = [1, 2, 3] if args.task == "all" else [int(args.task)]

    print("=" * 88)
    print(f" StockRL Mini - Benchmark Agent Execution (Seed: {args.seed})")
    print("=" * 88)

    results: List[Dict[str, Any]] = []

    for task_id in tasks_to_run:
        print(f"\n--- {TASK_CONFIGS[task_id].name} ---")
        for agent_key in agents_to_run:
            agent_cls = AGENT_REGISTRY[agent_key]
            agent = agent_cls(seed=args.seed) if agent_key == "random" else agent_cls()
            res = run_single_episode(agent, task_id, seed=args.seed)
            results.append(res)
            print(
                f"  [{res['agent']:<22}] "
                f"Final Value: ₹{res['final_portfolio_value']:>10,.2f} | "
                f"Net Return: {res['net_return_pct']:>6.2f}% | "
                f"Trades: {res['total_trades']:>3} | "
                f"Costs: ₹{res['total_costs']:>8.2f} | "
                f"Max DD: {res['max_drawdown_pct']:>5.2f}%"
            )

    print("\n" + "=" * 88)
    print(" SUMMARY BENCHMARK TABLE")
    print("=" * 88)
    print(f"{'Task':<6} {'Agent':<24} {'Final Value (₹)':<16} {'Net Return':<12} {'Trades':<8} {'Costs (₹)':<12} {'Max DD':<8}")
    print("-" * 88)
    for r in results:
        print(
            f"Task {r['task_id']:<1} "
            f"{r['agent']:<24} "
            f"₹{r['final_portfolio_value']:>13,.2f}  "
            f"{r['net_return_pct']:>9.2f}%  "
            f"{r['total_trades']:>6}  "
            f"₹{r['total_costs']:>9.2f}  "
            f"{r['max_drawdown_pct']:>6.2f}%"
        )
    print("=" * 88)


if __name__ == "__main__":
    main()
