# StockRL Mini: Environment Scope & Research Value

## 1. Project Objective and Environment Overview
StockRL Mini is a minimal, reproducible, and verifiable single-stock trading environment implemented using Gymnasium and Python 3.11. Designed specifically for benchmarking reinforcement learning (RL) agents under realistic financial constraints, the environment models discrete market decision-making over chronological price sequences without lookahead bias.

The simulation initializes an agent with ₹100,000 in simulated cash. The observation vector is an 8-dimensional normalized continuous state consisting of:
1. Normalized stock price ($P_t / P_0$)
2. 1-day rate of return ($R_t$)
3. 5-day rate of return ($R_{5, t}$)
4. 5-day Simple Moving Average relative ratio ($(P_t - \text{SMA}_5) / \text{SMA}_5$)
5. 20-day Simple Moving Average relative ratio ($(P_t - \text{SMA}_{20}) / \text{SMA}_{20}$)
6. Normalized cash balance ($\text{Cash}_t / V_0$)
7. Normalized equity holding value ($(\text{Shares}_t \times P_t) / V_0$)
8. Normalized total portfolio valuation ($V_t / V_0$)

Actions are discrete: `0 = Sell`, `1 = Hold`, and `2 = Buy` (executing 10 shares per order). The environment enforces mathematical invariants: negative cash balances and short selling are strictly barred, ensuring that illegal orders are rejected and penalized without altering portfolio holdings.

## 2. Three Progressive Benchmark Tasks & Success Criteria

The environment evaluates agents across three difficulty tiers:

* **Task 1 — Basic Trading (Frictionless):**
  * *Parameters:* Zero transaction fees, zero slippage, unconstrained position.
  * *Success Criteria:* Maximize final portfolio value ($V_T > V_0$) by timing directional trends.
* **Task 2 — Trading with Transaction Costs (Friction):**
  * *Parameters:* $0.10\%$ brokerage fee per trade, $0.05\%$ execution slippage (5 bps).
  * *Success Criteria:* Maximize net portfolio value after friction. Penalizes trade churning, forcing the agent to balance trade frequency against signal conviction.
* **Task 3 — Risk-Aware Trading (Constrained Risk):**
  * *Parameters:* $0.10\%$ fee, $0.05\%$ slippage, a strict 50-share position ceiling ($50\%$ max equity exposure), and high-water mark drawdown tracking.
  * *Success Criteria:* Maximize risk-adjusted return (analogous to Calmar/Sortino optimization) while strictly maintaining position constraints and minimizing portfolio drawdowns ($D_t \le 10\%$).

## 3. Why StockRL Mini is Useful for RL Research

1. **Strict Causality and Zero Lookahead:** Unlike naive trading environments that normalize indicators using full-dataset statistics (min/max or global mean), StockRL Mini computes all indicators causally up to timestep $t$. An agent cannot exploit future market states.
2. **Exploit-Resistant Reward Formulation:** In bull markets, trading environments often mistakenly reward invalid actions (e.g., attempting to buy with zero cash) because existing stock appreciates. StockRL Mini clips returns on invalid or violated actions to $\le 0$, eliminating free rewards.
3. **Independent Third-Party Verification:** Rather than trusting an agent's self-reported rewards or internal state, StockRL Mini introduces an independent verifier that replays raw action histories against canonical price feeds, validating accounting integrity.
4. **Lightweight & Fast:** Operating at thousands of steps per second without external network dependencies, StockRL Mini enables rapid algorithmic iteration, automated grading, and unit testing within standard CI/CD pipelines.
