# StockRL Mini: Benchmark Experimental Results

## 1. Experimental Setup
All baseline agents were evaluated on the canonical 252-day synthetic price sequence (`sample_prices.csv`) with identical initial capital of ₹100,000.00 and random seed 42.

Three baseline strategies were tested across all three tasks:
1. **Random Agent:** Uniform stochastic action selection.
2. **Buy-and-Hold Agent:** Fully allocates available cash to equity at episode inception and holds until termination.
3. **SMA Crossover Agent:** Trades on 5-step vs 20-step Simple Moving Average divergence.

---

## 2. Empirical Benchmark Results

| Task | Agent | Final Portfolio Value (₹) | Net Return (%) | Trades | Transaction Costs (₹) | Max Drawdown (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Task 1: Basic Trading** | Random Agent | ₹105,750.51 | +5.75% | 155 | ₹0.00 | 10.11% |
| | Buy & Hold Agent | ₹101,577.10 | +1.58% | 10 | ₹0.00 | 21.24% |
| | SMA Crossover Agent | ₹93,029.54 | -6.97% | 102 | ₹0.00 | 11.30% |
| **Task 2: Trading with Costs** | Random Agent | ₹103,395.46 | +3.40% | 155 | ₹2,165.09 | 10.81% |
| | Buy & Hold Agent | ₹101,428.07 | +1.43% | 10 | ₹149.03 | 21.28% |
| | SMA Crossover Agent | ₹91,688.85 | -8.31% | 102 | ₹1,388.86 | 11.91% |
| **Task 3: Risk-Aware Trading** | Random Agent | ₹97,984.10 | -2.02% | 142 | ₹2,016.60 | 8.14% |
| | Buy & Hold Agent | ₹101,699.48 | +1.70% | 5 | ₹77.02 | 11.24% |
| | SMA Crossover Agent | ₹97,146.99 | -2.85% | 54 | ₹770.31 | 5.54% |

---

## 3. Key Observations & Financial Analysis

1. **Transaction Costs Penalize Excessive Trading:**
   - In Task 1 (frictionless), the Random Agent achieved +5.75% purely by chance due to lucky timing over 155 trades.
   - When realistic transaction fees (0.10%) and slippage (0.05%) were activated in Task 2, the Random Agent lost ₹2,165.09 directly to churn costs, dropping its return by 2.35 percentage points.
   - Buy-and-Hold executed only 10 trades, incurring a modest ₹149.03 in fees, showing superior cost efficiency.
2. **SMA Lag and Whipsaw Losses:**
   - The SMA Crossover Agent lost capital in all three tasks (-6.97%, -8.31%, -2.85%). Moving average crossovers suffer from lag in range-bound and volatile markets: the agent bought at local tops and sold at local bottoms, incurring continuous whipsaw losses.
3. **Position Constraints Cut Drawdown in Half:**
   - In Task 3, capping the position at 50 shares cut Buy-and-Hold's maximum drawdown from 21.24% down to 11.24% (a ~47% reduction in risk exposure).
   - The SMA Agent in Task 3 had its maximum drawdown reduced to just 5.54%, demonstrating that hard position sizing constraints are essential for risk mitigation.

---

## 4. What These Experiments Do NOT Prove

1. **No Evidence of Learned Alpha:** None of these baseline agents learned a trading policy via gradient descent or value estimation. The positive returns of the Random Agent in Tasks 1 and 2 are artifacts of random seed variance on this specific price path, not an enduring edge.
2. **No Generalization Guarantee:** Performance on a single 252-day synthetic sequence does not imply viability on out-of-sample regimes (bear markets, liquidity crunches, flash crashes).
3. **Execution Realism Limitations:** Real-world market microstructure entails non-linear market impact, limit order queue placement, and execution delays that cannot be modeled by simple linear slippage.

---

## 5. How Larger Training Evaluations Could Measure Improvement

To rigorously validate an RL agent (such as PPO, SAC, or DQN) against these baselines:
1. **Multi-Path Cross-Validation:** Train and evaluate over thousands of diverse Geometric Brownian Motion paths with varying drift ($\mu \in [-0.2, 0.2]$) and volatility regimes ($\sigma \in [0.1, 0.5]$).
2. **Sharpe and Sortino Ratio Metrics:** Benchmark distribution of risk-adjusted excess returns rather than nominal final portfolio value alone.
3. **Out-of-Sample Regime Stress Testing:** Measure performance retention across regime-shifting market environments (trending, mean-reverting, and high-volatility panic regimes).
