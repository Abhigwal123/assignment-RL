# StockRL Mini — Adzzat Labs Take-Home Assignment

> **Lead Reinforcement Learning Environment Engineer Candidate Submission**  
> A minimal, reproducible, and verifiable stock market simulation environment for Reinforcement Learning research, conforming strictly to Gymnasium standards and audited by a zero-trust cryptographic verifier.

---

## 1. Project Objective

StockRL Mini is a lightweight, causal single-stock trading simulation designed to evaluate Reinforcement Learning algorithms under progressive market frictions and risk constraints.

Key design goals:
- **Causal RL Interface**: 8-dimensional continuous observation vector guaranteeing zero lookahead bias.
- **Three Progressive Tasks**: Basic trading $\to$ Transaction friction & slippage $\to$ Position caps & drawdown risk penalty.
- **Zero-Trust Grader**: Independent accounting verification that never trusts reported rewards or metrics, re-executing raw action sequences against canonical market prices with SHA-256 state sealing.
- **Adversarial Exploit Resistance**: Hardened against grading code tampering (monkeypatching) and state manipulation (cash inflation, price spoofing, fee evasion).
- **Production Telemetry Workbench**: Streamlit telemetry suite exposing real-time MDP state-action transitions, Shannon action entropy, reward attribution decomposition, and quantitative risk metrics.

---

## 2. Architecture & Directory Structure

The project conforms strictly to the assignment specifications:

```
stockrl-mini/
├── environment/
│   ├── __init__.py           # Package exports (StockTradingEnv, MarketData, etc.)
│   ├── trading_env.py        # Gymnasium-compliant StockTradingEnv
│   ├── market_data.py        # Causal feature computation from synthetic prices
│   └── reward.py             # Pure reward calculation & risk penalties
├── agents/
│   ├── __init__.py           # Agent package exports
│   └── baseline_agent.py     # RandomAgent, BuyAndHoldAgent, SMAAgent
├── grader/
│   ├── __init__.py           # Grader package exports
│   ├── verifier.py           # Zero-trust independent replay verifier & SHA-256 seal
│   └── exploit_tests.py      # Exploit 1 (tampering) and Exploit 2 (state spoofing)
├── data/
│   └── sample_prices.csv     # 252-day synthetic stock prices (GBM, S_0=₹1000)
├── tests/
│   ├── test_environment.py   # 17 Gymnasium env & accounting tests
│   └── test_grader.py        # 10 verifier, exploit, & checksum tests
├── reports/
│   ├── scope.md              # Scope, interface design & RL utility report
│   ├── results.md            # Empirical baseline benchmark results
│   └── exploit_results.md    # Exploit analysis & threat model boundaries
├── scripts/
│   ├── run_agent.py          # Benchmark runner CLI across all tasks
│   └── grade_episode.py      # Independent episode grader CLI
├── dashboard/
│   └── app.py                # Lead RL Engineer Streamlit Telemetry Workbench
├── api/
│   └── main.py               # FastAPI REST Backend & Embedded Browser UI
├── Dockerfile                # Production Python 3.11 container definition
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Standard packaging definition
└── README.md                 # Complete documentation
```

---

## 3. Installation & Setup

### Prerequisites
- Python 3.11+
- Virtual environment tool (`venv`) or Docker

### Local Setup
```bash
# Clone and enter the repository
cd stockrl-mini

# Create and activate virtual environment
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

# Install dependencies (or install as editable package)
pip install -r requirements.txt
pip install -e .
```

---

## 4. How to Run

### A. Run Automated Unit Tests (27 Tests)
```bash
pytest tests/ -v
```
All 27 tests validate:
- Gymnasium `check_env` conformance.
- Absence of lookahead bias.
- Non-negative cash and valid share boundaries.
- Slippage and fee accounting invariants.
- Independent verifier replay correctness.
- Interception of Exploit 1 and Exploit 2.

### B. Run Baseline Agents Across Tasks
```bash
# Run all agents across all 3 tasks
python scripts/run_agent.py --agent all --task all

# Run specific agent on Task 2
python scripts/run_agent.py --agent sma --task 2
```

### C. Run the Zero-Trust Grader
```bash
# Run and grade an episode
python scripts/grade_episode.py --agent sma --task 2

# Demonstrate an exploit attempt being caught
python scripts/grade_episode.py --agent sma --task 2 --exploit cash_inflation
```

### D. Launch the RL Telemetry Workbench (Streamlit)
```bash
python -m streamlit run dashboard/app.py
```
Open [http://localhost:8501](http://localhost:8501) to interact with:
1. **MDP Telemetry & Stepper**: Live 8-dim state tensor, transition engine ($S_t \to A_t \to R_t$).
2. **Policy Dynamics & Entropy**: Shannon action entropy $H(\pi)$, reward attribution decomposition.
3. **Quantitative Benchmark Matrix**: Sharpe, Sortino, Calmar, MDD, and equity curve overlays.
4. **Zero-Trust Grader Cockpit**: Step-by-step invariant auditing and SHA-256 seal.
5. **Adversarial Exploit Lab**: Interactive exploit attack and defense verification.

### E. Launch FastAPI Backend
```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```
- Interactive Swagger docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Embedded HTML5 dashboard: [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard)

---

## 5. Three Task Configurations

| Parameter | Task 1: Basic Trading | Task 2: Transaction Costs | Task 3: Risk-Aware Trading |
| :--- | :--- | :--- | :--- |
| **Objective** | Gross portfolio gain | Net gain after costs | Risk-adjusted return |
| **Transaction Fee** | `0.0%` | `0.10%` per trade | `0.10%` per trade |
| **Slippage** | `0.0%` | `0.05%` execution drag | `0.05%` execution drag |
| **Position Limit** | Unlimited | Unlimited | **50 shares max** |
| **Drawdown Penalty** | `0.0` | `0.0` | **0.50x linear weight** |
| **Invalid Action Penalty** | `-0.005` | `-0.005` | `-0.010` |

---

## 6. Reward Design & Mathematical Formulation

The reward function enforces financial realism and eliminates reward hacking:

$$R_t = \text{Effective Step Return} - \text{Drawdown Penalty} - \text{Action Penalties}$$

1. **Step Return**:
   $$\Delta r_t = \frac{PV_t - PV_{t-1}}{PV_{t-1}}$$
2. **Anti-Hacking Return Clipping**:
   If an agent takes an unexecutable action (buying with ₹0 cash, selling with 0 shares) or violates position limits, its return is strictly clipped:
   $$\text{Effective Return} = \min(0.0, \Delta r_t)$$
   *This ensures holding appreciation cannot reward an invalid trade.*
3. **Drawdown Penalty (Task 3)**:
   $$\text{Penalty}_{DD} = \lambda_{DD} \times \frac{HWM_t - PV_t}{HWM_t}$$
   where $HWM_t = \max_{0 \le i \le t} PV_i$.

---

## 7. Zero-Trust Grader & Threat Model Boundaries

### What the Grader Guarantees:
1. **Canonical Price Replay**: Prices are checked against read-only `sample_prices.csv`.
2. **Accounting Invariant Watchdog**: Proves $Cash_t \ge 0$, $Shares_t \ge 0$, and $Cash_t + Shares_t \cdot P_t = PV_t$.
3. **Friction Auditing**: Re-derives all fees and slippage independently.
4. **Cryptographic Sealing**: Emits a SHA-256 digest over the verified trajectory.

### What the Grader Cannot Protect Against:
- **Unrestricted Host/Kernel Access**: A student or agent running as the same OS user can monkeypatch in-memory Python bytecode or edit local test files.
- **Defense-in-Depth Solution**: Enforce grading exclusively inside an isolated CI/CD container where grading scripts are read-only and checksum-verified against an external trusted repository.

---

## 8. Docker Execution

### Build the Image
```bash
docker build -t stockrl-mini .
```

### Run Default Test Suite & Benchmarks
```bash
docker run --rm stockrl-mini
```

### Run the Grader inside Docker
```bash
docker run --rm stockrl-mini python scripts/grade_episode.py --agent sma --task 2
```

### Import Directly in Python Container
```bash
docker run --rm -it stockrl-mini python -c "
from environment import StockTradingEnv
env = StockTradingEnv(task=2)
obs, info = env.reset(seed=42)
print('Observation vector:', obs)
"
```

### Run Streamlit Telemetry Workbench from Docker
```bash
docker run --rm -p 8501:8501 stockrl-mini python -m streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
```

---

## 9. AI Tooling Disclosure

In accordance with transparent engineering disclosure:
- **AI Agent Used**: Google DeepMind Antigravity (powered by Gemini models).
- **Scope of AI Assistance**: Architecture scaffolding, causal indicator precomputation, test case generation, and report draft formatting.
- **Human Review**: All financial formulas, invariant verification algorithms, threat model boundaries, and empirical evaluation metrics were verified and audited for correctness.
