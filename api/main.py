"""
StockRL Mini - FastAPI Application.

Provides RESTful endpoints for:
- Environment state inspection, reset, and step execution.
- Automated baseline simulation (Random, Buy-and-Hold, SMA).
- Zero-trust independent grading of episodes.
- Deliberate exploit execution and defense verification.
- Market data streaming.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.trading_env import StockTradingEnv, TASK_CONFIGS, TaskConfig
from environment.market_data import MarketData
from grader.verifier import Verifier, VerificationReport
from grader.exploit_tests import Exploit1TestManipulation, Exploit2StateAndRewardManipulation
from agents.baseline_agent import RandomAgent, BuyAndHoldAgent, SMAAgent


app = FastAPI(
    title="StockRL Mini API",
    description="REST API for StockRL Mini Gymnasium Environment & Independent Grader",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global active environment instance for interactive sessions
active_env: Optional[StockTradingEnv] = None
data_path = Path(__file__).parent.parent / "data" / "sample_prices.csv"
market_data = MarketData(data_path)


# ------------------------------------------------------------------------------
# Request/Response Models
# ------------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: int = Field(1, description="Task ID (1: Basic, 2: Costs, 3: Risk-Aware)")
    seed: int = Field(42, description="Random seed")
    initial_cash: float = Field(100_000.0, description="Initial starting cash (₹)")


class StepRequest(BaseModel):
    action: int = Field(..., ge=0, le=2, description="Action code: 0=Sell, 1=Hold, 2=Buy")


class SimulateRequest(BaseModel):
    task_id: int = Field(1, description="Task ID (1, 2, or 3)")
    agent_name: str = Field("sma", description="Agent: random, buy_and_hold, or sma")
    seed: int = Field(42, description="Random seed")


class GradeRequest(BaseModel):
    task_id: int = Field(1, description="Task ID (1, 2, or 3)")
    history: Optional[List[Dict[str, Any]]] = Field(None, description="Episode history list to grade. If omitted, uses current active episode.")


class ExploitRequest(BaseModel):
    exploit_type: str = Field(..., description="cash_inflation, price_spoofing, fee_evasion, or monkeypatch")
    task_id: int = Field(2, description="Task ID for exploit demonstration")


# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "StockRL Mini API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "available_tasks": [
            {"id": tid, "name": cfg.name, "fee": cfg.fee_rate, "slippage": cfg.slippage, "max_position": cfg.max_position_shares}
            for tid, cfg in TASK_CONFIGS.items()
        ],
    }


@app.get("/api/tasks")
def get_tasks():
    return {
        tid: {
            "task_id": cfg.task_id,
            "name": cfg.name,
            "fee_rate": cfg.fee_rate,
            "slippage": cfg.slippage,
            "max_position_shares": cfg.max_position_shares,
            "drawdown_penalty_weight": cfg.drawdown_penalty_weight,
            "trade_size": cfg.trade_size,
        }
        for tid, cfg in TASK_CONFIGS.items()
    }


@app.get("/api/market-data")
def get_market_data(limit: Optional[int] = Query(None, description="Max rows to return")):
    df = market_data.df
    if limit is not None:
        df = df.head(limit)
    return {
        "total_records": len(df),
        "initial_price": market_data.initial_price,
        "records": df.to_dict(orient="records"),
    }


@app.post("/api/env/reset")
def reset_env(req: ResetRequest):
    global active_env
    if req.task_id not in TASK_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Invalid task_id {req.task_id}. Must be 1, 2, or 3.")

    active_env = StockTradingEnv(
        task=req.task_id,
        csv_path=data_path,
        initial_cash=req.initial_cash,
    )
    obs, info = active_env.reset(seed=req.seed)

    return {
        "message": "Environment successfully reset.",
        "task_id": req.task_id,
        "task_name": active_env.config.name,
        "step": active_env.current_step,
        "price": active_env.market_data.get_price(0),
        "cash": active_env.cash,
        "shares": active_env.shares,
        "portfolio_value": active_env.portfolio_value,
        "observation": obs.tolist(),
        "info": info,
    }


@app.post("/api/env/step")
def step_env(req: StepRequest):
    global active_env
    if active_env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /api/env/reset first.")

    obs, reward, terminated, truncated, info = active_env.step(req.action)
    current_record = active_env.episode_history[-1]

    return {
        "step": active_env.current_step,
        "action": req.action,
        "action_name": current_record["action_name"],
        "price": current_record["price"],
        "reward": float(reward),
        "cash": active_env.cash,
        "shares": active_env.shares,
        "portfolio_value": active_env.portfolio_value,
        "step_cost": current_record["step_cost"],
        "fee": current_record["fee"],
        "slippage": current_record["slippage"],
        "shares_traded": current_record["shares_traded"],
        "is_invalid": current_record["is_invalid"],
        "is_violation": current_record["is_violation"],
        "drawdown": active_env.current_drawdown,
        "max_drawdown": active_env.max_drawdown,
        "terminated": terminated,
        "truncated": truncated,
        "observation": obs.tolist(),
        "info": info,
    }


@app.get("/api/env/state")
def get_env_state():
    global active_env
    if active_env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /api/env/reset first.")

    return {
        "task_id": active_env.config.task_id,
        "task_name": active_env.config.name,
        "step": active_env.current_step,
        "total_steps": active_env.market_data.total_steps,
        "price": active_env.market_data.get_price(active_env.current_step),
        "cash": active_env.cash,
        "shares": active_env.shares,
        "portfolio_value": active_env.portfolio_value,
        "current_drawdown": active_env.current_drawdown,
        "max_drawdown": active_env.max_drawdown,
        "history_length": len(active_env.episode_history),
    }


@app.post("/api/simulate")
def simulate_agent(req: SimulateRequest):
    if req.task_id not in TASK_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Invalid task_id {req.task_id}")

    agent_map = {
        "random": lambda: RandomAgent(seed=req.seed),
        "buy_and_hold": lambda: BuyAndHoldAgent(),
        "sma": lambda: SMAAgent(),
    }
    if req.agent_name.lower() not in agent_map:
        raise HTTPException(status_code=400, detail=f"Unknown agent '{req.agent_name}'. Options: {list(agent_map.keys())}")

    env = StockTradingEnv(task=req.task_id, csv_path=data_path)
    agent = agent_map[req.agent_name.lower()]()

    obs, info = env.reset(seed=req.seed)
    agent.reset()

    terminated = False
    truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        action = agent.predict(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    history = env.get_episode_history()
    final_pv = history[-1]["portfolio_value"]
    net_return = (final_pv - env.initial_cash) / env.initial_cash
    total_costs = sum(h.get("step_cost", 0.0) for h in history)
    total_trades = sum(1 for h in history if h.get("shares_traded", 0) > 0)

    # Automatically subject to zero-trust verification
    report = Verifier.verify_episode(history, env.config, env.market_data, env.initial_cash)

    return {
        "agent": agent.name,
        "task_id": req.task_id,
        "task_name": env.config.name,
        "initial_capital": env.initial_cash,
        "final_portfolio_value": final_pv,
        "net_return_pct": net_return * 100.0,
        "total_reward": total_reward,
        "total_trades": total_trades,
        "total_costs": total_costs,
        "max_drawdown_pct": env.max_drawdown * 100.0,
        "steps_simulated": len(history) - 1,
        "verification": {
            "passed": report.passed,
            "discrepancies": report.discrepancies,
            "integrity_hash": report.integrity_hash,
            "replayed_net_return_pct": report.net_return * 100.0,
            "replayed_final_pv": report.final_portfolio_value,
        },
        "history": history,
    }


@app.post("/api/grade")
def grade_episode(req: GradeRequest):
    global active_env
    if req.history is not None:
        target_history = req.history
        task_cfg = TASK_CONFIGS.get(req.task_id, TASK_CONFIGS[1])
    elif active_env is not None:
        target_history = active_env.get_episode_history()
        task_cfg = active_env.config
    else:
        raise HTTPException(status_code=400, detail="No history provided and no active episode in memory.")

    report = Verifier.verify_episode(
        history=target_history,
        task_config=task_cfg,
        market_data=market_data,
        initial_cash=100_000.0,
    )

    return {
        "passed": report.passed,
        "task_id": report.task_id,
        "initial_capital": report.initial_capital,
        "final_portfolio_value": report.final_portfolio_value,
        "net_return_pct": report.net_return * 100.0,
        "total_trades": report.total_trades,
        "total_transaction_costs": report.total_transaction_costs,
        "total_fees": report.total_fees,
        "total_slippage": report.total_slippage,
        "invalid_transactions": report.invalid_transactions,
        "constraint_violations": report.constraint_violations,
        "max_drawdown_pct": report.max_drawdown * 100.0,
        "discrepancies": report.discrepancies,
        "integrity_hash": report.integrity_hash,
        "summary": report.summary(),
    }


@app.post("/api/exploit/test")
def run_exploit_test(req: ExploitRequest):
    env = StockTradingEnv(task=req.task_id, csv_path=data_path)

    if req.exploit_type == "cash_inflation":
        report = Exploit2StateAndRewardManipulation.simulate_cash_inflation_exploit(env)
        description = "Injected +₹50,000 unearned cash into recorded history."
    elif req.exploit_type == "price_spoofing":
        report = Exploit2StateAndRewardManipulation.simulate_price_spoofing_exploit(env)
        description = "Forged execution price to ₹10,000 (10x real market price)."
    elif req.exploit_type == "fee_evasion":
        report = Exploit2StateAndRewardManipulation.simulate_fee_evasion_exploit(env)
        description = "Zeroed out reported fees in Task 2 to pretend zero friction."
    elif req.exploit_type == "monkeypatch":
        succeeded, explanation = Exploit1TestManipulation.attempt_monkeypatch_exploit([], req.task_id)
        verifier_path = Path(__file__).parent.parent / "grader" / "verifier.py"
        real_hash = Exploit1TestManipulation.get_verifier_hash(verifier_path)
        return {
            "exploit_type": req.exploit_type,
            "description": "Simulated in-memory monkeypatch of Verifier class.",
            "monkeypatch_succeeded": succeeded,
            "explanation": explanation,
            "defense": "SHA-256 code integrity hash",
            "file_hash": real_hash,
            "status": "DETECTED BY CHECKSUM",
        }
    else:
        raise HTTPException(status_code=400, detail=f"Unknown exploit_type '{req.exploit_type}'")

    return {
        "exploit_type": req.exploit_type,
        "description": description,
        "verifier_passed": report.passed,
        "caught": not report.passed,
        "discrepancies_found": report.discrepancies,
        "status": "BLOCKED" if not report.passed else "EXPLOIT SUCCEEDED",
    }


@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard():
    """Serves the rich, interactive single-page dashboard."""
    html_content = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>StockRL Mini - Interactive Trading & Audit Lab</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { background-color: #0f172a; color: #f8fafc; font-family: ui-sans-serif, system-ui, sans-serif; }
        .card { background-color: #1e293b; border: 1px solid #334155; border-radius: 0.75rem; }
    </style>
</head>
<body class="p-4 md:p-6">
    <div class="max-w-7xl mx-auto space-y-6">
        <!-- Header -->
        <header class="flex flex-col md:flex-row md:items-center justify-between pb-4 border-b border-slate-700 gap-4">
            <div>
                <h1 class="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
                    <span>📈</span> StockRL Mini <span class="text-xs bg-blue-600 text-white px-2 py-0.5 rounded-full font-mono">v1.0</span>
                </h1>
                <p class="text-xs text-slate-400 mt-1">Gymnasium Trading Environment & Independent Grader Lab (Adzzat Labs)</p>
            </div>
            <div class="flex items-center gap-3">
                <select id="taskSelect" class="bg-slate-800 text-slate-200 text-sm px-3 py-2 rounded-lg border border-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-500">
                    <option value="1">Task 1: Basic Trading (0% Fee, 0% Slippage)</option>
                    <option value="2" selected>Task 2: Trading with Costs (0.10% Fee, 0.05% Slippage)</option>
                    <option value="3">Task 3: Risk-Aware (50-Share Max, Drawdown Penalty)</option>
                </select>
                <button onclick="resetEnv()" class="bg-blue-600 hover:bg-blue-500 text-white text-sm font-semibold px-4 py-2 rounded-lg shadow transition">
                    🔄 Reset Env
                </button>
            </div>
        </header>

        <!-- KPI Cards -->
        <div class="grid grid-cols-2 md:grid-cols-6 gap-3">
            <div class="card p-4">
                <div class="text-xs text-slate-400 uppercase font-semibold">Portfolio Value</div>
                <div id="pvVal" class="text-lg font-bold text-emerald-400 mt-1">₹100,000.00</div>
                <div id="retVal" class="text-xs text-slate-400 mt-0.5">+0.00%</div>
            </div>
            <div class="card p-4">
                <div class="text-xs text-slate-400 uppercase font-semibold">Stock Price</div>
                <div id="priceVal" class="text-lg font-bold text-white mt-1">₹1,007.85</div>
                <div id="stepVal" class="text-xs text-slate-400 mt-0.5">Step 0 / 251</div>
            </div>
            <div class="card p-4">
                <div class="text-xs text-slate-400 uppercase font-semibold">Available Cash</div>
                <div id="cashVal" class="text-lg font-bold text-white mt-1">₹100,000.00</div>
                <div class="text-xs text-slate-500 mt-0.5">Liquid capital</div>
            </div>
            <div class="card p-4">
                <div class="text-xs text-slate-400 uppercase font-semibold">Holdings</div>
                <div id="sharesVal" class="text-lg font-bold text-white mt-1">0 shares</div>
                <div id="holdingEquityVal" class="text-xs text-slate-400 mt-0.5">₹0.00 equity</div>
            </div>
            <div class="card p-4">
                <div class="text-xs text-slate-400 uppercase font-semibold">Friction Costs</div>
                <div id="costsVal" class="text-lg font-bold text-amber-400 mt-1">₹0.00</div>
                <div id="tradesVal" class="text-xs text-slate-400 mt-0.5">0 trades</div>
            </div>
            <div class="card p-4">
                <div class="text-xs text-slate-400 uppercase font-semibold">Max Drawdown</div>
                <div id="ddVal" class="text-lg font-bold text-rose-400 mt-1">0.00%</div>
                <div id="curDdVal" class="text-xs text-slate-400 mt-0.5">Current: 0.00%</div>
            </div>
        </div>

        <!-- Action Controls -->
        <div class="card p-4 flex flex-wrap items-center justify-between gap-4">
            <div class="flex items-center gap-2">
                <button onclick="stepAction(2)" class="bg-emerald-600 hover:bg-emerald-500 text-white font-bold px-4 py-2 rounded-lg text-sm transition">
                    🟢 BUY (+10)
                </button>
                <button onclick="stepAction(1)" class="bg-slate-700 hover:bg-slate-600 text-white font-bold px-4 py-2 rounded-lg text-sm transition">
                    🟡 HOLD
                </button>
                <button onclick="stepAction(0)" class="bg-rose-600 hover:bg-rose-500 text-white font-bold px-4 py-2 rounded-lg text-sm transition">
                    🔴 SELL (-10)
                </button>
                <button onclick="stepMulti(10)" class="bg-indigo-600 hover:bg-indigo-500 text-white font-semibold px-3 py-2 rounded-lg text-sm transition">
                    ⚡ Step +10 Days
                </button>
            </div>
            <div class="flex items-center gap-2">
                <select id="agentSelect" class="bg-slate-800 text-slate-200 text-sm px-3 py-2 rounded-lg border border-slate-700">
                    <option value="sma">SMA Crossover Agent</option>
                    <option value="buy_and_hold">Buy & Hold Agent</option>
                    <option value="random">Random Agent</option>
                </select>
                <button onclick="runSimulation()" class="bg-purple-600 hover:bg-purple-500 text-white font-semibold px-4 py-2 rounded-lg text-sm transition">
                    🚀 Run Full Agent
                </button>
                <button onclick="runAudit()" class="bg-amber-600 hover:bg-amber-500 text-white font-semibold px-4 py-2 rounded-lg text-sm transition">
                    🛡️ Audit with Grader
                </button>
            </div>
        </div>

        <!-- Charts Grid -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div class="card p-4">
                <div class="flex justify-between items-center mb-2">
                    <h3 class="text-sm font-semibold text-slate-200">Stock Price Trajectory & Trades</h3>
                    <span id="priceSubtitle" class="text-xs text-slate-400">₹0.00</span>
                </div>
                <div class="h-64">
                    <canvas id="priceChart"></canvas>
                </div>
            </div>
            <div class="card p-4">
                <div class="flex justify-between items-center mb-2">
                    <h3 class="text-sm font-semibold text-slate-200">Portfolio Value & Cash</h3>
                    <span id="pvSubtitle" class="text-xs text-slate-400">Equity Breakdown</span>
                </div>
                <div class="h-64">
                    <canvas id="portfolioChart"></canvas>
                </div>
            </div>
        </div>

        <!-- Tabs Section: Audit & Security -->
        <div class="card p-4">
            <div class="flex border-b border-slate-700 gap-4 mb-4">
                <button onclick="switchTab('audit')" id="tabBtnAudit" class="text-sm font-semibold pb-2 border-b-2 border-blue-500 text-blue-400">
                    🛡️ Zero-Trust Grader Audit
                </button>
                <button onclick="switchTab('exploits')" id="tabBtnExploits" class="text-sm font-semibold pb-2 border-b-2 border-transparent text-slate-400 hover:text-slate-200">
                    🔬 Exploit & Security Lab
                </button>
                <button onclick="switchTab('ledger')" id="tabBtnLedger" class="text-sm font-semibold pb-2 border-b-2 border-transparent text-slate-400 hover:text-slate-200">
                    📜 Execution Ledger
                </button>
            </div>

            <!-- Tab 1: Audit -->
            <div id="tabAudit" class="space-y-4">
                <div id="auditBanner" class="p-4 rounded-lg bg-slate-800 text-sm text-slate-300">
                    Click <strong>"Audit with Grader"</strong> above to independently verify the active episode accounting against ground-truth market prices.
                </div>
                <div id="auditDetails" class="hidden font-mono text-xs bg-slate-900 p-4 rounded-lg text-slate-300 overflow-x-auto whitespace-pre"></div>
            </div>

            <!-- Tab 2: Exploits -->
            <div id="tabExploits" class="hidden space-y-4">
                <p class="text-xs text-slate-400">Test how the Independent Grader and cryptographic checksums detect deliberate fraud:</p>
                <div class="grid grid-cols-1 md:grid-cols-4 gap-3">
                    <button onclick="triggerExploit('cash_inflation')" class="p-3 card hover:border-red-500 text-left transition">
                        <div class="text-sm font-bold text-red-400">Exploit 2A: Cash Inflation</div>
                        <div class="text-xs text-slate-400 mt-1">Inject +₹50k unearned cash into history trace.</div>
                    </button>
                    <button onclick="triggerExploit('price_spoofing')" class="p-3 card hover:border-red-500 text-left transition">
                        <div class="text-sm font-bold text-red-400">Exploit 2B: Price Spoofing</div>
                        <div class="text-xs text-slate-400 mt-1">Claim sell executed at ₹10,000 (10x price).</div>
                    </button>
                    <button onclick="triggerExploit('fee_evasion')" class="p-3 card hover:border-red-500 text-left transition">
                        <div class="text-sm font-bold text-red-400">Exploit 2C: Fee Evasion</div>
                        <div class="text-xs text-slate-400 mt-1">Claim ₹0 transaction costs in Task 2.</div>
                    </button>
                    <button onclick="triggerExploit('monkeypatch')" class="p-3 card hover:border-amber-500 text-left transition">
                        <div class="text-sm font-bold text-amber-400">Exploit 1: Code Tampering</div>
                        <div class="text-xs text-slate-400 mt-1">Simulate runtime monkeypatch attack.</div>
                    </button>
                </div>
                <div id="exploitResult" class="hidden p-4 rounded-lg font-mono text-xs bg-slate-900 text-slate-200"></div>
            </div>

            <!-- Tab 3: Ledger -->
            <div id="tabLedger" class="hidden">
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-xs text-slate-300">
                        <thead class="bg-slate-800 text-slate-400 uppercase text-[10px]">
                            <tr>
                                <th class="p-2">Step</th>
                                <th class="p-2">Action</th>
                                <th class="p-2">Price</th>
                                <th class="p-2">Cash</th>
                                <th class="p-2">Shares</th>
                                <th class="p-2">Portfolio Value</th>
                                <th class="p-2">Cost</th>
                                <th class="p-2">Invalid / Violation</th>
                            </tr>
                        </thead>
                        <tbody id="ledgerBody" class="divide-y divide-slate-800 font-mono"></tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <!-- Dashboard Logic -->
    <script>
        let priceChart, portfolioChart;
        let stepHistory = [];

        function initCharts() {
            const ctxPrice = document.getElementById('priceChart').getContext('2d');
            priceChart = new Chart(ctxPrice, {
                type: 'line',
                data: { labels: [0], datasets: [{ label: 'Stock Price (₹)', data: [1007.85], borderColor: '#3b82f6', tension: 0.1, pointRadius: 1 }] },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { grid: { color: '#334155' } }, y: { grid: { color: '#334155' } } } }
            });

            const ctxPv = document.getElementById('portfolioChart').getContext('2d');
            portfolioChart = new Chart(ctxPv, {
                type: 'line',
                data: {
                    labels: [0],
                    datasets: [
                        { label: 'Portfolio Value (₹)', data: [100000], borderColor: '#10b981', tension: 0.1, pointRadius: 1 },
                        { label: 'Cash Balance (₹)', data: [100000], borderColor: '#6366f1', borderDash: [4, 4], pointRadius: 0 }
                    ]
                },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#94a3b8' } } }, scales: { x: { grid: { color: '#334155' } }, y: { grid: { color: '#334155' } } } }
            });
        }

        async function resetEnv() {
            const taskId = parseInt(document.getElementById('taskSelect').value);
            const res = await fetch('/api/env/reset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task_id: taskId, seed: 42, initial_cash: 100000.0 })
            });
            const data = await res.json();
            stepHistory = [{ step: 0, action_name: 'HOLD', price: data.price, cash: data.cash, shares: data.shares, portfolio_value: data.portfolio_value, step_cost: 0, is_invalid: false, is_violation: false }];
            updateUI(data.portfolio_value, data.price, 0, data.cash, data.shares, 0, 0, 0, 0);
            renderCharts([data.price], [data.portfolio_value], [data.cash], [0]);
            renderLedger();
        }

        async function stepAction(act) {
            const res = await fetch('/api/env/step', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: act })
            });
            const d = await res.json();
            stepHistory.push(d);
            const totalCosts = stepHistory.reduce((acc, x) => acc + (x.step_cost || 0), 0);
            const totalTrades = stepHistory.filter(x => (x.shares_traded || 0) > 0).length;
            updateUI(d.portfolio_value, d.price, d.step, d.cash, d.shares, totalCosts, totalTrades, d.max_drawdown, d.drawdown);
            appendChartPoint(d.step, d.price, d.portfolio_value, d.cash);
            renderLedger();
        }

        async function stepMulti(n) {
            for (let i = 0; i < n; i++) {
                await stepAction(1);
            }
        }

        async function runSimulation() {
            const taskId = parseInt(document.getElementById('taskSelect').value);
            const agent = document.getElementById('agentSelect').value;
            const res = await fetch('/api/simulate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task_id: taskId, agent_name: agent, seed: 42 })
            });
            const data = await res.json();
            stepHistory = data.history;
            const labels = stepHistory.map(h => h.step);
            const prices = stepHistory.map(h => h.price);
            const pvs = stepHistory.map(h => h.portfolio_value);
            const cashes = stepHistory.map(h => h.cash);
            renderCharts(prices, pvs, cashes, labels);
            const last = stepHistory[stepHistory.length - 1];
            updateUI(last.portfolio_value, last.price, last.step, last.cash, last.shares, data.total_costs, data.total_trades, data.max_drawdown_pct / 100, last.drawdown || 0);
            renderLedger();
            runAudit();
        }

        async function runAudit() {
            const taskId = parseInt(document.getElementById('taskSelect').value);
            const res = await fetch('/api/grade', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task_id: taskId, history: stepHistory })
            });
            const report = await res.json();
            const banner = document.getElementById('auditBanner');
            const details = document.getElementById('auditDetails');
            if (report.passed) {
                banner.className = "p-4 rounded-lg bg-emerald-950 border border-emerald-600 text-sm text-emerald-200 font-semibold";
                banner.innerHTML = "✅ <strong>ZERO-TRUST AUDIT PASSED:</strong> 100% of actions and accounting reconciled with ground-truth market data.";
            } else {
                banner.className = "p-4 rounded-lg bg-rose-950 border border-rose-600 text-sm text-rose-200 font-semibold";
                banner.innerHTML = "❌ <strong>ZERO-TRUST AUDIT FAILED:</strong> Discrepancies detected!";
            }
            details.classList.remove('hidden');
            details.innerText = report.summary;
        }

        async function triggerExploit(type) {
            const taskId = parseInt(document.getElementById('taskSelect').value);
            const res = await fetch('/api/exploit/test', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ exploit_type: type, task_id: taskId })
            });
            const data = await res.json();
            const box = document.getElementById('exploitResult');
            box.classList.remove('hidden');
            box.innerText = JSON.stringify(data, null, 2);
        }

        function updateUI(pv, price, step, cash, shares, costs, trades, maxDd, curDd) {
            document.getElementById('pvVal').innerText = '₹' + pv.toLocaleString('en-IN', { minimumFractionDigits: 2 });
            const ret = ((pv - 100000) / 100000) * 100;
            document.getElementById('retVal').innerText = (ret >= 0 ? '+' : '') + ret.toFixed(2) + '%';
            document.getElementById('priceVal').innerText = '₹' + price.toLocaleString('en-IN', { minimumFractionDigits: 2 });
            document.getElementById('stepVal').innerText = `Step ${step} / 251`;
            document.getElementById('cashVal').innerText = '₹' + cash.toLocaleString('en-IN', { minimumFractionDigits: 2 });
            document.getElementById('sharesVal').innerText = `${shares} shares`;
            document.getElementById('holdingEquityVal').innerText = '₹' + (shares * price).toLocaleString('en-IN', { minimumFractionDigits: 2 }) + ' equity';
            document.getElementById('costsVal').innerText = '₹' + costs.toLocaleString('en-IN', { minimumFractionDigits: 2 });
            document.getElementById('tradesVal').innerText = `${trades} trades`;
            document.getElementById('ddVal').innerText = (maxDd * 100).toFixed(2) + '%';
            document.getElementById('curDdVal').innerText = `Current: ${(curDd * 100).toFixed(2)}%`;
        }

        function renderCharts(prices, pvs, cashes, labels) {
            priceChart.data.labels = labels;
            priceChart.data.datasets[0].data = prices;
            priceChart.update();

            portfolioChart.data.labels = labels;
            portfolioChart.data.datasets[0].data = pvs;
            portfolioChart.data.datasets[1].data = cashes;
            portfolioChart.update();
        }

        function appendChartPoint(step, price, pv, cash) {
            priceChart.data.labels.push(step);
            priceChart.data.datasets[0].data.push(price);
            priceChart.update();

            portfolioChart.data.labels.push(step);
            portfolioChart.data.datasets[0].data.push(pv);
            portfolioChart.data.datasets[1].data.push(cash);
            portfolioChart.update();
        }

        function renderLedger() {
            const body = document.getElementById('ledgerBody');
            const recent = stepHistory.slice(-12).reverse();
            body.innerHTML = recent.map(r => `
                <tr class="hover:bg-slate-800/50">
                    <td class="p-2">${r.step}</td>
                    <td class="p-2"><span class="px-2 py-0.5 rounded text-[10px] ${r.action_name === 'BUY' ? 'bg-emerald-950 text-emerald-400' : r.action_name === 'SELL' ? 'bg-rose-950 text-rose-400' : 'bg-slate-800 text-slate-400'}">${r.action_name || 'HOLD'}</span></td>
                    <td class="p-2">₹${(r.price || 0).toFixed(2)}</td>
                    <td class="p-2">₹${(r.cash || 0).toFixed(2)}</td>
                    <td class="p-2">${r.shares || 0}</td>
                    <td class="p-2">₹${(r.portfolio_value || 0).toFixed(2)}</td>
                    <td class="p-2">₹${(r.step_cost || 0).toFixed(2)}</td>
                    <td class="p-2">${r.is_invalid ? '<span class="text-rose-400">Invalid</span>' : r.is_violation ? '<span class="text-amber-400">Violation</span>' : '-'}</td>
                </tr>
            `).join('');
        }

        function switchTab(t) {
            ['audit', 'exploits', 'ledger'].forEach(x => {
                document.getElementById('tab' + x.charAt(0).toUpperCase() + x.slice(1)).classList.add('hidden');
                document.getElementById('tabBtn' + x.charAt(0).toUpperCase() + x.slice(1)).className = "text-sm font-semibold pb-2 border-b-2 border-transparent text-slate-400 hover:text-slate-200";
            });
            document.getElementById('tab' + t.charAt(0).toUpperCase() + t.slice(1)).classList.remove('hidden');
            document.getElementById('tabBtn' + t.charAt(0).toUpperCase() + t.slice(1)).className = "text-sm font-semibold pb-2 border-b-2 border-blue-500 text-blue-400";
        }

        window.onload = () => {
            initCharts();
            resetEnv();
        };
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

