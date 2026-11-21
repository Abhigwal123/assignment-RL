"""
StockRL Mini - Lead RL Environment Engineer Telemetry & Verification Workbench.

Designed for production RL environment analysis, Markov Decision Process (MDP) telemetry,
reward attribution diagnostics, policy action entropy tracking, zero-trust cryptographic
grading, and adversarial exploit auditing.

Complies with Gymnasium environment standards and zero-trust verification specifications.
"""

from pathlib import Path
import sys
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from environment.trading_env import StockTradingEnv, TASK_CONFIGS, TaskConfig
from environment.market_data import MarketData
from grader.verifier import Verifier, VerificationReport
from grader.exploit_tests import Exploit1TestManipulation, Exploit2StateAndRewardManipulation
from agents.baseline_agent import RandomAgent, BuyAndHoldAgent, SMAAgent

# ==============================================================================
# Page Configuration & Dark Cyber-Terminal RL Theme
# ==============================================================================

st.set_page_config(
    page_title="StockRL Mini — Lead RL Telemetry & Audit Workbench",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Dark Cyber-Terminal Aesthetics */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Top Header Bar */
    .telemetry-banner {
        background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 24px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .telemetry-title {
        font-size: 22px;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .telemetry-sub {
        font-size: 13px;
        color: #94a3b8;
        font-family: monospace;
    }
    
    /* Badge Elements */
    .gym-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
        font-family: monospace;
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .spec-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 700;
        font-family: monospace;
        background-color: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.4);
    }

    /* Telemetry Card Containers */
    .rl-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .rl-card-header {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #64748b;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .rl-card-val {
        font-size: 22px;
        font-weight: 700;
        color: #f8fafc;
        font-family: "JetBrains Mono", monospace, sans-serif;
    }
    .rl-card-sub {
        font-size: 11px;
        color: #94a3b8;
        margin-top: 4px;
        font-family: monospace;
    }

    /* Observation Tensor Inspection Grid */
    .obs-tensor {
        background: #030712;
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 10px 14px;
        font-family: "JetBrains Mono", monospace;
        font-size: 12px;
        color: #38bdf8;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Quantitative Financial RL Metrics Calculator
# ==============================================================================

def compute_rl_quantitative_metrics(
    history: List[Dict[str, Any]],
    initial_cash: float = 100_000.0,
    risk_free_rate: float = 0.05,
) -> Dict[str, Any]:
    """Compute production-grade quantitative RL and financial portfolio metrics."""
    if not history or len(history) < 2:
        return {
            "final_pv": initial_cash,
            "net_return": 0.0,
            "ann_return": 0.0,
            "ann_vol": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "mdd": 0.0,
            "calmar": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "turnover": 0.0,
            "total_friction": 0.0,
            "action_entropy": 0.0,
        }

    pvs = np.array([h["portfolio_value"] for h in history], dtype=float)
    final_pv = float(pvs[-1])
    net_return = float((final_pv - initial_cash) / initial_cash)
    n_steps = len(pvs) - 1

    # Annualization factor (assuming 252 market days per year)
    ann_factor = 252.0 / max(1, n_steps)
    ann_return = float((1.0 + net_return) ** ann_factor - 1.0) if net_return > -1.0 else -1.0

    # Step returns
    step_rets = np.diff(pvs) / pvs[:-1]
    ann_vol = float(np.std(step_rets) * math.sqrt(252.0)) if len(step_rets) > 0 else 0.0

    # Sharpe Ratio (annualized)
    daily_rf = risk_free_rate / 252.0
    excess_rets = step_rets - daily_rf
    mean_excess = float(np.mean(excess_rets)) if len(excess_rets) > 0 else 0.0
    std_excess = float(np.std(excess_rets)) if len(excess_rets) > 0 else 0.0
    sharpe = float((mean_excess / (std_excess + 1e-9)) * math.sqrt(252.0)) if std_excess > 0 else 0.0

    # Sortino Ratio (downside risk only)
    downside = np.minimum(0.0, excess_rets)
    downside_vol = float(np.sqrt(np.mean(downside ** 2)) * math.sqrt(252.0)) if len(downside) > 0 else 0.0
    sortino = float((mean_excess * 252.0) / (downside_vol + 1e-9)) if downside_vol > 0 else 0.0

    # Max Drawdown
    peaks = np.maximum.accumulate(pvs)
    drawdowns = (peaks - pvs) / peaks
    mdd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Calmar Ratio
    calmar = float(ann_return / (mdd + 1e-9)) if mdd > 0 else 0.0

    # Win rate of active trades or profitable days
    positive_days = np.sum(step_rets > 0.0)
    win_rate = float(positive_days / max(1, len(step_rets)))

    # Action counts & policy action entropy
    actions = [h.get("action", 1) for h in history[1:]]
    total_trades = sum(1 for h in history if h.get("shares_traded", 0) > 0)
    total_friction = sum(float(h.get("step_cost", 0.0)) for h in history)

    # Action distribution
    if actions:
        counts = np.bincount(actions, minlength=3)
        probs = counts / float(len(actions))
        # Shannon entropy H(pi) = -sum p(a) log(p(a))
        entropy = -float(sum(p * math.log(p + 1e-12) for p in probs if p > 0))
    else:
        entropy = 0.0

    avg_pv = float(np.mean(pvs)) if len(pvs) > 0 else initial_cash
    total_traded_val = sum(float(h.get("shares_traded", 0) * h.get("price", 0)) for h in history)
    turnover = float(total_traded_val / (avg_pv + 1e-9))

    return {
        "final_pv": final_pv,
        "net_return": net_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "mdd": mdd,
        "calmar": calmar,
        "win_rate": win_rate,
        "total_trades": total_trades,
        "turnover": turnover,
        "total_friction": total_friction,
        "action_entropy": entropy,
    }


def clean_history_df(history: List[Dict[str, Any]]) -> pd.DataFrame:
    """Ensure history DataFrame always has required columns with graceful fallbacks."""
    if not history:
        return pd.DataFrame(columns=[
            "step", "action", "action_name", "price", "execution_price",
            "cash", "shares", "portfolio_value", "step_cost", "fee",
            "slippage", "shares_traded", "is_invalid", "is_violation",
            "drawdown", "reward"
        ])
    df = pd.DataFrame(history)
    defaults = {
        "step": 0,
        "action": 1,
        "action_name": "HOLD",
        "price": 0.0,
        "execution_price": 0.0,
        "shares_traded": 0,
        "cash": 100_000.0,
        "shares": 0,
        "portfolio_value": 100_000.0,
        "step_cost": 0.0,
        "fee": 0.0,
        "slippage": 0.0,
        "reward": 0.0,
        "is_invalid": False,
        "is_violation": False,
        "drawdown": 0.0,
    }
    for col, default_val in defaults.items():
        if col not in df.columns:
            if col == "execution_price" and "price" in df.columns:
                df[col] = df["price"]
            else:
                df[col] = default_val
    return df


# ==============================================================================
# Session State Initialization
# ==============================================================================

if "env" not in st.session_state:
    st.session_state.env = StockTradingEnv(task=1)
    st.session_state.env.reset(seed=42)
if "history" not in st.session_state:
    st.session_state.history = st.session_state.env.get_episode_history()
if "last_obs" not in st.session_state:
    st.session_state.last_obs = st.session_state.env._get_observation()
if "last_reward" not in st.session_state:
    st.session_state.last_reward = 0.0
if "last_reward_breakdown" not in st.session_state:
    st.session_state.last_reward_breakdown = {}
if "last_report" not in st.session_state:
    st.session_state.last_report = None


# ==============================================================================
# Sidebar: Control Plane & Task Configuration
# ==============================================================================

with st.sidebar:
    st.markdown("""
    <div style="padding: 4px 0 12px 0;">
        <span class="gym-badge">GYMNASIUM v1.0+</span>
        <span class="spec-badge">MDP CAUSAL</span>
    </div>
    """, unsafe_allow_html=True)
    st.title("⚡ StockRL Mini")
    st.caption("Lead RL Environment Engineer Telemetry Lab")

    st.divider()

    nav_mode = st.radio(
        "Navigation Plane",
        [
            "🔬 MDP Telemetry & Stepper",
            "📈 Policy Dynamics & Reward Attribution",
            "📊 Quantitative Benchmark Matrix",
            "🛡️ Zero-Trust Grader & Replay",
            "🚨 Adversarial Exploit Lab",
            "📉 Causal Market Data Lineage",
        ],
        index=0,
    )

    st.divider()
    st.subheader("Environment Configuration")

    selected_task_id = st.selectbox(
        "Task Difficulty Setting",
        options=[1, 2, 3],
        format_func=lambda x: f"Task {x}: {TASK_CONFIGS[x].name.split('— ')[-1]}",
        help="Task 1: Basic Trading (Zero Friction)\nTask 2: Transaction Costs & Slippage\nTask 3: Risk-Aware + Max Position Limit + Drawdown Penalty",
    )
    task_cfg = TASK_CONFIGS[selected_task_id]

    st.markdown(f"""
    - **Transaction Fee:** `{task_cfg.fee_rate * 100:.2f}%`
    - **Slippage Factor:** `{task_cfg.slippage * 100:.2f}%`
    - **Position Constraint:** `{'Unlimited' if task_cfg.max_position_shares is None else f'{task_cfg.max_position_shares} shares max'}`
    - **Drawdown Penalty:** `{task_cfg.drawdown_penalty_weight:.2f}x`
    """)

    seed_input = st.number_input("Environment Random Seed", value=42, step=1, help="Guarantees bitwise reproducible episode generation.")

    if st.button("🔄 Reset Environment", use_container_width=True, type="secondary"):
        st.session_state.env = StockTradingEnv(task=selected_task_id)
        obs, _ = st.session_state.env.reset(seed=int(seed_input))
        st.session_state.last_obs = obs
        st.session_state.history = st.session_state.env.get_episode_history()
        st.session_state.last_reward = 0.0
        st.session_state.last_reward_breakdown = {}
        st.session_state.last_report = None
        st.rerun()


# ==============================================================================
# Helper Header & KPI Dashboard
# ==============================================================================

def render_top_kpis(env: StockTradingEnv, history: List[Dict[str, Any]]):
    metrics = compute_rl_quantitative_metrics(history, initial_cash=env.initial_cash)
    cur_step = env.current_step
    cur_price = env.market_data.get_price(cur_step)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.markdown(f"""
        <div class="rl-card">
            <div class="rl-card-header">Portfolio Equity</div>
            <div class="rl-card-val">₹{metrics['final_pv']:,.0f}</div>
            <div class="rl-card-sub" style="color: {'#10b981' if metrics['net_return'] >= 0 else '#ef4444'};">
                {metrics['net_return'] * 100:+.2f}% Net Return
            </div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="rl-card">
            <div class="rl-card-header">Market Price (S_t)</div>
            <div class="rl-card-val">₹{cur_price:,.2f}</div>
            <div class="rl-card-sub">Step {cur_step} / {env.market_data.total_steps - 1}</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="rl-card">
            <div class="rl-card-header">Cash Balance</div>
            <div class="rl-card-val">₹{env.cash:,.0f}</div>
            <div class="rl-card-sub">Holdings: {env.shares} units</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="rl-card">
            <div class="rl-card-header">Annualized Sharpe</div>
            <div class="rl-card-val">{metrics['sharpe']:.2f}</div>
            <div class="rl-card-sub">Sortino: {metrics['sortino']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with k5:
        st.markdown(f"""
        <div class="rl-card">
            <div class="rl-card-header">Max Drawdown</div>
            <div class="rl-card-val">{metrics['mdd'] * 100:.2f}%</div>
            <div class="rl-card-sub">Current: {env.current_drawdown * 100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with k6:
        st.markdown(f"""
        <div class="rl-card">
            <div class="rl-card-header">Policy Action Entropy</div>
            <div class="rl-card-val">{metrics['action_entropy']:.3f}</div>
            <div class="rl-card-sub">{metrics['total_trades']} Trades | ₹{metrics['total_friction']:,.1f} Drag</div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# TAB 1: MDP TELEMETRY & INTERACTIVE STEPPER
# ==============================================================================

if nav_mode == "🔬 MDP Telemetry & Stepper":
    st.markdown("""
    <div class="telemetry-banner">
        <div>
            <div class="telemetry-title">Markov Decision Process (MDP) Telemetry & Step Engine</div>
            <div class="telemetry-sub">Observe causal feature tensors, step transition dynamics S_t ➔ A_t ➔ (S_{t+1}, R_t), and inspect reward decomposition.</div>
        </div>
        <div>
            <span class="gym-badge">ActionSpace: Discrete(3)</span>
            <span class="spec-badge">ObsSpace: Box(8)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    render_top_kpis(st.session_state.env, st.session_state.history)

    # Observation Vector Inspection Panel
    obs = st.session_state.env._get_observation()
    feature_labels = [
        ("0. P_norm (Normalized Price)", f"{obs[0]:.4f}", "P_t / P_0"),
        ("1. R_1d (1-Day Log Return)", f"{obs[1]:.4f}", "(P_t - P_{t-1}) / P_{t-1}"),
        ("2. R_5d (5-Day Rolling Return)", f"{obs[2]:.4f}", "(P_t - P_{t-5}) / P_{t-5}"),
        ("3. SMA_5 Ratio", f"{obs[3]:.4f}", "SMA_5(P_t) / P_t"),
        ("4. SMA_20 Ratio", f"{obs[4]:.4f}", "SMA_20(P_t) / P_t"),
        ("5. Cash Ratio", f"{obs[5]:.4f}", "Cash_t / PortfolioValue_t"),
        ("6. Position Ratio", f"{obs[6]:.4f}", "(Shares_t * P_t) / PortfolioValue_t"),
        ("7. Total Portfolio Return", f"{obs[7]:.4f}", "(PV_t - PV_0) / PV_0"),
    ]

    st.subheader("State Vector: o_t ∈ ℝ⁸ (Zero Lookahead Guarantee)")
    cols = st.columns(4)
    for i, (label, val, formula) in enumerate(feature_labels):
        col_idx = i % 4
        with cols[col_idx]:
            st.markdown(f"""
            <div class="obs-tensor" style="margin-bottom: 8px;">
                <div style="font-size: 11px; color: #94a3b8; font-weight: 600;">{label}</div>
                <div style="font-size: 18px; font-weight: 700; color: #38bdf8; margin: 2px 0;">{val}</div>
                <div style="font-size: 10px; color: #64748b;">Formula: <code>{formula}</code></div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # MDP Action Controls & Transition Execution
    st.subheader("Step Action Selector: A_t ∈ {0: SELL, 1: HOLD, 2: BUY}")

    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1, 1, 1, 2])
    with ctrl_col1:
        if st.button("🔴 ACTION 0: SELL (-10 Units)", use_container_width=True):
            obs, r, term, trunc, info = st.session_state.env.step(StockTradingEnv.ACTION_SELL)
            st.session_state.last_obs = obs
            st.session_state.last_reward = r
            st.session_state.history = st.session_state.env.get_episode_history()
            st.rerun()
    with ctrl_col2:
        if st.button("🟡 ACTION 1: HOLD (0 Units)", use_container_width=True, type="secondary"):
            obs, r, term, trunc, info = st.session_state.env.step(StockTradingEnv.ACTION_HOLD)
            st.session_state.last_obs = obs
            st.session_state.last_reward = r
            st.session_state.history = st.session_state.env.get_episode_history()
            st.rerun()
    with ctrl_col3:
        if st.button("🟢 ACTION 2: BUY (+10 Units)", use_container_width=True, type="primary"):
            obs, r, term, trunc, info = st.session_state.env.step(StockTradingEnv.ACTION_BUY)
            st.session_state.last_obs = obs
            st.session_state.last_reward = r
            st.session_state.history = st.session_state.env.get_episode_history()
            st.rerun()
    with ctrl_col4:
        sub_c1, sub_c2 = st.columns(2)
        with sub_c1:
            if st.button("⏩ Step Forward 20 Days", use_container_width=True):
                for _ in range(20):
                    if st.session_state.env.current_step < st.session_state.env.market_data.total_steps - 1:
                        obs, r, term, trunc, info = st.session_state.env.step(StockTradingEnv.ACTION_HOLD)
                st.session_state.history = st.session_state.env.get_episode_history()
                st.rerun()
        with sub_c2:
            if st.button("🤖 Rollout SMA Policy", use_container_width=True):
                agent = SMAAgent()
                while st.session_state.env.current_step < st.session_state.env.market_data.total_steps - 1:
                    o = st.session_state.env._get_observation()
                    inf = st.session_state.env._get_info()
                    a = agent.predict(o, inf)
                    st.session_state.env.step(a)
                st.session_state.history = st.session_state.env.get_episode_history()
                st.rerun()

    # Step Transition Summary
    if len(st.session_state.history) > 1:
        last_step_record = st.session_state.history[-1]
        step_exec_price = float(last_step_record.get("execution_price", last_step_record.get("price", 0.0)))
        step_r = float(last_step_record.get("reward", 0.0))
        step_cost = float(last_step_record.get("step_cost", 0.0))
        step_invalid = bool(last_step_record.get("is_invalid", False))
        step_violation = bool(last_step_record.get("is_violation", False))
        step_dd = float(last_step_record.get("drawdown", 0.0))

        st.markdown(f"""
        <div class="rl-card" style="margin-top: 14px; border-left: 4px solid #38bdf8;">
            <div style="font-weight: 700; font-size: 13px; color: #f1f5f9;">
                Latest Transition at Step {last_step_record.get('step', 0)}: Action <code>{last_step_record.get('action_name', 'HOLD')}</code> | 
                Execution Price: <code>₹{step_exec_price:,.2f}</code> | 
                Cost: <code>₹{step_cost:.2f}</code> | 
                Step Reward: <code style="color: {'#34d399' if step_r >= 0 else '#f87171'};">{step_r:+.5f}</code>
            </div>
            <div style="font-size: 11px; color: #94a3b8; margin-top: 4px;">
                Invalid Action: <code>{step_invalid}</code> | 
                Position Constraint Violation: <code>{step_violation}</code> | 
                Drawdown Drag: <code>{step_dd * 100:.2f}%</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Trajectory Visualization: Price & Asset Valuation
    hist_df = clean_history_df(st.session_state.history)
    t1, t2, t3 = st.tabs(["📊 Market & Trade Execution Trajectory", "💼 Equity & Asset Allocation", "📜 Step-by-Step Transition Ledger"])

    with t1:
        st.subheader("Price Series & Execution Markers")
        price_chart = alt.Chart(hist_df).mark_line(color="#38bdf8", strokeWidth=2).encode(
            x=alt.X("step:Q", title="Chronological Step (Trading Days)"),
            y=alt.Y("price:Q", title="Price (₹)", scale=alt.Scale(zero=False)),
            tooltip=["step", "price", "action_name", "portfolio_value"],
        ).properties(height=320)

        # Buy points
        buys = hist_df[hist_df["action"] == StockTradingEnv.ACTION_BUY]
        sells = hist_df[hist_df["action"] == StockTradingEnv.ACTION_SELL]

        buy_markers = alt.Chart(buys).mark_point(color="#34d399", size=80, shape="triangle-up", filled=True).encode(
            x="step:Q",
            y="price:Q",
            tooltip=["step", "action_name", "price", "shares_traded"],
        )
        sell_markers = alt.Chart(sells).mark_point(color="#f87171", size=80, shape="triangle-down", filled=True).encode(
            x="step:Q",
            y="price:Q",
            tooltip=["step", "action_name", "price", "shares_traded"],
        )
        st.altair_chart(price_chart + buy_markers + sell_markers, use_container_width=True)

    with t2:
        st.subheader("Portfolio Equity Dynamics & Drawdown Profile")
        col_eq, col_dd = st.columns(2)
        with col_eq:
            st.caption("Cash vs Stock Holding Valuation")
            pv_chart = alt.Chart(hist_df).mark_line(color="#818cf8", strokeWidth=2).encode(
                x=alt.X("step:Q", title="Step"),
                y=alt.Y("portfolio_value:Q", title="Portfolio Equity (₹)", scale=alt.Scale(zero=False)),
                tooltip=["step", "portfolio_value", "cash", "shares"],
            ).properties(height=260)
            st.altair_chart(pv_chart, use_container_width=True)
        with col_dd:
            st.caption("Underwater Drawdown Profile")
            dd_chart = alt.Chart(hist_df).mark_area(color="#ef4444", opacity=0.4).encode(
                x=alt.X("step:Q", title="Step"),
                y=alt.Y("drawdown:Q", title="Drawdown Ratio", axis=alt.Axis(format="%")),
                tooltip=["step", "drawdown"],
            ).properties(height=260)
            st.altair_chart(dd_chart, use_container_width=True)

    with t3:
        st.subheader("Chronological Transition History")
        cols_to_render = [
            c for c in [
                "step", "action_name", "price", "execution_price", "shares_traded",
                "cash", "shares", "portfolio_value", "step_cost", "reward",
                "is_invalid", "is_violation"
            ] if c in hist_df.columns
        ]
        st.dataframe(
            hist_df[cols_to_render].iloc[::-1],
            use_container_width=True,
            height=300,
        )


# ==============================================================================
# TAB 2: POLICY DYNAMICS & REWARD ATTRIBUTION
# ==============================================================================

elif nav_mode == "📈 Policy Dynamics & Reward Attribution":
    st.markdown("""
    <div class="telemetry-banner">
        <div>
            <div class="telemetry-title">Policy Action Dynamics & Reward Attribution Diagnostics</div>
            <div class="telemetry-sub">Analyze policy action distribution, empirical action entropy, and isolate reward components (returns vs fee drag vs risk penalties).</div>
        </div>
        <div>
            <span class="gym-badge">H(π) Entropy</span>
            <span class="spec-badge">Credit Attribution</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    hist_df = clean_history_df(st.session_state.history)

    if len(hist_df) < 3:
        st.info("Execute trades or rollout a baseline policy in the 'MDP Telemetry & Stepper' tab to inspect policy dynamics.")
    else:
        render_top_kpis(st.session_state.env, st.session_state.history)

        col_left, col_right = st.columns(2)

        # 1. Action Distribution & Policy Entropy
        with col_left:
            st.subheader("Policy Action Distribution & Entropy")
            action_counts = hist_df["action_name"].value_counts().reset_index()
            action_counts.columns = ["Action", "Frequency"]

            chart_act = alt.Chart(action_counts).mark_bar(cornerRadius=4).encode(
                x=alt.X("Action:N", title="Discrete Action"),
                y=alt.Y("Frequency:Q", title="Count"),
                color=alt.Color("Action:N", scale=alt.Scale(
                    domain=["BUY", "HOLD", "SELL"],
                    range=["#34d399", "#facc15", "#f87171"]
                )),
                tooltip=["Action", "Frequency"],
            ).properties(height=280)
            st.altair_chart(chart_act, use_container_width=True)

            # Compute empirical entropy
            probs = action_counts["Frequency"].values / action_counts["Frequency"].sum()
            entropy = -float(sum(p * math.log(p + 1e-12) for p in probs if p > 0))
            max_entropy = math.log(3)  # Uniform over 3 actions is ln(3) ~ 1.0986
            norm_entropy = entropy / max_entropy

            st.markdown(f"""
            - **Shannon Action Entropy:** `{entropy:.4f}` nats (Max possible: `{max_entropy:.4f}`)
            - **Normalized Exploration Entropy:** `{norm_entropy * 100:.1f}%`
            - **Interpretation:** High entropy denotes balanced exploration; low entropy signifies policy action collapse onto a single discrete choice.
            """)

        # 2. Reward Attribution Over Time
        with col_right:
            st.subheader("Step Reward Attribution Breakdown")
            reward_chart = alt.Chart(hist_df).mark_bar().encode(
                x=alt.X("step:Q", title="Step"),
                y=alt.Y("reward:Q", title="Reward R_t"),
                color=alt.condition(
                    alt.datum.reward >= 0,
                    alt.value("#34d399"),
                    alt.value("#f87171"),
                ),
                tooltip=["step", "reward", "action_name"],
            ).properties(height=280)
            st.altair_chart(reward_chart, use_container_width=True)

            cum_r = hist_df["reward"].sum()
            cum_friction = hist_df["step_cost"].sum()
            st.markdown(f"""
            - **Cumulative Reward:** `{cum_r:+.4f}`
            - **Cumulative Transaction Friction:** `₹{cum_friction:,.2f}`
            - **Reward Formulation:**
              $$R_t = \\text{{Portfolio Return}} - \\text{{Fee Penalty}} - \\text{{Drawdown Risk Penalty}}$$
            """)

        st.divider()

        # Simulated Policy Softmax Decision Boundary
        st.subheader("Simulated Policy Action Logits: π_θ(a | s)")
        st.caption("Demonstrates how a moving average crossover feature translates into Softmax action probabilities.")

        # Compute synthetic logits based on SMA ratios
        last_obs = st.session_state.env._get_observation()
        sma5_ratio = last_obs[3]   # SMA5 / Price
        sma20_ratio = last_obs[4]  # SMA20 / Price
        spread = (sma5_ratio - sma20_ratio) * 100.0

        logit_buy = float(spread * 2.0)
        logit_hold = 0.5
        logit_sell = float(-spread * 2.0)

        logits = np.array([logit_sell, logit_hold, logit_buy])
        exp_l = np.exp(logits - np.max(logits))
        softmax_probs = exp_l / np.sum(exp_l)

        logit_df = pd.DataFrame({
            "Action": ["SELL (0)", "HOLD (1)", "BUY (2)"],
            "Probability": softmax_probs,
            "Logit Value": logits,
        })

        sc1, sc2 = st.columns([1, 2])
        with sc1:
            st.dataframe(logit_df, use_container_width=True)
        with sc2:
            prob_chart = alt.Chart(logit_df).mark_bar(cornerRadius=4).encode(
                x=alt.X("Probability:Q", title="Softmax Probability π(a|s)", axis=alt.Axis(format="%")),
                y=alt.Y("Action:N", title="Action"),
                color=alt.Color("Action:N", scale=alt.Scale(
                    domain=["SELL (0)", "HOLD (1)", "BUY (2)"],
                    range=["#f87171", "#facc15", "#34d399"]
                )),
                tooltip=["Action", "Probability", "Logit Value"],
            ).properties(height=160)
            st.altair_chart(prob_chart, use_container_width=True)


# ==============================================================================
# TAB 3: QUANTITATIVE BENCHMARK MATRIX
# ==============================================================================

elif nav_mode == "📊 Quantitative Benchmark Matrix":
    st.markdown("""
    <div class="telemetry-banner">
        <div>
            <div class="telemetry-title">Quantitative Financial RL Benchmark Suite</div>
            <div class="telemetry-sub">Empirically evaluate Random Exploration, Passive Buy-and-Hold, and Heuristic SMA policies across Tasks 1, 2, and 3 under identical market seeds.</div>
        </div>
        <div>
            <span class="gym-badge">Reproducible Seeds</span>
            <span class="spec-badge">Risk-Adjusted Ratios</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    benchmark_task = st.selectbox(
        "Select Task Configuration to Benchmark",
        [1, 2, 3],
        format_func=lambda x: f"Task {x}: {TASK_CONFIGS[x].name.split('— ')[-1]}",
    )

    if st.button("🚀 Execute Comprehensive Benchmark Suite", type="primary"):
        agents = {
            "Random Exploration Agent": RandomAgent(seed=int(seed_input)),
            "Passive Buy-and-Hold Agent": BuyAndHoldAgent(),
            "Dual SMA Crossover Agent": SMAAgent(),
        }

        bench_data = []
        equity_curves = {}

        for name, agent in agents.items():
            test_env = StockTradingEnv(task=benchmark_task)
            obs, info = test_env.reset(seed=int(seed_input))
            agent.reset()

            done = False
            total_r = 0.0
            while not done:
                a = agent.predict(obs, info)
                obs, r, term, trunc, info = test_env.step(a)
                total_r += r
                done = term or trunc

            h = test_env.get_episode_history()
            m = compute_rl_quantitative_metrics(h, initial_cash=test_env.initial_cash)

            bench_data.append({
                "Policy / Agent": name,
                "Final Equity (₹)": f"₹{m['final_pv']:,.2f}",
                "Net Return (%)": f"{m['net_return'] * 100:+.2f}%",
                "Ann. Sharpe": f"{m['sharpe']:.2f}",
                "Ann. Sortino": f"{m['sortino']:.2f}",
                "Max Drawdown (%)": f"{m['mdd'] * 100:.2f}%",
                "Calmar Ratio": f"{m['calmar']:.2f}",
                "Total Trades": m['total_trades'],
                "Turnover": f"{m['turnover']:.2f}x",
                "Friction Drag (₹)": f"₹{m['total_friction']:,.2f}",
                "Action Entropy": f"{m['action_entropy']:.3f}",
            })

            # Record trajectory for overlay plot
            for record in h:
                step_idx = record["step"]
                if step_idx not in equity_curves:
                    equity_curves[step_idx] = {"step": step_idx}
                equity_curves[step_idx][name] = record["portfolio_value"]

        st.subheader("Quantitative Evaluation Matrix")
        st.dataframe(pd.DataFrame(bench_data), use_container_width=True)

        st.subheader("Multi-Policy Equity Trajectory Overlay")
        eq_df = pd.DataFrame(list(equity_curves.values())).melt(
            id_vars=["step"],
            var_name="Agent",
            value_name="Equity",
        )

        eq_chart = alt.Chart(eq_df).mark_line().encode(
            x=alt.X("step:Q", title="Trading Day (Step)"),
            y=alt.Y("Equity:Q", title="Portfolio Equity (₹)", scale=alt.Scale(zero=False)),
            color=alt.Color("Agent:N", scale=alt.Scale(
                domain=["Random Exploration Agent", "Passive Buy-and-Hold Agent", "Dual SMA Crossover Agent"],
                range=["#f59e0b", "#3b82f6", "#10b981"]
            )),
            tooltip=["step", "Agent", "Equity"],
        ).properties(height=360)
        st.altair_chart(eq_chart, use_container_width=True)


# ==============================================================================
# TAB 4: ZERO-TRUST GRADER & REPLAY
# ==============================================================================

elif nav_mode == "🛡️ Zero-Trust Grader & Replay":
    st.markdown("""
    <div class="telemetry-banner">
        <div>
            <div class="telemetry-title">Zero-Trust Independent Grader & Cryptographic Audit</div>
            <div class="telemetry-sub">Never trust reported rewards. Independently reconstruct episode state purely from action history + canonical market CSV.</div>
        </div>
        <div>
            <span class="gym-badge">SHA-256 Verified</span>
            <span class="spec-badge">Zero-Trust Replay</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    The independent grader executes a zero-trust replay of the agent's action history:
    1. **Canonical Price Replay:** Replays trades against read-only `sample_prices.csv` to verify price authenticity.
    2. **Cash & Stock Invariant Watchdog:** Proves $Cash_t \\ge 0$, $Shares_t \\ge 0$, and $Cash_t + Shares_t \\times P_t = PV_t$.
    3. **Friction Auditing:** Recomputes transaction fees and slippage independently.
    4. **Risk Bounds:** Validates position constraints ($Shares_t \\le Max$) and drawdown calculations.
    5. **Cryptographic Sealing:** Computes a SHA-256 integrity hash over the replayed state ledger.
    """)

    if st.button("🛡️ Execute Independent Audit Replay", type="primary"):
        st.session_state.last_report = Verifier.verify_episode(
            history=st.session_state.history,
            task_config=st.session_state.env.config,
            market_data=st.session_state.env.market_data,
            initial_cash=st.session_state.env.initial_cash,
        )

    if st.session_state.last_report:
        report: VerificationReport = st.session_state.last_report

        if report.passed:
            st.success("✅ **CRYPTOGRAPHIC AUDIT PASSED:** Canonical action replay verified 100% accounting and invariant consistency!")
        else:
            st.error("🚨 **AUDIT FAILED:** Discrepancies detected between reported history and independent replay!")

        st.code(f"SHA-256 State Integrity Seal: {report.integrity_hash}", language="bash")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Audited Final Equity", f"₹{report.final_portfolio_value:,.2f}")
        c2.metric("Audited Net Return", f"{report.net_return * 100:+.2f}%")
        c3.metric("Audited Transaction Costs", f"₹{report.total_transaction_costs:,.2f}")
        c4.metric("Invalid Transactions", f"{report.invalid_transactions}")

        st.subheader("Audited Discrepancies Ledger")
        if report.discrepancies:
            st.warning(f"Found {len(report.discrepancies)} discrepancy records:")
            st.json(report.discrepancies)
        else:
            st.info("Zero discrepancies found across all replayed chronological steps.")

        st.subheader("Canonical Verification Summary")
        st.code(report.summary(), language="text")


# ==============================================================================
# TAB 5: ADVERSARIAL EXPLOIT LAB
# ==============================================================================

elif nav_mode == "🚨 Adversarial Exploit Lab":
    st.markdown("""
    <div class="telemetry-banner">
        <div>
            <div class="telemetry-title">Adversarial Exploit & Vulnerability Demonstration Lab</div>
            <div class="telemetry-sub">Demonstrate deliberate exploitation attempts against the RL environment and grader, and verify how zero-trust defenses intercept them.</div>
        </div>
        <div>
            <span class="gym-badge">Monkeypatch Attack</span>
            <span class="spec-badge">State Spoofing</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    attack_type = st.selectbox(
        "Select Attack Vector to Launch",
        [
            "Exploit 1: Grading Code In-Memory Monkeypatching",
            "Exploit 2A: State Manipulation — Cash Balance Inflation (+₹50,000)",
            "Exploit 2B: State Manipulation — Execution Price Spoofing (Sell at ₹10,000)",
            "Exploit 2C: State Manipulation — Fee Evasion in Task 2 (Claim ₹0 Costs)",
        ],
    )

    exploit_env = StockTradingEnv(task=2)

    if st.button("💥 Launch Adversarial Exploit & Test Defense", type="primary"):
        if "Exploit 1" in attack_type:
            st.subheader("Attacking Grader via In-Memory Monkeypatching")
            succeeded, explanation = Exploit1TestManipulation.attempt_monkeypatch_exploit([], 1)
            verifier_file = PROJECT_ROOT / "grader" / "verifier.py"
            real_hash = Exploit1TestManipulation.get_verifier_hash(verifier_file)

            st.warning(f"**Exploit Mechanism:** {explanation}")
            st.error("⚠️ **Local File Threat Model:** Any code running with user privileges can monkeypatch in-memory Python classes!")
            st.success(f"🛡️ **Zero-Trust Defense:** External CI/CD SHA-256 verification checks `verifier.py` against trusted digest: `{real_hash}`.")

        elif "Exploit 2A" in attack_type:
            st.subheader("Attempting Cash Balance Inflation (+₹50,000)")
            rep = Exploit2StateAndRewardManipulation.simulate_cash_inflation_exploit(exploit_env)
            st.error(f"🚨 **Defense Caught Attack!** Verifier detected {len(rep.discrepancies)} discrepancy records:")
            st.json(rep.discrepancies)
            st.info("The independent grader recomputed cash flows from execution prices and caught the injected funds.")

        elif "Exploit 2B" in attack_type:
            st.subheader("Attempting Execution Price Spoofing")
            rep = Exploit2StateAndRewardManipulation.simulate_price_spoofing_exploit(exploit_env)
            st.error(f"🚨 **Defense Caught Attack!** Verifier detected price divergence from canonical CSV:")
            st.json(rep.discrepancies)
            st.info("The grader looks up canonical CSV prices directly rather than trusting reported prices.")

        elif "Exploit 2C" in attack_type:
            st.subheader("Attempting Fee Evasion in Task 2")
            rep = Exploit2StateAndRewardManipulation.simulate_fee_evasion_exploit(exploit_env)
            st.error(f"🚨 **Defense Caught Attack!** Verifier caught fee evasion:")
            st.json(rep.discrepancies)
            st.info("The grader enforces Task 2's 0.10% transaction fee and caught the discrepancy.")


# ==============================================================================
# TAB 6: CAUSAL MARKET DATA LINEAGE
# ==============================================================================

elif nav_mode == "📉 Causal Market Data Lineage":
    st.markdown("""
    <div class="telemetry-banner">
        <div>
            <div class="telemetry-title">Causal Market Data Lineage & No-Lookahead Proof</div>
            <div class="telemetry-sub">Inspect the 252-day synthetic Geometric Brownian Motion dataset and verify causal feature engineering.</div>
        </div>
        <div>
            <span class="gym-badge">Synthetic GBM</span>
            <span class="spec-badge">Zero Lookahead</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    md = st.session_state.env.market_data

    st.subheader("Synthetic Price Trajectory (252 Trading Days)")
    st.caption("Generated via Geometric Brownian Motion (S_0 = ₹1,000, μ = 12%, σ = 20%, seed = 42). Explicitly labeled as synthetic.")

    md_df = md.df.copy()
    if "step" not in md_df.columns:
        md_df["step"] = np.arange(len(md_df))
    md_df["date_str"] = md_df["date"].astype(str)

    chart_price = alt.Chart(md_df).mark_line(color="#38bdf8", strokeWidth=2).encode(
        x=alt.X("step:Q", title="Step"),
        y=alt.Y("close:Q", title="Closing Price (₹)", scale=alt.Scale(zero=False)),
        tooltip=["step", "date_str", "close", "high", "low", "volume"],
    ).properties(height=320)
    st.altair_chart(chart_price, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Descriptive Summary Statistics")
        st.dataframe(md_df.describe(), use_container_width=True)
    with col2:
        st.subheader("First 10 Chronological Rows")
        st.dataframe(md_df.head(10), use_container_width=True)

    st.markdown("""
    ### Mathematical Causality Guarantee
    For any time step $t \\in [0, T-1]$, all features in observation vector $o_t$ depend strictly on $P_{0:t}$.
    - $P_{norm}(t) = P_t / P_0$
    - $R_{1d}(t) = (P_t - P_{t-1}) / P_{t-1}$
    - $R_{5d}(t) = (P_t - P_{t-5}) / P_{t-5}$
    - $SMA_k(t) = \\frac{1}{k} \\sum_{i=0}^{k-1} P_{t-i}$
    No future price $P_{t+1}, P_{t+2}, \\dots$ is ever visible to the agent's observation tensor.
    """)
