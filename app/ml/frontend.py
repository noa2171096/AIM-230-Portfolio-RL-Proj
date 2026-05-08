import os
import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# This always works regardless of working directory
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH = os.path.join(BASE_DIR, "data", "datasets_2015-01-01_2023-12-31.pkl")

print(f"Dataset path: {DATASET_PATH}")  # add temporarily to verify
# ── Path fix so sibling modules are found ────────────────────────────────
sys.path.append(os.path.dirname(__file__))

from env import InvestingEnvRich, build_features, FEATURE_COLS
from model import InvestingAgent

# ── Load existing dataset ─────────────────────────────────────────────────
import pickle

#DATASET_PATH = r"data/datasets_2015-01-01_2023-12-31.pkl"

with open(DATASET_PATH, "rb") as f:
    datasets = pickle.load(f)

print(f"Loaded {len(datasets)} tickers: {list(datasets.keys())}")
for t, df in datasets.items():
    print(f"  {t}: {len(df)} rows")

def generate_recommendation(
    user_tickers,
    horizon_days       = 21,
    episodes           = 200,
    capital            = 10_000.0,
    save_dir           = "saved_model",
    force_retrain      = False,
    preloaded_datasets = None,             
):
    ticker_str = "_".join(sorted(user_tickers))
    model_path = os.path.join(save_dir, f"best_agent.pt")
    os.makedirs(save_dir, exist_ok=True)
    data_path  = os.path.join(save_dir, f"datasets_{ticker_str}.pkl")


    # ── Use preloaded data if provided ────────────────────────────────────
    if preloaded_datasets is not None:
        print("Using preloaded datasets — skipping download")
        datasets = preloaded_datasets

    elif os.path.exists(data_path) and not force_retrain:
        print(f"Loading cached data: {data_path}")
        with open(data_path, "rb") as f:
            datasets = pickle.load(f)

    else:
        print("Downloading and building features...")
        datasets = {}
        start_date = "2015-01-01"
        end_date   = datetime.today().strftime("%Y-%m-%d")

        for t in user_tickers:
            try:
                # Option B — build_features handles download internally
                feat = build_features(t, start_date, end_date)
                feat = feat.reset_index()
                feat.rename(columns={"index": "date", "Date": "date"},
                            errors="ignore", inplace=True)
                feat.columns = [c.lower() for c in feat.columns]
                feat = feat.loc[:, ~feat.columns.duplicated()]

                # Fill any missing feature columns with 0
                for col in FEATURE_COLS:
                    if col not in feat.columns:
                        feat[col] = 0.0

                if len(feat) > 300:
                    datasets[t] = feat
                    print(f"  {t}: {len(feat)} rows")
                else:
                    print(f"  {t}: skipped — insufficient data ({len(feat)} rows)")

            except Exception as e:
                print(f"  {t}: failed — {e}")

        if len(datasets) < 2:
            raise ValueError(
                f"Need at least 2 valid tickers, only got {list(datasets.keys())}"
            )

        with open(data_path, "wb") as f:
            pickle.dump(datasets, f)
        print(f"Data saved: {data_path}")

    # Always set valid after either branch
    valid = list(datasets.keys())
    print(f"\nValid tickers: {valid}")

    # ── Build environment ─────────────────────────────────────────────────
    val_start = "2022-01-01"


    env = InvestingEnvRich(
        datasets     = datasets,
        tickers      = valid,
        steps        = 252,
        random_start = True,
        val_start    = val_start,
    )

    state_dim = env.observation_space.shape[0]
    input_dim = state_dim + env.n_assets        # combine before sizing
    hidden    = max(64, (input_dim * 2 // 32) * 32)
    agent     = InvestingAgent(env, hidden=hidden)

    # ── Load existing checkpoint or train fresh ───────────────────────────
    if os.path.exists(model_path) and not force_retrain:
        print(f"\nLoading existing model: {model_path}")
        agent.load(model_path)
        agent.epsilon = 0.0
        print("Skipping training — using saved weights")

    else:
        print(f"\nTraining from scratch ({episodes} episodes)...")
        best_reward = -np.inf

        for ep in range(1, episodes + 1):
            state, _  = env.reset()
            ep_reward = 0.0

            while True:
                action            = agent.act(state)
                ns, r, done, _, _ = env.step(action)
                agent.remember(state, action, r, ns, done)
                agent.replay()
                ep_reward += r
                state      = ns
                if done: break

            if ep % agent.target_update == 0:
                agent.update_target()

            if ep_reward > best_reward:
                best_reward = ep_reward
                agent.save(model_path)

            if ep % 50 == 0:
                print(f"  Ep {ep:3d}/{episodes} | "
                      f"ε={agent.epsilon:.3f} | "
                      f"reward={ep_reward:+.3f} | "
                      f"value=${env.portfolio_value:.4f}")

        agent.load(model_path)
        agent.epsilon = 0.0
        print(f"Training complete. Best reward: {best_reward:.3f}")

    # ── Build today's state ───────────────────────────────────────────────
    print("\nBuilding today's state...")
    today_features = []
    current_prices = {}

    for t in valid:
        df     = datasets[t]
        latest = df.iloc[-1]
        raw_f  = latest[FEATURE_COLS].values.reshape(1, -1)
        scaled = env.scalers[t].transform(raw_f).clip(-5, 5)
        today_features.append(scaled.squeeze())
        current_prices[t] = float(latest["close"])

    equal_weights = np.ones(len(valid)) / len(valid)
    today_state   = np.concatenate(today_features + [equal_weights])

    assert today_state.shape[0] == env.observation_space.shape[0], \
        f"State dim mismatch: {today_state.shape[0]} != {env.observation_space.shape[0]}"

    optimal_weights = agent.opt_action(today_state)

    # ── Val period rollout — proxy for near-term performance ──────────────
    print("Running validation rollout...")
    val_steps = min(horizon_days, len(env.val_dates) - 1)

    if val_steps < 2:
        print("WARNING: not enough val dates for rollout — using training env stats")
        expected_return = 0.0
        sharpe          = 0.0
        max_dd          = 0.0
    else:
        val_env = InvestingEnvRich(
            datasets     = datasets,
            tickers      = valid,
            steps        = val_steps,
            random_start = False,
            val_start    = val_start,
        )
        state, _ = val_env.reset(use_val=True)
        while True:
            action = agent.opt_action(state)
            state, _, done, _, info = val_env.step(action)
            if done: break

        expected_return = val_env.total_return
        sharpe          = val_env.sharpe_ratio
        max_dd          = val_env.max_drawdown

    # ── Build result dict ─────────────────────────────────────────────────
    allocation = {}
    for t, w in zip(valid, optimal_weights):
        dollar_val = capital * w
        action     = "BUY" if w > 0.20 else "HOLD" if w > 0.08 else "REDUCE"
        allocation[t] = {
            "weight":        round(float(w), 4),
            "weight_pct":    round(float(w) * 100, 1),
            "dollar_value":  round(dollar_val, 2),
            "action":        action,
            "current_price": current_prices[t],
            "shares_to_buy": round(dollar_val / (current_prices[t] + 1e-8), 4),
        }

    result = {
        "tickers":    valid,
        "allocation": allocation,
        "portfolio": {
            "expected_return_pct": round(expected_return * 100, 2),
            "sharpe_ratio":        round(sharpe, 3),
            "max_drawdown_pct":    round(max_dd * 100, 2),
            "final_value":         round(capital * (1 + expected_return), 2),
        },
        "horizon_days": horizon_days,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n{'═'*52}")
    print(f"  PORTFOLIO RECOMMENDATION")
    print(f"  Generated: {result['generated_at']}")
    print(f"  Horizon:   {horizon_days} trading days")
    print(f"{'═'*52}")
    print(f"\n  Expected return: {result['portfolio']['expected_return_pct']:+.2f}%")
    print(f"  Sharpe ratio:    {result['portfolio']['sharpe_ratio']:.3f}")
    print(f"  Max drawdown:    {result['portfolio']['max_drawdown_pct']:.2f}%")
    print(f"  Final value:     ${result['portfolio']['final_value']:,.2f}")
    print(f"\n  Allocation:")
    for t, a in sorted(allocation.items(), key=lambda x: -x[1]["weight"]):
        bar = "█" * int(a["weight_pct"] / 3)
        print(f"    [{a['action']:6s}] {t:6s}  {a['weight_pct']:5.1f}%  "
              f"${a['dollar_value']:>8,.2f}  "
              f"{a['shares_to_buy']:.3f} shares  {bar}")
    print(f"{'═'*52}\n")

    return result


# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = generate_recommendation(
        user_tickers  = list(datasets.keys()),  # use whatever tickers are in the file
        horizon_days  = 21,
        episodes      = 200,
        capital       = 10_000.0,
        force_retrain = False,
        preloaded_datasets = datasets,          # pass datasets directly
    )