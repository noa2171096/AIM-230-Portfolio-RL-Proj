import numpy as np
from ml.model import InvestingAgent
from ml.env import InvestingEnvRich

TICKERS    = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
START      = "2015-01-01"
END        = "2023-12-31"
EPISODES   = 200
STEPS      = 252          # trading days per episode
VAL_START  = "2022-01-01"


import pickle

with open("data/datasets_2015-01-01_2023-12-31.pkl", "rb") as f:
    datasets = pickle.load(f)

print(f"Loaded {len(datasets)} tickers: {list(datasets.keys())}")

from datetime import datetime
import os

# Creates a unique folder for every run e.g. saved_models/run_20240315_143022
RUN_ID   = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = os.path.join("saved_model", f"run_{RUN_ID}")
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Save directory: {SAVE_DIR}")

train_env = InvestingEnvRich(
    datasets     = datasets,
    tickers      = TICKERS,
    steps        = STEPS,
    random_start = True,
    val_start    = VAL_START,
)

state_dim = train_env.observation_space.shape[0]
n_assets  = train_env.n_assets
hidden    = max(64, ((state_dim + n_assets) * 2 // 32) * 32)
agent     = InvestingAgent(train_env, hidden=hidden)

best_reward = -np.inf

for ep in range(1, EPISODES + 1):
    state, _ = train_env.reset()
    ep_reward = 0.0
    losses    = []

    while True:
        action            = agent.act(state)           # ← removed training=True
        ns, r, done, _, _ = train_env.step(action)
        agent.remember(state, action, r, ns, done)
        loss = agent.replay()
        if loss: losses.append(loss)
        ep_reward += r
        state      = ns
        if done: break

    if ep % agent.target_update == 0:
        agent.update_target()

    if ep_reward > best_reward:
        best_reward = ep_reward
        agent.save(f"{SAVE_DIR}/best_agent.pt")

    if ep % 10 == 0:
        avg_loss = np.mean(losses) if losses else 0
        pv       = train_env.portfolio_value
        print(f"Ep {ep:3d}/{EPISODES} | "
              f"ε={agent.epsilon:.3f} | "
              f"reward={ep_reward:+.3f} | "
              f"value=${pv:.4f} | "
              f"loss={avg_loss:.5f}")