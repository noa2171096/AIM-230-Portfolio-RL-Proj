import math
import random
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import minimize
from collections import deque
from sklearn.preprocessing import RobustScaler
from env import build_features
import warnings
warnings.filterwarnings("ignore")

"""
TICKERS    = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
START      = "2015-01-01"
END        = "2023-12-31"
EPISODES   = 200
STEPS      = 252          # trading days per episode
"""

def download(tickers, start, end):
    data = {}
    for t in tickers:
        df = yf.download(t, start=start, end=end,
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        data[t] = df
        print(f"  {t}: {len(df)} days")
    return data

#raw = download(TICKERS, START, END)

FEATURE_COLS = [
    "return_5d","return_30d","return_60d",
    "ma_5d_ratio","ma_30d_ratio","ma_60d_ratio","sma50_vs_200",
    "macd_norm","macd_signal","rsi_norm",
    "bb_pct_b","atr","vol_regime","vol_zscore","volume_ratio",
]

"""
featured = {}
for t, df in raw.items():
    featured[t] = build_features(df)
    print(f"  {t}: {len(featured[t])} rows, {len(FEATURE_COLS)} features")

VAL_START  = "2022-01-01"

splits = {}
for t, df in featured.items():
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    train = d[d["date"] <  pd.Timestamp(VAL_START)].copy()
    val   = d[d["date"] >= pd.Timestamp(VAL_START)].copy()

    scaler = RobustScaler()
    train[FEATURE_COLS] = scaler.fit_transform(train[FEATURE_COLS]).clip(-5,5)
    val[FEATURE_COLS]   = scaler.transform(val[FEATURE_COLS]).clip(-5,5)

    splits[t] = {
        "train": train.reset_index(drop=True),
        "val":   val.reset_index(drop=True),
    }
    print(f"  {t}: train={len(train)}d  val={len(val)}d")
"""

class observation_space:
    def __init__(self, n):
        self.shape = (n,)

class action_space:
    def __init__(self, n):
        self.n = n
    def seed(self, seed):
        random.seed(seed)
    def sample(self):
        rn = np.random.random(3)
        return rn / rn.sum()

class QNetwork(nn.Module):
    """
    Outputs a single Q-value for a (state, action) pair.
    Continuous action space — we optimize over actions at inference.
    Same architecture as the book's Keras model, just in PyTorch.
    """
    def __init__(self, n_features, hidden=256):
        super().__init__()
        # State (2N) + Action (N) concatenated as input
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),         # single Q-value output
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, state, action):
        # Concatenate state and action before passing through network
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

class InvestingAgent:
    """
    Continuous-action DQL agent for N-asset portfolio allocation.
    Mirrors the book's InvestingAgent but in PyTorch.

    Key design: Q(s, a) → scalar. At inference, we run scipy.optimize
    to find the action (weight vector) that maximizes Q(s, a).
    This is identical to the book's opt_action() approach.
    """

    def __init__(self, env, hidden=128, lr=1e-4, gamma=0.99,
                epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995,
                batch_size=32, memory_size=5000, target_update=10):

        self.env           = env
        self.n_assets      = env.n_assets
        self.state_dim     = env.observation_space.shape[0]
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update      # ← this line was missing
        self.memory        = deque(maxlen=memory_size)
        self.device        = "cuda" if torch.cuda.is_available() else "cpu"

        input_dim   = self.state_dim + self.n_assets
        self.model  = QNetwork(input_dim, hidden).to(self.device)
        self.target = QNetwork(input_dim, hidden).to(self.device)

        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        print(f"InvestingAgent | assets={self.n_assets} | state={self.state_dim} | "
              f"device={self.device} | hidden={hidden} | "
              f"params={sum(p.numel() for p in self.model.parameters()):,}")

    # ── Action selection ──────────────────────────────────────────────────

    def opt_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        bounds  = [(0, 1)] * self.n_assets
        cons    = [{"type": "eq", "fun": lambda x: x.sum() - 1}]

        # Warm start: last N elements of state are current weights
        x0 = np.array(state[-self.n_assets:], dtype=np.float64)
        x0 = np.clip(x0, 0, 1)
        x0 = x0 / (x0.sum() + 1e-8)   # ensure valid starting point

        def neg_q(w):
            w_t = torch.FloatTensor(w).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q = self.model(state_t, w_t).item()
            pen = np.mean((x0 - w) ** 2)
            return -(q - pen)

        result = minimize(
            neg_q,
            x0          = x0,
            bounds      = bounds,
            constraints = cons,
            method      = "SLSQP",
            options     = {"eps": 1e-4, "maxiter": 50, "ftol": 1e-6},
        )
        return result["x"]

    def act(self, state):
        """Epsilon-greedy: random allocation or optimized allocation."""
        if random.random() <= self.epsilon:
            return self.env.action_space.sample()
        return self.opt_action(state)

    # ── Memory + replay ───────────────────────────────────────────────────

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        total_loss = 0.0

        for state, action, reward, next_state, done in batch:
            # Target Q-value
            if done:
                target = reward
            else:
                # Book approach: find best action in next state using target net
                next_action = self.opt_action(next_state)
                ns_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                na_t = torch.FloatTensor(next_action).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    target = reward + self.gamma * self.target(ns_t, na_t).item()

            # Current Q-value
            s_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            a_t = torch.FloatTensor(action).unsqueeze(0).to(self.device)
            q   = self.model(s_t, a_t)

            loss = F.smooth_l1_loss(q, torch.tensor([[target]],
                                    dtype=torch.float32).to(self.device))
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            self.optimizer.step()
            total_loss += loss.item()

        # Decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return total_loss / self.batch_size

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())

    def save(self, path):
        torch.save({
            "model":   self.model.state_dict(),
            "target":  self.target.state_dict(),
            "epsilon": self.epsilon,
        }, path)
        print(f"Saved → {path}")

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ck["model"])
        self.target.load_state_dict(ck["target"])
        self.epsilon = ck["epsilon"]
        print(f"Loaded ← {path} (ε={self.epsilon:.3f})")
        
    # ── Train ─────────────────────────────────────────────────────────────

    def train(self, episodes=100, target_update=10, verbose=True):
        history = []
        for e in range(1, episodes + 1):
            state, _ = self.env.reset()
            treward  = 0.0
            losses   = []

            for _ in range(len(self.env.data) - 1):
                action               = self.act(state)
                next_state, reward, done, _, _ = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                loss = self.replay()
                if loss: losses.append(loss)
                treward += reward
                state    = next_state
                if done:
                    break

            if e % target_update == 0:
                self.update_target()

            final_pv = self.env.portfolio_value
            history.append({
                "episode":  e,
                "reward":   treward,
                "loss":     np.mean(losses) if losses else 0,
                "pv":       final_pv,
                "epsilon":  self.epsilon,
            })

            if verbose and e % 10 == 0:
                print(f"Ep {e:3d} | ε={self.epsilon:.3f} | "
                      f"reward={treward:+.3f} | "
                      f"portfolio=${final_pv:.4f} | "
                      f"loss={np.mean(losses) if losses else 0:.5f}")

        return pd.DataFrame(history)

    # ── Test ──────────────────────────────────────────────────────────────

    def test(self, episodes=5, verbose=True):
        saved_eps    = self.epsilon
        self.epsilon = 0.0    # pure exploitation

        for e in range(1, episodes + 1):
            state, _ = self.env.reset()
            treward  = 0.0

            for _ in range(len(self.env.data) - 1):
                action = self.opt_action(state)
                state, reward, done, _, _ = self.env.step(action)
                treward += reward
                if done:
                    break

            final_pv = self.env.portfolio_value
            if verbose:
                print(f"Test Ep {e} | reward={treward:+.4f} | "
                      f"portfolio=${final_pv:.4f}")

        self.epsilon = saved_eps