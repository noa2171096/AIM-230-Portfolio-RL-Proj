import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
import finnhub
import time
import random

tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
finbert   = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
finbert.eval()

def get_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        logits = finbert(**inputs).logits
    probs  = F.softmax(logits, dim=-1).squeeze()
    score  = probs[0].item() - probs[1].item()  # positive - negative
    label  = ["positive", "negative", "neutral"][probs.argmax().item()]
    return score, label

FINNHUB_KEY = "d7qflfhr01qi8jan9efgd7qflfhr01qi8jan9eg0"  # free at finnhub.io

def fetch_daily_sentiment(ticker, start_date, end_date):
    """
    For each trading day, fetch prior-day headlines and score with FinBERT.
    Returns a DataFrame with daily sentiment features, index = Date.

    Uses a 1-day lag — yesterday's news informs today's action.
    This is mandatory to prevent lookahead bias.
    """
    client       = finnhub.Client(api_key=FINNHUB_KEY)
    trading_days = pd.bdate_range(start=start_date, end=end_date)
    records      = []

    for i, date in enumerate(trading_days):
        # 1-day lag: use news from previous calendar day
        news_date  = date - pd.Timedelta(days=1)
        date_str   = date.strftime("%Y-%m-%d")
        news_str   = news_date.strftime("%Y-%m-%d")

        try:
            news = client.company_news(ticker, _from=news_str, to=date_str)
            headlines = [n["headline"] for n in news[:10]]  # cap at 10

            if headlines:
                scores = [get_sentiment(h)[0] for h in headlines]
                raw    = float(np.mean(scores))
            else:
                raw    = 0.0   # neutral if no news

            records.append({"date": date, "sentiment_raw": raw})

        except Exception:
            records.append({"date": date, "sentiment_raw": 0.0})

        # Rate limit: Finnhub free = 60 calls/min
        if i % 50 == 0 and i > 0:
            print(f"  {i}/{len(trading_days)} days scored")
            time.sleep(1)

    df = pd.DataFrame(records).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df


def build_sentiment_features(sentiment_df):
    """
    Derives richer features from raw daily sentiment score.
    All features are already 1-day lagged from fetch_daily_sentiment.
    """
    s = sentiment_df["sentiment_raw"].copy()

    sentiment_df["sentiment_3d_ema"]    = s.ewm(span=3).mean()
    sentiment_df["sentiment_momentum"]  = s.diff(3)
    sentiment_df["sentiment_zscore"]    = (
        (s - s.rolling(20).mean()) / (s.rolling(20).std() + 1e-8)
    )
    sentiment_df["sentiment_shock"]     = (
        sentiment_df["sentiment_zscore"].abs() > 2.0
    ).astype(float)

    return sentiment_df


SENTIMENT_COLS = [
    "sentiment_raw",
    "sentiment_3d_ema",
    "sentiment_momentum",
    "sentiment_zscore",
    "sentiment_shock",
]

def build_features(ticker, start_date, end_date, use_sentiment=True):


    # ── 1. OHLCV from yfinance ────────────────────────────────────────────
    data = yf.download(ticker, start=start_date, end=end_date,
                       auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.index = pd.to_datetime(data.index)
    data.index.name = "date"

    # ── 2. Your existing features ─────────────────────────────────────────
    data["return_5d"]  = data["Close"].pct_change(5)
    data["return_30d"] = data["Close"].pct_change(30)
    data["return_60d"] = data["Close"].pct_change(60)

    data["ma_5d"]  = data["Close"].rolling(5).mean()
    data["ma_30d"] = data["Close"].rolling(30).mean()
    data["ma_60d"] = data["Close"].rolling(60).mean()

    data["ema12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["ema26"] = data["Close"].ewm(span=26, adjust=False).mean()
    data["macd"]  = data["ema12"] - data["ema26"]

    delta    = data["Close"].diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    data["rsi"] = 100 - 100 / (1 + avg_gain / (avg_loss + 1e-8))

    # Normalize MAs relative to price so they're scale-free
    data["ma_5d_ratio"]  = data["ma_5d"]  / data["Close"]
    data["ma_30d_ratio"] = data["ma_30d"] / data["Close"]
    data["ma_60d_ratio"] = data["ma_60d"] / data["Close"]
    data["macd_norm"]    = data["macd"]   / data["Close"]
    data["rsi_norm"]     = data["rsi"]    / 100.0   # scale to [0,1]

    if use_sentiment:
        print(f"Scoring sentiment for {ticker}...")
        sent_raw  = fetch_daily_sentiment(ticker, start_date, end_date)
        sent_full = build_sentiment_features(sent_raw)

        # Align on trading days — left join keeps all price rows
        data = data.join(sent_full, how="left")
        # Fill gaps (weekends already excluded, but some days may miss news)
        data[SENTIMENT_COLS] = data[SENTIMENT_COLS].fillna(0.0)
    else:
        for col in SENTIMENT_COLS:
            data[col] = 0.0

    if use_sentiment:
        def normalize_series(s):
            return (s - s.rolling(60).mean()) / (s.rolling(60).std() + 1e-8)

        data["sent_price_divergence"] = (
            normalize_series(data["return_5d"]) -
            normalize_series(data["sentiment_3d_ema"])
        )
    else:
        data["sent_price_divergence"] = 0.0

    drop = ["ema12", "ema26", "ma_5d", "ma_30d", "ma_60d", "macd", "rsi"]
    data.drop(columns=drop, inplace=True)
    data.dropna(inplace=True)
    data.reset_index(inplace=True)

    print(f"{ticker}: {len(data)} clean rows, {len(data.columns)} columns")
    return data


FEATURE_COLS = [
    "return_5d", "return_30d", "return_60d",
    "ma_5d_ratio", "ma_30d_ratio", "ma_60d_ratio",
    "macd_norm", "rsi_norm",
    "sentiment_raw", "sentiment_3d_ema",
    "sentiment_momentum", "sentiment_zscore",
    "sentiment_shock", "sent_price_divergence",
]

class observation_space:
    def __init__(self, n):
        self.shape = (n,)

class action_space:
    def __init__(self, n_assets):
        self.n = n_assets
    def seed(self, seed):
        random.seed(seed)
    def sample(self):
        # Random allocation that sums to 1
        w = np.random.random(self.n_assets)
        return w / w.sum()

from sklearn.preprocessing import RobustScaler

import gymnasium as gym

class InvestingEnvRich(gym.Env):
    """
    N-asset portfolio environment using full feature dataset.

    State:  [features_asset1..N (n_feat dims each), weights (N dims)]
    Action: weight vector [w1..wN], sums to 1, all >= 0
    Reward: rolling Sharpe ratio
    """

    def __init__(self, datasets, tickers, steps=252,
                 amount=1.0, random_start=True, val_start="2023-01-01"):

        self.tickers         = tickers
        self.n_assets        = len(tickers)
        self.steps           = steps
        self.initial_balance = amount
        self.random_start    = random_start
        self.datasets        = datasets
        self.n_feat          = len(FEATURE_COLS)

        state_dim = self.n_feat * self.n_assets + self.n_assets
        self.observation_space = observation_space(state_dim)
        self.action_space      = action_space(self.n_assets)
        self.action_space.n_assets = self.n_assets
        self.episode    = 0
        self.portfolios = pd.DataFrame()

        self._align_data(val_start)
        self._fit_scalers()


    def _clean_dates(self, df):
        """Parse dates, strip timezone, normalise to midnight. Always returns clean Timestamps."""
        dates = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        return dates

    def _align_data(self, val_start):
        """Find common trading days across all tickers for train and val splits."""
        val_ts = pd.Timestamp(val_start).normalize()

        train_sets = []
        val_sets   = []

        for t in self.tickers:
            df    = self.datasets[t].copy()
            dates = self._clean_dates(df)
            train_sets.append(set(dates[dates <  val_ts]))
            val_sets.append(  set(dates[dates >= val_ts]))

        train_common = sorted(set.intersection(*train_sets))
        val_common   = sorted(set.intersection(*val_sets))

        self.train_dates = [pd.Timestamp(d) for d in train_common]
        self.val_dates   = [pd.Timestamp(d) for d in val_common]

        # Sanity check
        if len(self.train_dates) == 0:
            for t in self.tickers:
                df    = self.datasets[t].copy()
                dates = self._clean_dates(df)
                print(f"  {t}: {dates.min().date()} → {dates.max().date()} "
                      f"({len(dates)} days)")
            raise ValueError("No common training dates found — see ranges above")

        print(f"Aligned: {len(self.train_dates)} train days, "
              f"{len(self.val_dates)} val days "
              f"across {self.n_assets} tickers")

    def _fit_scalers(self):
        """Fit one RobustScaler per ticker on training dates only."""
        train_set = set(self.train_dates)
        self.scalers = {}

        for t in self.tickers:
            df        = self.datasets[t].copy()
            df["date"] = self._clean_dates(df)
            train_df  = df[df["date"].isin(train_set)]

            if len(train_df) == 0:
                raise ValueError(f"No training rows found for {t}")

            scaler = RobustScaler()
            scaler.fit(train_df[FEATURE_COLS].values)
            self.scalers[t] = scaler

    def _get_feature_row(self, ticker, date):
        """Normalised feature vector for one ticker on one date."""
        df         = self.datasets[ticker].copy()
        df["date"] = self._clean_dates(df)
        row        = df[df["date"] == pd.Timestamp(date).normalize()]

        if len(row) == 0:
            return np.zeros(self.n_feat, dtype=np.float32)

        raw    = row[FEATURE_COLS].values          # (1, n_feat)
        scaled = self.scalers[ticker].transform(raw).clip(-5, 5)
        return scaled.squeeze().astype(np.float32)

    def _get_close(self, ticker, date):
        """Closing price for one ticker on one date."""
        df         = self.datasets[ticker].copy()
        df["date"] = self._clean_dates(df)
        row        = df[df["date"] == pd.Timestamp(date).normalize()]
        return float(row["close"].values[0]) if len(row) > 0 else np.nan

    def _get_state(self):
        date     = self.current_dates[self.bar]
        features = np.concatenate([
            self._get_feature_row(t, date) for t in self.tickers
        ])
        return np.concatenate([features, self.weights])

    def _sample_window(self, use_val=False):
        pool = self.val_dates if use_val else self.train_dates

        if len(pool) < self.steps:
            raise ValueError(
                f"Not enough {'val' if use_val else 'train'} dates: "
                f"{len(pool)} < {self.steps}"
            )

        if self.random_start and not use_val:
            start_idx = random.randint(0, len(pool) - self.steps)
        else:
            start_idx = len(pool) - self.steps   # most recent window

        self.current_dates = pool[start_idx : start_idx + self.steps]

    def reset(self, seed=None, use_val=False):

        if seed is not None:
            random.seed(seed)

        self.bar                 = 0
        self.weights             = np.ones(self.n_assets) / self.n_assets  # always N
        self.portfolio_value     = self.initial_balance
        self.portfolio_value_new = self.initial_balance
        self.pnl_pct_history     = []
        self.portfolios          = pd.DataFrame()
        self.episode            += 1

        self._sample_window(use_val=use_val)
        self.state = self._get_state()

        # Sanity check — catch mismatches early
        assert len(self.weights) == self.n_assets, \
            f"weights length {len(self.weights)} != n_assets {self.n_assets}"

        return self.state, {}

    def step(self, action):
        # Enforce valid allocation
        action = np.clip(action, 0, 1)
        action = action / (action.sum() + 1e-8)

        self.bar += 1
        if self.bar >= len(self.current_dates):
            return self.state, 0.0, True, False, {}

        date_prev = self.current_dates[self.bar - 1]
        date_curr = self.current_dates[self.bar]

        # First step — just set weights, no return yet
        if self.bar == 1:
            self.weights = action
            self.state   = self._get_state()
            return self.state, 0.0, False, False, {}

        # Portfolio return
        asset_returns = np.array([
            self._get_close(t, date_curr) /
            (self._get_close(t, date_prev) + 1e-8)
            for t in self.tickers
        ])

        self.portfolio_value_new = (
            self.portfolio_value * np.dot(self.weights, asset_returns)
        )
        pl      = self.portfolio_value_new - self.portfolio_value
        pnl_pct = pl / (self.portfolio_value + 1e-8) * 100
        self.pnl_pct_history.append(pnl_pct)

        # Update weights
        self.weights = action

        # Sharpe reward
        window  = min(20, len(self.pnl_pct_history))
        arr     = np.array(self.pnl_pct_history[-window:])

        # Need at least 2 points to compute meaningful vol
        if len(arr) < 2 or arr.std() < 1e-4:
            reward = 0.0
        else:
            ret    = arr.mean() / 100 * 252       # annualised mean return
            vol    = arr.std()  / 100 * np.sqrt(252)  # annualised vol
            sharpe = ret / (vol + 1e-8)
            reward = float(np.clip(sharpe, -5.0, 5.0))  # hard clip



        self.portfolio_value = self.portfolio_value_new
        self.state = self._get_state()

        done = self.bar >= len(self.current_dates) - 1
        info = {
            "portfolio_value": self.portfolio_value_new,
            "date":            date_curr,
        }
        return self.state, reward, done, False, info

    def _add_results(self, pl, asset_returns, date):
        row = {
            "episode": self.episode, "date": date,
            "pv":      self.portfolio_value,
            "pv_new":  self.portfolio_value_new,
            "pnl":     pl,
            "pnl_pct": pl / (self.portfolio_value + 1e-8) * 100,
        }
        for i, t in enumerate(self.tickers):
            row[f"w_{t}"]   = self.weights[i]
            row[f"ret_{t}"] = float(asset_returns[i])
        self.portfolios = pd.concat(
            [self.portfolios, pd.DataFrame([row])], ignore_index=True
        )

    def seed(self, seed=None):
        if seed is not None:
            random.seed(seed)

    @property
    def total_return(self):
        return self.portfolio_value / self.initial_balance - 1.0

    @property
    def sharpe_ratio(self):
        if len(self.pnl_pct_history) < 2:
            return 0.0
        r = np.array(self.pnl_pct_history) / 100
        return float(r.mean() / (r.std() + 1e-8) * np.sqrt(252))

    @property
    def max_drawdown(self):
        if len(self.pnl_pct_history) < 2:
            return 0.0
        vals = np.cumprod(1 + np.array(self.pnl_pct_history) / 100)
        peak = np.maximum.accumulate(vals)
        return float(((vals - peak) / (peak + 1e-8)).min())       