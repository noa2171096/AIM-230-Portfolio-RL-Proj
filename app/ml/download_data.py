#from google.colab import drive
import os, pickle
from env import build_features
from datetime import datetime


# Creates a unique folder for every run e.g. saved_models/run_20240315_143022
RUN_ID   = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = os.path.join("saved_datasets", f"run_{RUN_ID}")
os.makedirs(SAVE_DIR, exist_ok=True)

TICKERS    = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
START_DATE = "2020-01-01"   # shorter range for sentiment — Finnhub limit
END_DATE   = "2023-12-31"

datasets = {}
for ticker in TICKERS:
    try:
        datasets[ticker] = build_features(
            ticker, START_DATE, END_DATE, use_sentiment=True
        )
    except Exception as e:
        print(f"Failed {ticker}: {e}")

# Save 
with open(f"{SAVE_DIR}/datasets_with_sentiment.pkl", "wb") as f:
    pickle.dump(datasets, f)
print("Saved to Drive ✓")

# Load in future sessions
# with open(f"{SAVE_DIR}/datasets_with_sentiment.pkl", "rb") as f:
#     datasets = pickle.load(f)