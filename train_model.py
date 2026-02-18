import yfinance as yf
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# -----------------------------
# CONFIG
# -----------------------------
TRAIN_SYMBOLS = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "SBIN.NS","LT.NS","ITC.NS","HINDUNILVR.NS","AXISBANK.NS",
    "KOTAKBANK.NS","BAJFINANCE.NS","ASIANPAINT.NS","MARUTI.NS",
    "TITAN.NS","SUNPHARMA.NS","ULTRACEMCO.NS","WIPRO.NS",
    "NTPC.NS","POWERGRID.NS","ONGC.NS","TECHM.NS",
]

PERIOD = "max"
INTERVAL = "1d"

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def create_features(df):
    df["return_1d"] = df["Close"].pct_change(1)
    df["return_3d"] = df["Close"].pct_change(3)
    df["return_7d"] = df["Close"].pct_change(7)

    df["ma20"] = df["Close"].rolling(20).mean()
    df["ma50"] = df["Close"].rolling(50).mean()
    df["ma_ratio"] = df["ma20"] / (df["ma50"] + 1e-9)

    df["volatility"] = df["Close"].pct_change().rolling(20).std()
    df["rsi"] = compute_rsi(df["Close"])
    df["volume_change"] = df["Volume"].pct_change()
    df["momentum"] = df["ma20"] - df["ma50"]

    # Future 7-day return
    df["future_return"] = df["Close"].pct_change(7).shift(-7)

    # Volatility-adjusted target
    df["target"] = df["future_return"] / (df["volatility"] + 1e-6)

    df = df.dropna()

    features = [
        "return_1d","return_3d","return_7d",
        "ma_ratio","volatility","rsi",
        "volume_change","momentum"
    ]

    X = df[features].copy()
    y = df["target"].copy()

    # -----------------------------
    # CLEANING STEP (CRITICAL FIX)
    # -----------------------------
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    y.replace([np.inf, -np.inf], np.nan, inplace=True)

    X = X.dropna()
    y = y.loc[X.index]

    # Clip extreme outliers
    X = X.clip(lower=-10, upper=10)
    y = y.clip(lower=-10, upper=10)

    return X, y

# -----------------------------
# DATA COLLECTION
# -----------------------------
all_X = []
all_y = []

for symbol in TRAIN_SYMBOLS:
    print(f"Downloading {symbol}...")
    df = yf.download(symbol, period=PERIOD, interval=INTERVAL, progress=False)

    if len(df) < 200:
        continue

    X, y = create_features(df)

    if len(X) > 100:
        all_X.append(X)
        all_y.append(y)

X = pd.concat(all_X)
y = pd.concat(all_y)

print("Total training samples:", len(X))

# -----------------------------
# TRAIN MODEL
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3
)

model.fit(X_train, y_train)

print("Training complete.")

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")
