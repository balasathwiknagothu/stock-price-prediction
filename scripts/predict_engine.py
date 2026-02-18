import yfinance as yf
import pandas as pd
import numpy as np
import pickle

# -----------------------------
# LOAD MODEL
# -----------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

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
    df = df.copy()

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

    df = df.dropna()

    features = [
        "return_1d","return_3d","return_7d",
        "ma_ratio","volatility","rsi",
        "volume_change","momentum"
    ]

    return df, df[features]


# -----------------------------
# MAIN PREDICTION FUNCTION
# -----------------------------
def predict_stock(symbol):

    if not symbol.endswith(".NS"):
        symbol = symbol + ".NS"

    df = yf.download(symbol, period="1y", interval="1d", progress=False)

    if df.empty or len(df) < 100:
        return None

    # 🔥 FIX: Flatten multi-index columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df, X = create_features(df)

    if len(X) == 0:
        return None

    latest_features = X.iloc[-1:].values

    predicted_vol_adj_return = float(model.predict(latest_features)[0])

    latest_volatility = float(df["volatility"].iloc[-1])
    current_price = float(df["Close"].iloc[-1])

    predicted_return = predicted_vol_adj_return * latest_volatility
    predicted_price = current_price * (1 + predicted_return)

    pct_change = predicted_return * 100
    direction = "UP" if pct_change >= 0 else "DOWN"

    return {
        "current_price": round(current_price, 2),
        "predicted_price": round(predicted_price, 2),
        "pct_change": round(pct_change, 2),
        "direction": direction
    }
