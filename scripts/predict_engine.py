import joblib
import yfinance as yf
import pandas as pd
import numpy as np

# -----------------------------
# LOAD TRAINED MODELS
# -----------------------------
model_1d = joblib.load("model_1d.pkl")
model_3d = joblib.load("model_3d.pkl")
model_7d = joblib.load("model_7d.pkl")
scaler = joblib.load("scaler.pkl")


# -----------------------------
# FEATURE ENGINEERING
# (Must match train_model.py exactly)
# -----------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_features(df):

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
        "return_1d",
        "return_3d",
        "return_7d",
        "ma_ratio",
        "volatility",
        "rsi",
        "volume_change",
        "momentum"
    ]

    return df, features


# -----------------------------
# SINGLE STOCK PREDICTION (kept for API)
# -----------------------------
def predict_stock(symbol):

    if not symbol.endswith(".NS"):
        symbol += ".NS"

    try:
        df = yf.download(symbol, period="1y", interval="1d", progress=False)
    except Exception:
        return None

    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df, features = compute_features(df)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    if len(df) < 10:
        return None

    latest = df.iloc[-1]
    X = pd.DataFrame([latest[features]])

    if X.shape[1] != 8:
        return None

    X_scaled = scaler.transform(X.values)

    current_price = float(latest["Close"])

    r1 = model_1d.predict(X_scaled)[0]
    r3 = model_3d.predict(X_scaled)[0]
    r7 = model_7d.predict(X_scaled)[0]

    return {
        "symbol": symbol,
        "current_price": current_price,
        "predictions": {
            "1D": {
                "expected_return": float(r1),
                "predicted_price": float(current_price * (1 + r1))
            },
            "3D": {
                "expected_return": float(r3),
                "predicted_price": float(current_price * (1 + r3))
            },
            "7D": {
                "expected_return": float(r7),
                "predicted_price": float(current_price * (1 + r7))
            }
        }
    }


# -----------------------------
# BATCH PREDICTION (for universe job)
# -----------------------------
def predict_from_dataframe(symbol, df):

    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df, features = compute_features(df)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    if len(df) < 10:
        return None

    latest = df.iloc[-1]
    X = pd.DataFrame([latest[features]])

    if X.shape[1] != 8:
        return None

    X_scaled = scaler.transform(X.values)

    current_price = float(latest["Close"])

    r1 = model_1d.predict(X_scaled)[0]
    r3 = model_3d.predict(X_scaled)[0]
    r7 = model_7d.predict(X_scaled)[0]

    return {
        "symbol": symbol,
        "current_price": current_price,
        "predictions": {
            "1D": {
                "expected_return": float(r1),
                "predicted_price": float(current_price * (1 + r1))
            },
            "3D": {
                "expected_return": float(r3),
                "predicted_price": float(current_price * (1 + r3))
            },
            "7D": {
                "expected_return": float(r7),
                "predicted_price": float(current_price * (1 + r7))
            }
        }
    }