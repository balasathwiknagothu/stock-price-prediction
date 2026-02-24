import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

# -----------------------------
# LOAD FULL STOCK UNIVERSE
# -----------------------------
df_universe = pd.read_csv("data/master/stocks_universe.csv")
SYMBOLS = df_universe["symbol"].dropna().unique().tolist()

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

    df["y1"] = df["Close"].pct_change(1).shift(-1)
    df["y3"] = df["Close"].pct_change(3).shift(-3)
    df["y7"] = df["Close"].pct_change(7).shift(-7)

    df = df.dropna()

    features = [
        "return_1d","return_3d","return_7d",
        "ma_ratio","volatility","rsi",
        "volume_change","momentum"
    ]

    X = df[features].copy()
    y1 = df["y1"]
    y3 = df["y3"]
    y7 = df["y7"]

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.dropna()

    y1 = y1.loc[X.index]
    y3 = y3.loc[X.index]
    y7 = y7.loc[X.index]

    return X, y1, y3, y7

# -----------------------------
# COLLECT DATA
# -----------------------------
all_X, all_y1, all_y3, all_y7 = [], [], [], []
processed = 0

for symbol in SYMBOLS:
    try:
        print(f"Downloading: {symbol}")
        df = yf.download(symbol, period=PERIOD, interval=INTERVAL, progress=False)

        if df is None or df.empty or len(df) < 300:
            continue

        X, y1, y3, y7 = create_features(df)

        if len(X) < 200:
            continue

        all_X.append(X)
        all_y1.append(y1)
        all_y3.append(y3)
        all_y7.append(y7)

        processed += 1

    except Exception:
        continue

print("Processed symbols:", processed)

X = pd.concat(all_X)
y1 = pd.concat(all_y1)
y3 = pd.concat(all_y3)
y7 = pd.concat(all_y7)

print("Total training samples:", len(X))

# -----------------------------
# SCALE FEATURES
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# TRAIN MODELS
# -----------------------------
model_1d = XGBRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    n_jobs=-1
)

model_3d = XGBRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    n_jobs=-1
)

model_7d = XGBRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    n_jobs=-1
)

print("Training 1D model...")
model_1d.fit(X_scaled, y1)

print("Training 3D model...")
model_3d.fit(X_scaled, y3)

print("Training 7D model...")
model_7d.fit(X_scaled, y7)

# -----------------------------
# SAVE MODELS
# -----------------------------
joblib.dump(model_1d, "model_1d.pkl")
joblib.dump(model_3d, "model_3d.pkl")
joblib.dump(model_7d, "model_7d.pkl")
joblib.dump(scaler, "scaler.pkl")

print("All models saved successfully.")
