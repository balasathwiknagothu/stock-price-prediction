import yfinance as yf
import time

# simple in-memory cache
CACHE = {}
CACHE_TTL = 15 * 60  # 15 minutes

def get_stock_data(symbol):
    now = time.time()

    # return cached data if valid
    if symbol in CACHE:
        data, timestamp = CACHE[symbol]
        if now - timestamp < CACHE_TTL:
            return data

    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d", interval="1d")

        if hist.empty:
            CACHE[symbol] = (None, now)
            return None

        latest = hist.iloc[-1]

        data = {
            "symbol": symbol,
            "price": float(latest["Close"]),
            "high": float(latest["High"]),
            "low": float(latest["Low"]),
            "volume": int(latest["Volume"]),
            "timestamp": str(latest.name)
        }

        CACHE[symbol] = (data, now)
        return data

    except Exception:
        CACHE[symbol] = (None, now)
        return None