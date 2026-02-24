from dotenv import load_dotenv
load_dotenv()

from config import Config
from flask import Flask, render_template, jsonify, request
import pandas as pd
import yfinance as yf
from scripts.predict_engine import predict_stock

from models import db, Prediction, JobRun

app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)

with app.app_context():
    db.create_all()


# -----------------------------
# ROUTES
# -----------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/stocks")
def stocks():
    df = pd.read_csv("data/master/stocks_universe.csv")
    return jsonify(
        df[["symbol", "name"]]
        .sort_values("name")
        .to_dict(orient="records")
    )


@app.route("/history")
def history():
    symbol = request.args.get("symbol")

    if not symbol:
        return jsonify([])

    symbol = symbol.upper()

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="max", interval="1d")

        if df.empty:
            return jsonify([])

        df = df.reset_index()

        formatted = []

        for i in range(len(df)):
            formatted.append({
                "time": int(pd.to_datetime(df.loc[i, "Date"]).timestamp()),
                "open": float(df.loc[i, "Open"]),
                "high": float(df.loc[i, "High"]),
                "low": float(df.loc[i, "Low"]),
                "close": float(df.loc[i, "Close"]),
                "volume": float(df.loc[i, "Volume"])
            })

        return jsonify(formatted)

    except Exception:
        return jsonify([])


@app.route("/predict")
def predict():
    symbol = request.args.get("symbol")

    result = predict_stock(symbol)

    if not result:
        return jsonify({"status": "failed"})

    return jsonify({
        "status": "done",
        "current_price": result["current_price"],
        "predictions": result["predictions"]
    })


def get_market_data(kind, limit=None):
    rows = Prediction.query.all()
    results = []

    for r in rows:
        if not r.pred_1d:
            continue

        change = (r.pred_1d - r.current_price) / r.current_price * 100
        direction = "UP" if change > 0 else "DOWN"

        results.append({
            "symbol": r.symbol,
            "current_price": r.current_price,
            "predicted_price": r.pred_1d,
            "pct_change": round(change, 2),
            "direction": direction
        })

    if kind == "gainers":
        results.sort(key=lambda x: x["pct_change"], reverse=True)
    elif kind == "losers":
        results.sort(key=lambda x: x["pct_change"])
    elif kind == "movers":
        results.sort(key=lambda x: abs(x["pct_change"]), reverse=True)

    if limit:
        results = results[:limit]

    return results


@app.route("/market/<kind>/top")
def market_top(kind):
    rows = get_market_data(kind, limit=5)
    return jsonify(rows)


#HEALTH
@app.route("/health")
def health():
    return {
        "status": "OK"
    }


@app.route("/last-job")
def last_job():
    job = JobRun.query.order_by(JobRun.id.desc()).first()

    if not job:
        return {"status": "No jobs run yet"}

    return {
        "id": job.id,
        "start_time": job.start_time,
        "end_time": job.end_time,
        "status": job.status,
        "total_inserted": job.total_inserted,
        "duration_seconds": job.duration_seconds
    }


@app.route("/run-universe")
def run_universe_job():
    from run_universe import run_universe
    run_universe(limit=20)
    return {"status": "Universe job completed (20 stocks)"}

if __name__ == "__main__":
    app.run(debug=True)
