from flask import Flask, render_template, jsonify, request
from flask_sqlalchemy import SQLAlchemy
import os
from datetime import datetime, timezone
from sqlalchemy import func
from apscheduler.schedulers.background import BackgroundScheduler
from scripts.predict_engine import predict_stock
import yfinance as yf
import pandas as pd

# -----------------------------
# APP INITIALIZATION
# -----------------------------
app = Flask(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL")

if DATABASE_URL:
    app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
else:
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///stocks.db"

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# -----------------------------
# DATABASE MODELS
# -----------------------------
class Prediction(db.Model):
    __tablename__ = "prediction"

    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(20), index=True)
    current_price = db.Column(db.Float)
    predicted_price = db.Column(db.Float)
    pct_change = db.Column(db.Float)
    direction = db.Column(db.String(10))
    prediction_date = db.Column(db.String(20), index=True)


class PredictionJob(db.Model):
    __tablename__ = "prediction_job"

    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(20), index=True)
    status = db.Column(db.String(20), default="pending")
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime,
                           default=lambda: datetime.now(timezone.utc),
                           onupdate=lambda: datetime.now(timezone.utc))


with app.app_context():
    db.create_all()

# -----------------------------
# BACKGROUND JOB PROCESSOR
# -----------------------------
def process_pending_jobs():
    with app.app_context():

        job = PredictionJob.query.filter_by(status="pending").first()
        if not job:
            return

        try:
            job.status = "running"
            db.session.commit()

            symbol = job.symbol
            today = datetime.now(timezone.utc).date().isoformat()

            # If already predicted today → mark done
            existing = Prediction.query.filter_by(
                symbol=symbol,
                prediction_date=today
            ).first()

            if existing:
                job.status = "done"
                db.session.commit()
                return

            result = predict_stock(symbol)

            if not result:
                job.status = "failed"
                db.session.commit()
                return

            new_prediction = Prediction(
                symbol=symbol,
                current_price=result["current_price"],
                predicted_price=result["predicted_price"],
                pct_change=result["pct_change"],
                direction=result["direction"],
                prediction_date=today
            )

            db.session.add(new_prediction)
            job.status = "done"
            db.session.commit()

            print(f"[✔] Processed job for {symbol}")

        except Exception as e:
            print("[ERROR]", e)
            job.status = "failed"
            db.session.commit()

# -----------------------------
# START SCHEDULER
# -----------------------------
scheduler = BackgroundScheduler()
scheduler.add_job(func=process_pending_jobs, trigger="interval", seconds=20)
scheduler.start()

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# STOCK LIST ROUTE
# -----------------------------
STOCKS_FILE = "data/master/stocks_universe.csv"

@app.route("/stocks")
def stocks():
    if not os.path.exists(STOCKS_FILE):
        return jsonify([])

    df = pd.read_csv(STOCKS_FILE)

    return jsonify(
        df[["symbol", "name"]]
        .sort_values("name")
        .to_dict(orient="records")
    )


# -----------------------------
# HISTORY ROUTE (FIXED)
# -----------------------------
@app.route("/history")
def history():
    symbol = request.args.get("symbol")

    if not symbol:
        return jsonify([])

    # 🔥 Force NSE format
    if not symbol.endswith(".NS"):
        symbol = symbol + ".NS"

    print("Fetching history for:", symbol)

    try:
        df = yf.download(
            symbol,
            period="max",
            interval="1d",
            progress=False,
            auto_adjust=False,
            threads=False
        )

        if df.empty:
            print("No data returned")
            return jsonify([])

        # Flatten MultiIndex if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        formatted = []

        for _, row in df.iterrows():
            formatted.append({
                "time": int(pd.to_datetime(row["Date"]).timestamp()),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"])
            })

        print("Rows returned:", len(formatted))
        return jsonify(formatted)

    except Exception as e:
        print("History error:", e)
        return jsonify([])


# -----------------------------
# PREDICT ROUTE
# -----------------------------
@app.route("/predict")
def predict():
    symbol = request.args.get("symbol")

    if not symbol:
        return jsonify({"error": "Missing symbol"})

    today = datetime.now(timezone.utc).date().isoformat()

    existing = Prediction.query.filter_by(
        symbol=symbol,
        prediction_date=today
    ).first()

    if existing:
        return jsonify({
            "status": "done",
            "stock": symbol,
            "current_price": existing.current_price,
            "predicted_price": existing.predicted_price,
            "pct_change": existing.pct_change,
            "direction": existing.direction
        })

    existing_job = PredictionJob.query.filter_by(
        symbol=symbol,
        status="pending"
    ).first()

    if not existing_job:
        new_job = PredictionJob(symbol=symbol, status="pending")
        db.session.add(new_job)
        db.session.commit()

    return jsonify({
        "status": "processing",
        "stock": symbol
    })


# -----------------------------
# PREDICTION STATUS
# -----------------------------
@app.route("/prediction-status")
def prediction_status():
    symbol = request.args.get("symbol")
    today = datetime.now(timezone.utc).date().isoformat()

    prediction = Prediction.query.filter_by(
        symbol=symbol,
        prediction_date=today
    ).first()

    if prediction:
        return jsonify({
            "status": "done",
            "current_price": prediction.current_price,
            "predicted_price": prediction.predicted_price,
            "pct_change": prediction.pct_change,
            "direction": prediction.direction
        })

    return jsonify({"status": "processing"})


# -----------------------------
# MARKET ROUTES
# -----------------------------
def get_market_data(kind, limit=None):
    query = db.session.query(
        Prediction.symbol,
        Prediction.current_price,
        Prediction.predicted_price,
        Prediction.pct_change,
        Prediction.direction
    )

    if kind == "gainers":
        query = query.filter_by(direction="UP").order_by(Prediction.pct_change.desc())

    elif kind == "losers":
        query = query.filter_by(direction="DOWN").order_by(Prediction.pct_change.asc())

    elif kind == "movers":
        query = query.order_by(func.abs(Prediction.pct_change).desc())

    else:
        return []

    if limit:
        query = query.limit(limit)

    return query.all()


@app.route("/market/<kind>")
def market(kind):
    rows = get_market_data(kind)

    return jsonify([
        {
            "symbol": r.symbol,
            "current_price": r.current_price,
            "predicted_price": r.predicted_price,
            "pct_change": r.pct_change,
            "direction": r.direction
        }
        for r in rows
    ])


@app.route("/market/<kind>/top")
def market_top(kind):
    rows = get_market_data(kind, limit=5)

    return jsonify([
        {
            "symbol": r.symbol,
            "current_price": r.current_price,
            "predicted_price": r.predicted_price,
            "pct_change": r.pct_change,
            "direction": r.direction
        }
        for r in rows
    ])


# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
