import logging
import os
from dotenv import load_dotenv
load_dotenv()

from models import db, Prediction, JobRun
from scripts.predict_engine import predict_from_dataframe
import pandas as pd
from datetime import datetime, UTC
import yfinance as yf


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_universe_job():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(BASE_DIR, "data", "master", "stocks_universe.csv")

    if not os.path.exists(CSV_PATH):
        raise Exception(f"CSV file not found at {CSV_PATH}")

    df = pd.read_csv(CSV_PATH).head(100)

    if "symbol" in df.columns:
        column_name = "symbol"
    elif "Symbol" in df.columns:
        column_name = "Symbol"
    elif "SYMBOL" in df.columns:
        column_name = "SYMBOL"
    else:
        raise Exception(f"No valid symbol column found. Found columns: {df.columns.tolist()}")

    symbols = [
        s if str(s).endswith(".NS") else f"{s}.NS"
        for s in df[column_name].tolist()
    ]

    job = JobRun(
        start_time=datetime.now(UTC),
        status="RUNNING"
    )
    db.session.add(job)
    db.session.commit()

    try:
        count = 0
        failed_symbols = []

        existing_rows = Prediction.query.all()
        existing_map = {row.symbol: row for row in existing_rows}

        for symbol in symbols:

            print("Processing symbol:", symbol)

            logging.info(f"PROCESSING: {symbol}")
            print(f"PROCESSING: {symbol}")

            try:
                df_symbol = yf.download(
                    symbol,
                    period="1y",
                    interval="1d",
                    progress=False,
                    threads=False,
                    timeout=10
                )
                print("Downloaded rows:", len(df_symbol))

                if df_symbol is None or df_symbol.empty:
                    print("NO DATA FROM YF:", symbol)
                    logging.warning(f"NO DATA: {symbol}")
                    failed_symbols.append(symbol)
                    continue

                result = predict_from_dataframe(symbol, df_symbol)

                if not result:
                    print("MODEL FAILED:", symbol)
                    logging.warning(f"PREDICTION FAILED: {symbol}")
                    failed_symbols.append(symbol)
                    continue

                existing = existing_map.get(symbol)

                if existing:
                    existing.current_price = result["current_price"]
                    existing.pred_1d = result["predictions"]["1D"]["predicted_price"]
                    existing.pred_3d = result["predictions"]["3D"]["predicted_price"]
                    existing.pred_7d = result["predictions"]["7D"]["predicted_price"]
                else:
                    new_prediction = Prediction(
                        symbol=symbol,
                        current_price=result["current_price"],
                        pred_1d=result["predictions"]["1D"]["predicted_price"],
                        pred_3d=result["predictions"]["3D"]["predicted_price"],
                        pred_7d=result["predictions"]["7D"]["predicted_price"],
                    )
                    db.session.add(new_prediction)

                count += 1
                logging.info(f"SUCCESS: {symbol}")

            except Exception as e:
                logging.error(f"ERROR ON {symbol}: {e}")
                failed_symbols.append(symbol)

        db.session.commit()

        job.status = "SUCCESS"
        job.total_inserted = count
        job.failed_count = len(failed_symbols)
        job.end_time = datetime.now(UTC)
        job.duration_seconds = (
            job.end_time - job.start_time
        ).total_seconds()

        db.session.commit()

        logging.info("UNIVERSE JOB COMPLETED")

    except Exception as e:

        job.status = "FAILED"
        job.error_message = str(e)
        job.end_time = datetime.now(UTC)
        job.duration_seconds = (
            job.end_time - job.start_time
        ).total_seconds()

        db.session.commit()

        logging.error(f"UNIVERSE JOB FAILED: {e}")


if __name__ == "__main__":
    run_universe_job()