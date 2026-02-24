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
    filename="universe_job.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_universe_job():

    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        CSV_PATH = os.path.join(BASE_DIR, "data", "master", "stocks_universe.csv")

        if not os.path.exists(CSV_PATH):
            raise Exception(f"CSV file not found at {CSV_PATH}")

        df = pd.read_csv(CSV_PATH).head(20)

        symbols = [
            s if s.endswith(".NS") else f"{s}.NS"
            for s in df["symbol"].tolist()
        ]

    except Exception as e:
        logging.error(f"CSV loading failed: {e}")
        raise

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

        logging.info("Starting sequential downloads...")

        for symbol in symbols:
            try:
                df_symbol = yf.download(
                    symbol,
                    period="1y",
                    interval="1d",
                    progress=False,
                    threads=False
                )

                if df_symbol is None or df_symbol.empty:
                    logging.warning(f"No data for {symbol}")
                    failed_symbols.append(symbol)
                    continue

                result = predict_from_dataframe(symbol, df_symbol)

                if not result:
                    logging.warning(f"Prediction failed for {symbol}")
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

            except Exception as e:
                logging.error(f"Symbol failed: {symbol} | {e}")
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

        logging.info(
            f"Universe job completed. Updated: {count}, "
            f"Failed: {len(failed_symbols)}"
        )

    except Exception as e:
        job.status = "FAILED"
        job.error_message = str(e)
        job.end_time = datetime.now(UTC)
        job.duration_seconds = (
            job.end_time - job.start_time
        ).total_seconds()

        db.session.commit()

        logging.error(f"Universe job failed completely: {e}")
        raise


if __name__ == "__main__":
    run_universe_job()