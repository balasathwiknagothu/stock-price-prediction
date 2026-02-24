import logging
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

    df = pd.read_csv("data/master/stocks_universe.csv").head(20)

    symbols = [
        s if s.endswith(".NS") else f"{s}.NS"
        for s in df["symbol"].tolist()
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

        logging.info("Downloading all symbols in batch...")

        batch_data = yf.download(
            symbols,
            period="1y",
            interval="1d",
            group_by="ticker",
            progress=False,
            threads=True
        )

        # FIRST PASS
        for symbol in symbols:
            try:
                df_symbol = batch_data.get(symbol)

                if df_symbol is None or df_symbol.empty:
                    failed_symbols.append(symbol)
                    continue

                result = predict_from_dataframe(symbol, df_symbol)

                if not result:
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
                logging.warning(f"Symbol failed (pass 1): {symbol} | {e}")
                failed_symbols.append(symbol)

        # RETRY PASS
        retry_failed = []

        if failed_symbols:
            logging.info(f"Retrying {len(failed_symbols)} failed symbols...")

        for symbol in failed_symbols:
            try:
                df_symbol = batch_data.get(symbol)

                if df_symbol is None or df_symbol.empty:
                    retry_failed.append(symbol)
                    continue

                result = predict_from_dataframe(symbol, df_symbol)

                if not result:
                    retry_failed.append(symbol)
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
                logging.error(f"Symbol permanently failed: {symbol} | {e}")
                retry_failed.append(symbol)

        db.session.commit()

        job.status = "SUCCESS"
        job.total_inserted = count
        job.failed_count = len(retry_failed)
        job.end_time = datetime.now(UTC)

        duration = (job.end_time - job.start_time).total_seconds()
        job.duration_seconds = duration

        db.session.commit()

        logging.info(
            f"Universe job completed. Updated: {count}, "
            f"Failed: {len(retry_failed)}, Duration: {duration}s"
        )

    except Exception as e:
        job.status = "FAILED"
        job.error_message = str(e)
        job.end_time = datetime.now(UTC)

        duration = (job.end_time - job.start_time).total_seconds()
        job.duration_seconds = duration

        db.session.commit()

        logging.error(f"Universe job failed completely: {e}")


if __name__ == "__main__":
    run_universe_job()