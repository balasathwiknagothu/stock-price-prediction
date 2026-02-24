from apscheduler.schedulers.background import BackgroundScheduler
from run_universe import run_universe_job

scheduler = BackgroundScheduler()

scheduler.add_job(
    func=run_universe_job,
    trigger="cron",
    hour=0,
    minute=0,
    max_instances=1,
    coalesce=True,
    misfire_grace_time=3600
)

scheduler.start()

print("Scheduler started. Universe job will run daily at midnight.")

import time
while True:
    time.sleep(60)