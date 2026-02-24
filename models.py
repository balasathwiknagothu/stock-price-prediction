from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Prediction(db.Model):
    __tablename__ = "prediction"

    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(20), unique=True, nullable=False, index=True)
    current_price = db.Column(db.Float)
    pred_1d = db.Column(db.Float)
    pred_3d = db.Column(db.Float)
    pred_7d = db.Column(db.Float)


class JobRun(db.Model):
    __tablename__ = "job_run"

    id = db.Column(db.Integer, primary_key=True)
    start_time = db.Column(db.DateTime(timezone=True))
    end_time = db.Column(db.DateTime(timezone=True))
    status = db.Column(db.String(20))
    total_inserted = db.Column(db.Integer)
    error_message = db.Column(db.Text)
    duration_seconds = db.Column(db.Float)