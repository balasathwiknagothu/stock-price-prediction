# Engineering Notes

These notes document the engineering decisions, development phases, and implementation details behind the Stock Price Prediction platform.

Unlike the README, which explains how to use the application, this document explains how the project was designed and why particular implementation choices were made.

---

# Project Goal

The original objective was not simply to predict stock prices using a machine learning model.

Instead, the goal evolved into building an end-to-end stock analytics platform capable of:

- Collecting historical market data
- Processing and validating datasets
- Training reusable machine learning models
- Serving predictions through REST APIs
- Displaying market insights through an interactive web interface

The emphasis throughout development was on building a maintainable software project rather than a standalone machine learning notebook.

---

# Development Philosophy

Several design principles guided development:

- Separate data collection from model inference.
- Keep machine learning independent of the web layer.
- Prefer modular components over large monolithic scripts.
- Make every stage reproducible.
- Avoid hardcoding stock-specific logic.
- Design APIs that can support additional frontend clients in the future.

These principles influenced the project structure and continue to make the codebase easier to maintain and extend.

---

# Phase 1 — Historical Data Collection

The first challenge was obtaining reliable historical market data.

Rather than manually downloading CSV files, Yahoo Finance was selected as the primary data source through the `yfinance` Python library.

Reasons for this choice:

- Free to use
- No API keys required
- Large coverage of NSE-listed companies
- Consistent historical OHLCV data
- Easy integration with Pandas

Downloaded datasets are stored locally so that repeated training does not require additional network requests.

---

# Phase 2 — Dataset Validation

Raw financial data frequently contains missing values, holidays, suspended trading sessions, and incomplete records.

Before training, every dataset passes through a validation stage that:

- removes invalid rows,
- handles missing values,
- ensures chronological ordering,
- verifies sufficient history exists for prediction.

This preprocessing step improves model reliability and reduces training inconsistencies.

---

# Phase 3 — Feature Engineering

Raw closing prices alone rarely provide enough information for effective forecasting.

Additional features are derived from historical prices to capture short-term trends and momentum.

Examples include:

- Daily Returns
- Moving Averages
- Rolling Windows
- Price Momentum

Feature engineering is centralized within a dedicated module so that the exact same transformations are applied during both training and prediction.

This avoids discrepancies between model development and production inference.

---

# Phase 4 — Model Training

Training is intentionally separated from the web application.

Instead of retraining every time the server starts, dedicated scripts generate machine learning models offline.

Advantages include:

- Faster application startup
- Lower memory usage
- Reproducible training
- Easier experimentation with different algorithms

Once training completes, serialized models are stored for future inference.

---

# Phase 5 — Prediction Engine

Prediction requests are handled by a dedicated inference layer.

For every request:

1. Historical data is loaded.
2. Features are regenerated.
3. The appropriate trained model is loaded.
4. Predictions are generated.
5. Results are returned through the API.

Separating prediction logic from Flask routes keeps request handlers lightweight and easier to maintain.

---

# Phase 6 — Backend API

Flask was selected as the backend framework because the project primarily exposes prediction and analytics endpoints rather than a large business application.

REST endpoints were organized around application features:

- stock search
- prediction
- market movers
- gainers
- dashboard

This organization keeps the API intuitive for frontend integration.

---

# Phase 7 — Frontend

The frontend focuses on presenting financial information rather than implementing business logic.

Responsibilities include:

- Stock search
- Dashboard rendering
- Displaying predictions
- Market movers
- Interactive charts

All prediction calculations remain on the server.

---

# Current Architecture

The project now consists of four independent layers:

Data Collection

↓

Machine Learning

↓

REST API

↓

Frontend

Each layer communicates through clearly defined interfaces, making future improvements easier to implement.

---

# Lessons Learned

Several important software engineering lessons emerged during development:

- Good project structure is as important as model accuracy.
- Machine learning code should remain independent of web frameworks.
- Reusable feature engineering significantly simplifies maintenance.
- APIs should expose business concepts rather than implementation details.
- Documentation written during development is far more valuable than documentation written afterward.

---

# Future Engineering Goals

Potential architectural improvements include:

- Incremental model retraining
- Background job queues
- Redis caching
- Docker deployment
- CI/CD pipelines
- Automated testing
- Model versioning
- Feature store implementation
- Experiment tracking
- Cloud-native deployment

These improvements would move the project closer to production-grade machine learning systems.