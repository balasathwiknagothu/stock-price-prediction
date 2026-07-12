# Decision Log

This document records the major architectural and engineering decisions made during the development of the Stock Price Prediction platform.

Rather than documenting only *what* was implemented, it explains *why* each decision was made, the alternatives considered, and the trade-offs that were accepted.

---

# Decision 1 — Flask Instead of Django

## Decision

The backend was implemented using **Flask** rather than Django.

## Alternatives Considered

- Django
- FastAPI

## Reasoning

The project primarily exposes REST endpoints for stock search, prediction, and market analytics while serving a lightweight web interface.

Flask provides a minimal framework that allows the machine learning pipeline to integrate naturally without unnecessary abstractions.

Django's built-in ORM, authentication system, and administration interface were not required for the scope of this project.

FastAPI was considered because of its excellent asynchronous support and automatic OpenAPI documentation. However, Flask was selected due to its simplicity, extensive ecosystem, and familiarity during development.

## Trade-offs

Pros

- Lightweight
- Easy ML integration
- Flexible project structure
- Lower complexity

Cons

- Less built-in functionality
- Manual API documentation
- More application structure must be designed explicitly

---

# Decision 2 — Yahoo Finance as the Data Source

## Decision

Historical market data is collected using the **yfinance** library.

## Alternatives Considered

- Alpha Vantage
- Polygon.io
- Twelve Data
- NSE APIs

## Reasoning

Yahoo Finance provides free historical OHLCV data covering a large number of Indian stocks without requiring API keys.

This significantly simplifies both development and reproducibility.

Commercial APIs generally offer higher reliability and lower latency but introduce authentication requirements, rate limits, or subscription costs that were unnecessary for this project's objectives.

## Trade-offs

Pros

- Free
- Large historical dataset
- Easy integration
- No authentication

Cons

- Community-maintained
- Subject to upstream changes
- Not intended for production trading systems

---

# Decision 3 — Local Historical CSV Storage

## Decision

Downloaded market data is stored locally as CSV files.

## Alternatives Considered

- PostgreSQL
- MongoDB
- Direct API requests on every prediction

## Reasoning

Historical data changes only once per trading day.

Persisting datasets locally reduces unnecessary API requests, improves prediction latency, and allows offline experimentation during model development.

Separating data collection from prediction also improves reproducibility.

## Trade-offs

Pros

- Fast access
- Offline availability
- Simple storage
- Easy debugging

Cons

- Additional storage space
- Periodic refresh required

---

# Decision 4 — Offline Model Training

## Decision

Machine learning models are trained independently from the web application.

## Alternatives Considered

Training models during server startup.

## Reasoning

Retraining models every time the application starts would dramatically increase startup time and consume unnecessary computational resources.

Instead, models are trained once and serialized using Pickle.

The prediction service loads these serialized models directly during runtime.

## Trade-offs

Pros

- Faster startup
- Lower memory usage
- Stable inference
- Repeatable training

Cons

- Models require manual retraining when datasets change

---

# Decision 5 — Multiple Prediction Models

## Decision

Separate models are maintained for different prediction horizons.

Examples include:

- 1 Day
- 3 Day
- 7 Day

## Alternatives Considered

A single generalized forecasting model.

## Reasoning

Different forecasting horizons exhibit different statistical behavior.

Maintaining dedicated models allows each model to specialize for its respective prediction window.

This approach also simplifies future experimentation with different algorithms.

## Trade-offs

Pros

- Independent optimization
- Easier experimentation
- Modular training

Cons

- Additional storage
- More training time

---

# Decision 6 — Feature Engineering Module

## Decision

Feature generation is isolated in a dedicated module.

## Alternatives Considered

Generating features directly inside training scripts.

## Reasoning

Training and prediction must always use identical preprocessing steps.

Keeping feature generation in a single reusable module prevents inconsistencies and reduces duplicated logic.

## Trade-offs

Pros

- Code reuse
- Easier maintenance
- Consistent inference

Cons

- Slightly larger abstraction layer

---

# Decision 7 — REST APIs

## Decision

Prediction functionality is exposed through RESTful endpoints.

## Alternatives Considered

Direct function calls from the frontend.

## Reasoning

REST APIs separate presentation from business logic.

Future clients—including mobile applications or third-party integrations—can consume the same prediction services without modifying the machine learning layer.

## Trade-offs

Pros

- Scalable architecture
- Frontend independent
- Easier testing

Cons

- Slight serialization overhead

---

# Decision 8 — SQLAlchemy for Data Access

## Decision

Database interactions use SQLAlchemy.

## Alternatives Considered

Raw SQL queries.

## Reasoning

SQLAlchemy provides an ORM abstraction that improves maintainability while remaining flexible enough for future database migrations.

Using an ORM also reduces repetitive SQL boilerplate.

## Trade-offs

Pros

- Cleaner code
- Easier maintenance
- Database abstraction

Cons

- Slight ORM overhead

---

# Decision 9 — Modular Project Structure

## Decision

The application is organized into separate directories for data, machine learning, frontend assets, scripts, and configuration.

## Alternatives Considered

Keeping all logic in a small number of scripts.

## Reasoning

Separating responsibilities reduces coupling and improves readability.

The resulting structure allows each component to evolve independently without affecting unrelated modules.

## Trade-offs

Pros

- Better maintainability
- Easier collaboration
- Cleaner architecture

Cons

- More files to manage

---

# Future Decisions

Several architectural improvements are planned for future iterations:

- Docker containerization
- Model versioning
- Redis caching
- CI/CD pipelines
- Automated retraining
- Experiment tracking
- Cloud-native deployment
- Distributed prediction services

These enhancements were intentionally deferred to keep the current project focused on delivering a complete, maintainable machine learning application without introducing unnecessary complexity.