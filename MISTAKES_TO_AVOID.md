# Mistakes to Avoid

This document records implementation mistakes encountered during development, explains why they occurred, and documents the solutions adopted.

The goal is not only to avoid repeating the same mistakes but also to provide context for future contributors who extend or maintain the project.

---

# 1. Mixing Data Collection with Prediction

## Problem

Initially, the prediction logic downloaded fresh stock data every time a prediction request was made.

```
Prediction Request
        ↓
Download Dataset
        ↓
Generate Prediction
```

Although this approach worked during early development, it introduced unnecessary latency and made predictions dependent on network availability.

---

## Why It Was a Problem

- Increased API response time.
- Repeated network requests.
- Higher chance of failures due to internet connectivity.
- Difficult to reproduce predictions during testing.

---

## Solution

Historical datasets are downloaded independently and stored locally.

Prediction requests now operate only on validated local datasets.

```
Prediction Request

↓

Historical Dataset

↓

Prediction
```

This significantly improves performance and ensures consistent inference.

---

# 2. Training During Application Startup

## Problem

The first implementation trained machine learning models every time the Flask server started.

```
Start Flask

↓

Train Model

↓

Launch Server
```

For small datasets this was acceptable, but as the number of supported stocks increased, startup time became excessively long.

---

## Why It Was a Problem

- Slow server startup.
- Unnecessary CPU usage.
- Models were retrained even when no new data existed.

---

## Solution

Training was moved into dedicated scripts.

The application now loads pre-trained models stored as serialized `.pkl` files.

---

# 3. Duplicate Feature Engineering Logic

## Problem

Feature generation was initially implemented separately inside the training and prediction scripts.

Although both implementations appeared similar, small changes eventually caused the generated feature sets to diverge.

---

## Consequences

The model was trained using one feature representation but performed inference using another.

This inconsistency reduced prediction reliability.

---

## Solution

Feature engineering was centralized into a single reusable module.

Both training and prediction now invoke exactly the same feature generation logic.

---

# 4. Missing Historical Data Validation

## Problem

Some downloaded datasets contained incomplete trading sessions, missing values, or insufficient historical records.

Attempting to train models directly on these datasets produced unstable results.

---

## Solution

A validation stage was introduced before model training.

The validation process now verifies:

- Missing values
- Dataset length
- Chronological ordering
- Required columns
- Invalid records

Only validated datasets are used for training.

---

# 5. Tight Coupling Between Backend and Machine Learning

## Problem

Early versions placed prediction logic directly inside Flask route handlers.

Example:

```
@app.route("/predict")
```

contained

- preprocessing,
- feature generation,
- model loading,
- prediction,
- response formatting.

This made the route difficult to maintain and nearly impossible to test independently.

---

## Solution

Prediction logic was extracted into dedicated modules.

The Flask application now serves only as an HTTP interface while the machine learning layer handles inference.

---

# 6. Hardcoded Configuration Values

## Problem

Several configuration values—including file paths and database settings—were initially hardcoded.

Changing environments required modifying source code.

---

## Solution

Configuration values were externalized.

The application now supports environment-specific configuration, making deployment significantly easier.

---

# 7. Large Monolithic Scripts

## Problem

During the first stages of development, many responsibilities were implemented inside a small number of Python files.

Examples included:

- downloading data,
- preprocessing,
- training,
- prediction,
- API logic.

As the project grew, these scripts became difficult to understand.

---

## Solution

The project was reorganized into dedicated modules with clear responsibilities.

Each component now focuses on a single concern.

---

# 8. Insufficient Error Handling

## Problem

Early API implementations assumed valid input.

Requests for unsupported symbols or missing datasets often resulted in generic server errors.

---

## Solution

Validation was added before processing requests.

Meaningful JSON error responses are now returned for invalid symbols, unavailable datasets, and prediction failures.

---

# 9. Ignoring Scalability Early

## Observation

The project originally targeted only a handful of stocks.

As additional datasets and models were introduced, assumptions made during the initial implementation no longer held.

Examples included:

- repeated model loading,
- duplicated preprocessing,
- inefficient dataset scanning.

---

## Lesson Learned

Designing for moderate scalability from the beginning reduces future refactoring effort.

Although the current application targets educational and portfolio purposes, its modular structure makes future expansion significantly easier.

---

# Key Lessons

The following engineering principles emerged during development:

- Separate data collection from prediction.
- Never duplicate feature engineering logic.
- Train models offline whenever possible.
- Validate datasets before training.
- Keep REST APIs lightweight.
- Separate business logic from HTTP handling.
- Modularize early.
- Handle errors explicitly.
- Document architectural decisions as they are made.

Following these principles improved both the maintainability and reliability of the application and will continue to guide future development.