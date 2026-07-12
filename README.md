# 📈 Stock Price Prediction using Machine Learning

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-336791?logo=postgresql)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-ORM-red)
![License](https://img.shields.io/badge/License-MIT-green)

</p>

A production-inspired **Stock Market Analytics and Machine Learning platform** built with **Python, Flask, Scikit-learn, Pandas, SQLAlchemy, and PostgreSQL**. The application combines historical market data collection, feature engineering, machine learning prediction, and interactive visualization into a single end-to-end system.

Unlike a standalone Jupyter Notebook or a basic regression model, this project focuses on building a complete software application around machine learning. Historical stock data is collected automatically, transformed into meaningful technical indicators, used to train reusable prediction models, and served through REST APIs that power an interactive dashboard.

The objective was to treat the prediction model as one component of a larger software system rather than the entire application.

---

# Table of Contents

- Highlights
- Tech Stack
- System Architecture
- Machine Learning Pipeline
- Data Pipeline
- Project Structure
- REST API Reference
- Prediction Workflow
- Security & Reliability
- Getting Started
- Testing
- Project Documentation
- Known Limitations
- Future Work
- License

---

# Highlights

Rather than focusing solely on prediction accuracy, this project emphasizes software engineering practices, modular architecture, and maintainability.

## Machine Learning

- Historical stock price prediction using Scikit-learn regression models
- Independent prediction models for multiple forecasting horizons
- Centralized feature engineering pipeline shared between training and inference
- Serialized models for fast runtime prediction
- Modular training scripts allowing model retraining without modifying the backend

## Market Analytics

- Intelligent stock search
- Historical market analysis
- Market movers
- Top gainers
- Interactive stock dashboard
- Prediction visualization

## Backend

- Flask REST APIs
- Modular service architecture
- SQLAlchemy ORM integration
- JSON-based API responses
- Scheduled data updates
- Configuration-driven application

## Data Processing

- Automated historical data collection
- Dataset validation
- Missing value handling
- Technical indicator generation
- Prediction dataset preparation

## Engineering

- Separation of concerns between data collection, ML, backend, and frontend
- Reusable feature engineering
- Independent model training pipeline
- Configurable project structure
- Deployment-ready architecture

---

# Tech Stack

| Layer | Technology |
|---------|------------|
| Language | Python 3 |
| Backend | Flask |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Database | PostgreSQL |
| ORM | SQLAlchemy |
| Scheduler | APScheduler |
| Data Source | Yahoo Finance (yfinance) |
| Frontend | HTML, CSS, JavaScript |
| Charts | TradingView Lightweight Charts |
| Deployment | Render |
| Version Control | Git & GitHub |

---

# System Architecture

The project follows a layered architecture where each component is responsible for a single concern.

```
                     User

                      │

                      ▼

             Flask Web Application

        ┌────────────┼────────────┐

        │            │            │

    Dashboard     REST APIs    Search

        │            │

        └────────────┼────────────┘

                     ▼

             Business Logic Layer

        ┌────────────┼────────────┐

        │            │            │

 Prediction     Market Data    Analytics

        │            │

        └────────────┼────────────┘

                     ▼

          Machine Learning Models

                     │

                     ▼

          Historical Stock Dataset

                     │

                     ▼

             Yahoo Finance API
```

The architecture intentionally separates data acquisition, prediction, and presentation. This makes the application easier to maintain, test, and extend as additional prediction models or analytics features are introduced.

---

# Machine Learning Pipeline

Unlike notebook-based workflows where data preparation and prediction are tightly coupled, the application separates training and inference into reusable stages.

```
Historical Data

        │

        ▼

Dataset Validation

        │

        ▼

Feature Engineering

        │

        ▼

Training Dataset

        │

        ▼

Scikit-learn Regression Model

        │

        ▼

Serialized Model (.pkl)

        │

        ▼

Prediction Service

        │

        ▼

REST API Response
```

This approach guarantees that both training and prediction use the same preprocessing pipeline, eliminating inconsistencies that commonly occur when feature generation is duplicated.

---

# Data Pipeline

Historical market data is downloaded using Yahoo Finance and stored locally before entering the machine learning workflow.

```
Yahoo Finance

        │

        ▼

Historical CSV Generation

        │

        ▼

Dataset Validation

        │

        ▼

Feature Engineering

        │

        ▼

Model Training

        │

        ▼

Saved Models

        │

        ▼

Prediction Engine

        │

        ▼

Dashboard & REST APIs
```

Separating data collection from prediction improves reproducibility and significantly reduces prediction latency by avoiding repeated network requests during inference.

---

---

# Project Structure

The project is organized into independent modules that separate machine learning, backend services, data processing, and frontend components. This modular structure improves maintainability, encourages code reuse, and allows individual components to evolve independently.

```
stock-price-prediction/
│
├── app.py                    # Flask application entry point
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── README.md
├── ARCHITECTURE.md
├── PROJECT_PLAN.md
│
├── data/
│   ├── historical/           # Historical stock datasets
│   ├── master/               # Stock metadata
│   └── predictions/          # Generated predictions
│
├── ml/
│   ├── features.py           # Feature engineering
│   ├── train_model.py        # Model training
│   ├── predict.py            # Prediction engine
│   └── models/               # Serialized ML models
│
├── scripts/                  # Utility scripts
├── static/                   # CSS, JavaScript, images
├── templates/                # HTML templates
├── instance/                 # Local configuration/database
└── models/                   # SQLAlchemy models
```

Every directory serves a single responsibility, reducing coupling between machine learning, data collection, backend logic, and presentation.

---

# Prediction Workflow

A prediction request passes through multiple stages before a response is returned to the client.

```
Client Request

        │

        ▼

Flask Route

        │

        ▼

Input Validation

        │

        ▼

Historical Dataset

        │

        ▼

Feature Engineering

        │

        ▼

Load Trained Model

        │

        ▼

Generate Prediction

        │

        ▼

JSON Response

        │

        ▼

Frontend Dashboard
```

This workflow ensures that predictions are generated consistently using the same preprocessing pipeline that was applied during model training.

---

# REST API Overview

The application exposes RESTful endpoints for stock search, prediction, and market analytics.

## Available Endpoints

| Method | Endpoint | Description |
|----------|-----------------------|---------------------------------------------|
| GET | `/` | Load the dashboard |
| GET | `/stocks` | Retrieve supported stocks |
| GET | `/predict` | Predict future stock prices |
| GET | `/market/movers` | View market movers |
| GET | `/market/gainers` | View top gainers |

All endpoints return lightweight JSON responses that can easily be consumed by web or mobile clients.

---

## Example Request

```
GET /predict?symbol=RELIANCE.NS
```

Example Response

```json
{
    "symbol": "RELIANCE.NS",
    "prediction": {
        "1_day": 1542.36,
        "3_day": 1554.80,
        "7_day": 1572.14
    }
}
```

---

# Feature Engineering

Feature engineering is one of the most important stages of the prediction pipeline.

Rather than relying only on historical closing prices, the application generates additional indicators that help machine learning models better capture market trends.

Current features include:

- Daily Returns
- Moving Average (5)
- Moving Average (10)
- Moving Average (20)
- Percentage Change
- Rolling Statistics

Feature generation is centralized within a dedicated module to guarantee identical preprocessing during both training and prediction.

---

# Model Training

Training is intentionally performed outside the Flask application.

The workflow consists of:

1. Download historical market data.
2. Validate and clean datasets.
3. Generate technical indicators.
4. Split training and testing data.
5. Train regression models.
6. Evaluate model performance.
7. Serialize trained models.

The web application never retrains models during startup. Instead, it loads pre-trained models from disk, resulting in significantly faster application startup and lower computational overhead.

---

# Model Inference

When a prediction request is received:

1. The requested stock symbol is validated.
2. Historical data is loaded.
3. Features are regenerated.
4. The appropriate serialized model is loaded.
5. Predictions are generated.
6. Results are returned as JSON.

Separating inference from training allows new models to be deployed without modifying the API layer.

---

# Smart Stock Search

The dashboard includes a responsive stock search interface designed for fast navigation across supported companies.

Features include:

- Search by company name
- Search by stock symbol
- Instant filtering
- Lightweight client-side search
- Responsive suggestions

This minimizes unnecessary server requests while providing a smooth user experience.

---

# Market Analytics

Beyond price prediction, the platform includes several analytics modules to help users explore market activity.

Current analytics include:

- Market Movers
- Top Gainers
- Historical Stock Performance
- Interactive Charts
- Daily Price Trends

These features transform the application from a simple prediction tool into a broader market analytics platform.

---

# Frontend

The frontend is intentionally lightweight.

Responsibilities include:

- Rendering prediction results
- Displaying charts
- Stock search
- Market overview
- Dashboard updates

Business logic and prediction calculations remain entirely on the backend.

---

# Database

SQLAlchemy is used as the ORM layer to simplify database interactions.

Responsibilities include:

- Managing stock metadata
- Storing prediction information
- Application configuration
- Future extensibility for user-specific features

Using an ORM keeps the application database-agnostic while reducing repetitive SQL code.

---

# Security & Reliability

Although this is a machine learning project rather than an authentication-focused application, several engineering practices improve reliability.

- Input validation for API requests.
- Structured project organization.
- Reusable feature engineering pipeline.
- Offline model training.
- Modular architecture.
- Consistent preprocessing during inference.
- Configuration-based application settings.

These practices improve maintainability and reduce the likelihood of prediction inconsistencies.

---

---

# Getting Started

## Prerequisites

Before running the application, ensure the following software is installed.

| Software | Version |
|-----------|----------|
| Python | 3.11 or later |
| Git | Latest |
| PostgreSQL | 15+ (Optional depending on configuration) |
| pip | Latest |

A virtual environment is strongly recommended to isolate project dependencies.

---

# Installation

## Clone the Repository

```bash
git clone https://github.com/balasathwiknagothu/stock-price-prediction.git

cd stock-price-prediction
```

---

## Create a Virtual Environment

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Environment

Depending on your setup, configure database credentials and application settings.

Typical configuration includes:

- Database URL
- Flask environment
- Secret keys
- Scheduler settings

Environment variables are preferred over hardcoded configuration for easier deployment across development and production environments.

---

## Download Historical Data

Run the data collection scripts to download historical market data.

```bash
python scripts/download_data.py
```

After downloading, the datasets are stored inside

```
data/historical/
```

These datasets form the foundation of both model training and prediction.

---

## Train Machine Learning Models

Generate prediction models using the training pipeline.

```bash
python ml/train_model.py
```

Training performs the following steps:

- Load historical datasets
- Clean invalid records
- Generate features
- Train regression models
- Evaluate performance
- Save serialized models

Generated models are stored as

```
ml/models/
```

---

## Start the Flask Server

```bash
python app.py
```

or

```bash
flask run
```

The application starts on

```
http://localhost:5000
```

---

# Using the Application

Once the server is running, users can

- Search supported stocks
- View market movers
- Explore top gainers
- Generate predictions
- Analyze historical price movements
- View interactive charts

The application is designed to provide an intuitive workflow without requiring technical knowledge of machine learning.

---

# Testing

The project has been manually tested across the complete prediction workflow.

Verification includes:

- Historical dataset loading
- Data preprocessing
- Feature engineering
- Model inference
- REST API responses
- Dashboard rendering
- Market movers
- Top gainers
- Search functionality

Every prediction request was validated using real historical datasets to ensure consistent preprocessing and inference.

Future improvements include automated unit tests and integration testing for the complete prediction pipeline.

---

# Performance Considerations

Several implementation decisions were made to improve responsiveness.

## Offline Training

Models are trained independently from the web application.

This reduces application startup time and avoids unnecessary retraining.

---

## Serialized Models

Trained models are stored as Pickle files.

Loading serialized models is significantly faster than rebuilding them on every startup.

---

## Local Historical Datasets

Historical market data is stored locally after download.

Advantages include:

- Faster predictions
- Reduced network requests
- Offline experimentation
- Reproducible results

---

## Modular Feature Engineering

Training and prediction share the same preprocessing pipeline.

This guarantees identical feature generation during both phases.

---

# Deployment

The application is designed to be deployed on cloud platforms such as

- Render
- Railway
- PythonAnywhere
- Azure App Service
- AWS EC2
- Google Cloud Run

Production deployment typically consists of:

```
GitHub Repository

↓

Cloud Platform

↓

Install Dependencies

↓

Configure Environment Variables

↓

Start Flask Application

↓

Public Web URL
```

For production environments it is recommended to:

- Disable Flask Debug mode
- Use Gunicorn or Waitress
- Configure PostgreSQL
- Store secrets as environment variables
- Enable HTTPS

---

# Project Documentation

In addition to this README, the repository includes detailed engineering documentation.

## ENGINEERING_NOTES.md

Documents how the application evolved throughout development and explains the reasoning behind major implementation phases.

---

## DECISION_LOG.md

Records important architectural decisions, alternatives considered, and the trade-offs accepted.

---

## MISTAKES_TO_AVOID.md

Documents implementation mistakes encountered during development and explains how they were resolved.

---

## REVISION_NOTES.md

Provides a concise summary of the project for interview preparation and quick revision.

---

## ARCHITECTURE.md

Describes the system architecture, module interactions, and request flow.

---

## PROJECT_PLAN.md

Tracks development milestones, completed features, and future objectives.

---

# Known Limitations

The project intentionally focuses on demonstrating end-to-end machine learning integration rather than building a production trading platform.

Current limitations include:

- Traditional machine learning models only
- No deep learning forecasting
- No user authentication
- No portfolio management
- No watchlists
- No news sentiment analysis
- Daily historical updates instead of real-time streaming
- No automated model retraining

These limitations were accepted to keep the project focused, modular, and suitable for educational and portfolio purposes.

---

# Future Work

Several enhancements are planned for future iterations.

## Machine Learning

- LSTM-based forecasting
- Transformer models
- XGBoost
- Ensemble learning
- Hyperparameter optimization

---

## Backend

- Authentication
- User accounts
- Portfolio management
- Watchlists
- REST API versioning
- Background task queues

---

## Data

- Live market feeds
- News sentiment
- Financial statement integration
- Technical indicator expansion

---

## Infrastructure

- Docker support
- CI/CD pipelines
- Model versioning
- Redis caching
- Kubernetes deployment
- Automated retraining

---

# License

This project is licensed under the MIT License.

The project was developed for educational purposes, software engineering practice, and portfolio demonstration.

---

# Author

**Bala Sathwik Nagothu**

Computer Science Engineering Student

VIT-AP University

GitHub

https://github.com/balasathwiknagothu

---

# Acknowledgements

This project builds upon several outstanding open-source technologies.

Special thanks to

- Flask
- Scikit-learn
- Pandas
- NumPy
- SQLAlchemy
- APScheduler
- Yahoo Finance (yfinance)
- TradingView Lightweight Charts
- PostgreSQL
- GitHub

Their libraries and documentation greatly simplified the implementation of this project.
