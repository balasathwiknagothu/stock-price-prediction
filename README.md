# 📈 Stock Price Prediction using Machine Learning

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-REST_API-black?logo=flask)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange?logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical_Computing-013243?logo=numpy)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-336791?logo=postgresql)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-ORM-red)
![License](https://img.shields.io/badge/License-MIT-green)

</p>

A full-stack **Stock Market Analytics and Machine Learning Platform** built using **Python, Flask, Scikit-learn, Pandas, and SQLAlchemy**. The application provides intelligent stock search, historical market analysis, real-time market movers, top gainers, and machine learning-based price prediction through a clean web interface and RESTful APIs.

Unlike a simple prediction model, this project was designed as an end-to-end platform that combines **data engineering**, **machine learning**, **backend development**, and **interactive visualization** into a single application.

Every stage—from collecting historical market data to preprocessing, feature engineering, model training, prediction generation, and API delivery—was implemented with scalability and maintainability in mind.

---

# 📑 Table of Contents

- Overview
- Key Features
- Tech Stack
- Project Architecture
- Machine Learning Pipeline
- Data Flow
- Folder Structure
- REST API Overview
- Getting Started
- Installation
- Running the Application
- Deployment
- Future Improvements
- License

---

# 🚀 Overview

The stock market produces an enormous amount of historical and real-time data every trading day. Extracting meaningful insights from this data requires efficient preprocessing, predictive modeling, and intuitive presentation.

This project addresses those challenges by building a platform capable of:

- Collecting historical stock market data
- Cleaning and preprocessing financial datasets
- Generating technical indicators
- Training Machine Learning models
- Predicting future stock movement
- Serving predictions through REST APIs
- Displaying market information through a responsive dashboard

Rather than focusing solely on model accuracy, the project emphasizes **software engineering best practices**, modular architecture, reusable components, and a clean separation between data processing, machine learning, backend APIs, and presentation.

---

# ✨ Key Features

## 📊 Market Analytics

- Historical stock analysis
- Top market gainers
- Market movers
- Daily market overview
- Search thousands of stocks instantly

---

## 🤖 Machine Learning

- Feature Engineering
- Regression-based prediction models
- Multi-day prediction support
- Pre-trained model loading
- Automatic prediction generation

---

## 🌐 Backend APIs

- Flask REST APIs
- JSON responses
- Modular routing
- Prediction endpoints
- Market data endpoints

---

## 📈 Data Processing

- Historical data collection
- Data validation
- Missing value handling
- Technical indicator generation
- Automated preprocessing pipeline

---

## 🔎 Smart Search

Efficient stock search powered by:

- Symbol lookup
- Company name lookup
- Instant filtering
- Client-side search optimization

---

## 💾 Database Integration

- SQLAlchemy ORM
- PostgreSQL support
- Persistent prediction storage
- Historical market storage

---

## 📱 User Interface

Responsive dashboard featuring:

- Stock search
- Prediction results
- Market movers
- Gainers
- Interactive charts
- Clean modern interface

---

# 🛠 Tech Stack

| Layer | Technology |
|----------|-------------------------|
| Language | Python |
| Backend | Flask |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Database | PostgreSQL, SQLAlchemy |
| Scheduler | APScheduler |
| Market Data | yfinance |
| Frontend | HTML, CSS, JavaScript |
| Charts | Lightweight Charts |
| Deployment | Render |
| Version Control | Git & GitHub |

---

# 🏗 Project Architecture

The application follows a modular architecture separating data collection, machine learning, backend APIs, and frontend presentation.

```

                 +-----------------------+
                 |      User Browser     |
                 +----------+------------+
                            |
                            |
                     Flask REST API
                            |
      +----------+----------+-----------+
      |          |                      |
      |          |                      |
 Prediction   Market APIs          Search API
      |          |                      |
      +----------+----------+-----------+
                            |
                     Business Logic
                            |
      +----------+----------+-----------+
      |                      |          |
      |                      |          |
 ML Models             Historical Data  Database
      |                      |
      +----------+-----------+
                 |
            yFinance Data

```

The modular design allows each component to evolve independently. Machine learning models can be retrained without changing the API layer, while frontend improvements can be made without affecting prediction logic.

---

# 🧠 Machine Learning Pipeline

The prediction workflow consists of several stages:

### 1. Data Collection

Historical stock prices are downloaded from Yahoo Finance.

↓

### 2. Data Cleaning

Missing values, duplicate records, and inconsistent data are removed.

↓

### 3. Feature Engineering

Technical indicators are generated to improve model performance.

↓

### 4. Model Training

Scikit-learn regression models learn relationships between historical patterns and future prices.

↓

### 5. Model Serialization

Trained models are stored as `.pkl` files for fast inference.

↓

### 6. Prediction

Incoming requests are transformed into model-ready features and passed through the trained models.

↓

### 7. REST API Response

Predicted values are returned as JSON and rendered in the web interface.

---

# 🔄 Data Flow

```

Yahoo Finance
       │
       ▼
Historical Dataset
       │
       ▼
Data Cleaning
       │
       ▼
Feature Engineering
       │
       ▼
Machine Learning Model
       │
       ▼
Prediction Engine
       │
       ▼
REST API
       │
       ▼
Frontend Dashboard

```

This pipeline ensures consistent preprocessing during both training and prediction phases.

---

# 📂 Folder Structure

```

stock-price-prediction/

│

├── app.py

├── models/

├── ml/

├── data/

│ ├── historical/

│ ├── master/

│ └── predictions/

├── scripts/

├── static/

├── templates/

├── instance/

├── requirements.txt

├── ARCHITECTURE.md

├── PROJECT_PLAN.md

└── README.md

```

Each directory has a dedicated responsibility, improving maintainability and scalability as the project grows.

---

---

# 📡 REST API Overview

The backend exposes a collection of RESTful APIs that power the frontend dashboard and allow clients to retrieve stock information, market trends, and machine learning predictions.

## Available Endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Load the application dashboard |
| GET | `/stocks` | Retrieve the list of supported stocks |
| GET | `/predict?symbol=<SYMBOL>` | Predict stock price for the requested stock |
| GET | `/market/movers` | Return market movers |
| GET | `/market/gainers` | Return top gaining stocks |

All APIs return JSON responses and are designed to be lightweight, making them suitable for both web and mobile applications.

---

## Example Request

```
GET /predict?symbol=RELIANCE.NS
```

### Example Response

```json
{
    "symbol": "RELIANCE.NS",
    "prediction": {
        "1_day": 1543.82,
        "3_day": 1550.67,
        "7_day": 1564.24
    },
    "last_updated": "2026-07-10T09:30:00"
}
```

---

# ⚙️ Getting Started

## Prerequisites

Before running the project, ensure the following software is installed:

- Python 3.11 or later
- Git
- pip
- Virtual Environment (recommended)
- PostgreSQL (optional if using database features)

---

# 📥 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/balasathwiknagothu/stock-price-prediction.git

cd stock-price-prediction
```

---

## 2. Create Virtual Environment

Windows

```bash
python -m venv venv
```

Activate

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment

Update the required configuration values such as database credentials and application settings before launching the application.

Example configuration:

```
DATABASE_URL=postgresql://username:password@localhost:5432/stocks

FLASK_ENV=development
```

---

## 5. Run the Application

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

# 🖥 Using the Application

After launching the application you can

- Search stocks using ticker symbols or company names.
- View historical market data.
- Explore market movers.
- View top gaining stocks.
- Generate machine learning predictions.
- Analyze historical trends.
- Explore prediction confidence.

The application has been designed to provide an intuitive workflow suitable for both beginners and experienced users interested in stock market analytics.

---

# ☁️ Deployment

The project is deployment-ready and can be hosted on cloud platforms such as:

- Render
- Railway
- Heroku (with modifications)
- PythonAnywhere
- AWS EC2
- Azure App Service
- Google Cloud Run

For production deployment:

- Configure environment variables.
- Use PostgreSQL for persistent storage.
- Disable Flask Debug mode.
- Serve using Gunicorn or Waitress.
- Configure reverse proxy (Nginx) if required.

Example:

```bash
gunicorn app:app
```

---

# 📈 Performance Considerations

Several optimizations have been incorporated to improve responsiveness and scalability.

### Data Layer

- Cached historical datasets
- Efficient CSV loading
- Lightweight preprocessing

### Machine Learning

- Pre-trained serialized models
- Fast prediction inference
- Reusable feature engineering pipeline

### Backend

- Modular Flask architecture
- JSON-based REST APIs
- Efficient request handling

### Frontend

- Client-side search
- Responsive UI
- Lightweight JavaScript components

These optimizations help minimize response times while keeping the application simple and maintainable.

---

# 🔮 Future Improvements

The current implementation focuses on providing an end-to-end stock prediction platform while maintaining simplicity. Future enhancements may include:

- Deep Learning models using TensorFlow or PyTorch
- LSTM-based time series forecasting
- Transformer-based forecasting models
- Live WebSocket-based stock updates
- Portfolio management
- Watchlists
- User authentication
- Personalized dashboards
- Technical indicator visualization
- Candlestick charts
- News sentiment analysis
- AI-powered investment insights
- Docker support
- Kubernetes deployment
- CI/CD pipelines using GitHub Actions
- Model retraining automation
- Multi-market support
- Mobile application

---

# 📚 Learning Outcomes

This project provided practical experience in several areas of software engineering and machine learning.

Key learnings include:

- Designing RESTful APIs using Flask.
- Building modular Python applications.
- Processing financial datasets.
- Feature engineering for machine learning.
- Training and evaluating regression models.
- Deploying machine learning applications.
- Integrating databases with SQLAlchemy.
- Building responsive web interfaces.
- Version control using Git and GitHub.
- Cloud deployment fundamentals.

---

# 🤝 Contributing

Contributions are welcome.

If you would like to improve the project:

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push the branch.
5. Open a Pull Request.

Please ensure that new features include appropriate documentation and follow the existing project structure.

---

# 📄 License

This project is licensed under the MIT License.

You are free to use, modify, and distribute this project for educational and personal purposes while preserving the original license.

---

# 👨‍💻 Author

**Bala Sathwik Nagothu**

Computer Science Engineering Student

VIT-AP University

GitHub: https://github.com/balasathwiknagothu

LinkedIn: *(Add your LinkedIn profile here)*

---

# 🙏 Acknowledgements

This project would not have been possible without the excellent open-source ecosystem.

Special thanks to:

- Flask
- Scikit-learn
- Pandas
- NumPy
- SQLAlchemy
- yfinance
- Lightweight Charts
- PostgreSQL
- GitHub

Their tools and documentation significantly contributed to the development of this project.


