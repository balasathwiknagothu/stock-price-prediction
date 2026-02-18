## System Architecture Overview

The project follows a layered architecture design where each component has a clearly defined responsibility. The system is designed to automatically collect real-time stock market data, process and store the data, apply machine learning models for prediction, and expose the results through a web-based interface.

## Architecture Layers

### 1. Data Source Layer
This layer consists of external stock market data providers. These sources provide live and historical stock price data such as open price, close price, high, low, and volume. The system relies only on reliable and publicly available APIs.

### 2. Data Fetching and Scheduling Layer
This layer is responsible for automatically fetching stock data at fixed intervals (every 15 minutes). A scheduling mechanism ensures that data collection happens without manual intervention. Error handling and logging are included to manage API failures or network issues.

### 3. Data Storage Layer
Fetched data is cleaned, validated, and stored in a structured format. This layer maintains both historical and recent data to support training and prediction. The storage solution is lightweight, efficient, and easy to version.

### 4. Machine Learning Layer
This layer performs feature engineering, model training, and prediction. Multiple machine learning models are evaluated, and the best-performing model is selected. The trained model is saved and reused for real-time predictions.

### 5. Backend API Layer
The backend exposes RESTful endpoints using Flask. It acts as a bridge between the machine learning model and the frontend interface. The backend handles user requests, invokes prediction logic, and returns results in a structured format.

### 6. Frontend Layer
The frontend provides a user-friendly interface accessible through a web browser. Users can input stock symbols, view predictions, and visualize trends. The frontend communicates with the backend API to fetch results.

## Data Flow Description

1. The system periodically requests stock data from external APIs.
2. Retrieved data is validated and cleaned to ensure consistency.
3. Clean data is stored for historical reference and future predictions.
4. Feature engineering transforms raw data into model-ready inputs.
5. Machine learning models generate predictions based on processed data.
6. The backend API serves prediction results to the frontend.
7. Users interact with the system through a web browser.
