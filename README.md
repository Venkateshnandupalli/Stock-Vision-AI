# StockVision AI 📈
## Market Analytics, Risk Intelligence & Price Forecasting Platform

> An end-to-end stock market analytics platform that collects historical market data,
> analyses price trends and volatility, engineers technical indicators, forecasts
> short-term returns, evaluates prediction reliability, and presents actionable
> market insights through interactive dashboards.

**⚠️ Educational and analytical use only. Not financial advice.**

---

## 🎯 Business Problem

Investors and financial analysts work with large quantities of stock-market data.
Raw prices alone do not clearly explain:

- How a stock performs relative to its benchmark (NIFTY 50)
- Whether risk and volatility are increasing
- What market conditions are associated with price movements
- Whether a forecasting model performs better than a simple random-walk baseline
- How different stocks compare by return, risk, drawdown, and momentum

### Solution

StockVision AI automates this analysis end-to-end:

```
API (yfinance) → ETL Pipeline → PostgreSQL → Feature Engineering
     → ML Models → FastAPI → Streamlit Dashboard
```

---

## 🏗️ Architecture

```
Stock Data API (yfinance / Alpha Vantage)
         │
         ▼
Python Data Ingestion Pipeline
  ├── Validation (OHLC integrity, nulls, duplicates)
  ├── Cleaning (forward-fill, dtype enforcement)
  └── Incremental loading (only fetch new data)
         │
         ▼
PostgreSQL Database
  ├── companies, stock_prices, technical_indicators
  ├── model_predictions, model_metrics, pipeline_logs
  └── Analytical Views (Power BI & API layer)
         │
    ┌────┴────┐
    │         │
    ▼         ▼
Analytics   Feature Engineering Pipeline
Pipeline      ├── 50+ indicators (return, trend,
    │         │   momentum, risk, volume, calendar)
    │         └── Lag-safe (no future leakage)
    │               │
    ▼               ▼
Power BI        ML Models
Dashboard         ├── Naive Baseline
                  ├── Linear / Ridge Regression
                  ├── Random Forest
                  ├── Gradient Boosting
                  └── XGBoost (best performer)
                        │
                        ▼
                 FastAPI REST API
                        │
                        ▼
                 Streamlit Dashboard
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Data ingestion | yfinance, requests, tenacity |
| Data processing | Pandas, NumPy |
| Database | PostgreSQL 16 + SQLAlchemy |
| Machine Learning | Scikit-learn, XGBoost, Statsmodels |
| Experiment tracking | MLflow |
| Web dashboard | Streamlit |
| REST API | FastAPI + Uvicorn |
| Visualization | Plotly, Matplotlib |
| Testing | Pytest |
| Containerization | Docker + Docker Compose |
| Version control | Git + GitHub |

---

## 📊 Stock Universe

| Sector | Stocks |
|---|---|
| Information Technology | TCS.NS, INFY.NS, WIPRO.NS |
| Banking | HDFCBANK.NS, ICICIBANK.NS, SBIN.NS |
| Energy | RELIANCE.NS, ONGC.NS |
| Automobile | TATAMOTORS.NS, MARUTI.NS |
| Benchmark | ^NSEI (NIFTY 50) |

**Historical data:** January 2019 — present (daily OHLCV)

---

## 📐 Forecasting Targets

| Target | Type | Description |
|---|---|---|
| `target_return_1d` | Regression | Next-day % return |
| `target_return_5d` | Regression | Next-5-day % return |
| `target_direction_1d` | Classification | UP (1) or DOWN (0) next day |
| `target_direction_5d` | Classification | UP (1) or DOWN (0) in 5 days |

---

## 🗂️ Project Structure

```
├── src/
│   ├── ingestion/      ← fetch_market_data.py (ETL pipeline)
│   ├── processing/     ← clean_data.py, validate_data.py
│   ├── features/       ← build_features.py (50+ indicators)
│   ├── models/         ← train.py, evaluate.py, predict.py
│   ├── database/       ← connection.py, queries.py
│   └── utils/          ← config.py, logger.py
├── dashboard/
│   ├── app.py          ← Streamlit home (market overview)
│   └── pages/
│       ├── 01_stock_explorer.py
│       ├── 02_forecast_centre.py
│       ├── 03_model_lab.py
│       └── 04_data_quality.py
├── api/
│   └── main.py         ← FastAPI (8 endpoints)
├── sql/
│   ├── schema.sql      ← PostgreSQL schema (6 tables)
│   └── analytical_views.sql ← 6 analytical views
├── tests/
│   ├── test_ingestion.py
│   ├── test_features.py  ← includes leakage detection
│   └── test_models.py
├── notebooks/          ← EDA, features, model experiments
├── data/               ← raw, processed, external
├── models/             ← saved .joblib models
├── reports/            ← generated reports
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL 16 (local or Docker)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/your-username/stockvision-ai.git
cd stockvision-ai
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set your PostgreSQL password:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=stockvision_db
DB_USER=postgres
DB_PASSWORD=your_password_here
```

### 5. Initialize the database

```bash
# Create database in PostgreSQL
psql -U postgres -c "CREATE DATABASE stockvision_db;"

# Run schema
psql -U postgres -d stockvision_db -f sql/schema.sql
psql -U postgres -d stockvision_db -f sql/analytical_views.sql
```

### 6. Run the ingestion pipeline

```bash
# Download all historical data (first run — takes 3–5 minutes)
python -m src.ingestion.fetch_market_data --full-reload

# Subsequent runs (incremental — only fetches new data)
python -m src.ingestion.fetch_market_data
```

### 7. Build features

```bash
python -m src.features.build_features
```

### 8. Train models

```bash
# Train all models for all tickers
python -m src.models.train

# Train for a specific ticker
python -m src.models.train --ticker TCS.NS --task regression
```

### 9. Launch the Streamlit dashboard

```bash
streamlit run dashboard/app.py
```
Open [http://localhost:8501](http://localhost:8501)

### 10. Launch the FastAPI

```bash
uvicorn api.main:app --reload --port 8000
```
API docs at [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🐳 Docker Deployment

```bash
# Build and start all services
docker compose up --build

# Services:
#   PostgreSQL    → localhost:5432
#   FastAPI       → localhost:8000
#   Streamlit     → localhost:8501
#   MLflow UI     → localhost:5000
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v --tb=short

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 📊 Model Results

> ⚠️ **Note:** Results below are populated after training. Run the training pipeline to generate actual metrics.

```
Best Regression Model:    XGBoost Regressor
Horizon:                  1-day next return

Test MAE:                 [Run training to fill]
Baseline MAE (naive):     [Run training to fill]
Improvement over baseline:[Run training to fill]

Directional Accuracy:     [Run training to fill]

Best Classification Model: XGBoost Classifier
Direction F1-Score:       [Run training to fill]
ROC-AUC:                  [Run training to fill]

Top Features:
  1. rsi_14
  2. macd_hist
  3. volatility_20d
  4. relative_volume
  5. return_5d

Important Limitation:
  Historical market patterns may not remain stable
  under changing market conditions. Model performance
  during high-volatility periods may degrade.
```

---

## 🗄️ Database Schema

```
companies          ← Reference: ticker, sector, name
stock_prices       ← Daily OHLCV (6M+ rows after full load)
technical_indicators ← Pre-computed RSI, MACD, SMA, volatility
model_predictions  ← Stored forecasts with actuals
model_metrics      ← Evaluation results per model/ticker
pipeline_logs      ← Audit trail for all ETL runs
```

**Key analytical views:**
- `vw_daily_stock_performance` — Daily returns + OHLCV
- `vw_monthly_returns` — Calendar heatmap data
- `vw_stock_risk_summary` — Risk KPIs (volatility, drawdown, Sharpe)
- `vw_sector_performance` — Sector-level aggregates
- `vw_latest_predictions` — Most recent model forecasts
- `vw_model_performance` — Model leaderboard

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | API and DB health check |
| GET | `/stocks` | List all tracked stocks |
| GET | `/stocks/{ticker}/history` | Historical OHLCV data |
| GET | `/stocks/{ticker}/indicators` | Technical indicators |
| GET | `/stocks/{ticker}/analytics` | Risk KPIs (volatility, drawdown, Sharpe) |
| GET | `/stocks/{ticker}/forecast` | ML-based return forecast |
| GET | `/models/metrics` | Model evaluation results |
| POST | `/pipeline/refresh` | Trigger data ingestion |

---

## 📓 Notebooks

| Notebook | Purpose |
|---|---|
| `01_data_understanding.ipynb` | Data shape, distributions, quality |
| `02_eda.ipynb` | Returns, volatility, correlation, seasonality |
| `03_feature_engineering.ipynb` | Feature analysis, importance preview |
| `04_model_experiments.ipynb` | Walk-forward validation, leaderboard |

---

## ⚠️ Important Limitations

1. **Market unpredictability**: Financial markets are influenced by news, policy, and sentiment that cannot be captured in historical price data.
2. **Non-stationarity**: Return distributions shift over time — models trained on past data may underperform in changed conditions.
3. **No look-ahead**: All features are lagged to prevent future data leakage, but real-world factors (earnings releases, geopolitical events) are not modelled.
4. **Not investment advice**: This platform is designed for educational analytics and portfolio demonstration only.

---

## 🔮 Future Enhancements

- [ ] Sentiment analysis from financial news
- [ ] Options and derivatives data
- [ ] Portfolio optimization (Markowitz efficient frontier)
- [ ] Real-time streaming data
- [ ] React frontend replacing Streamlit
- [ ] LSTM and Transformer models
- [ ] Automated daily pipeline scheduling

---

## 📋 Resume Description

> **StockVision AI — Market Analytics and Forecasting Platform**
> Developed an end-to-end stock market analytics platform using Python, PostgreSQL, SQL,
> Streamlit and machine learning. Automated historical market-data ingestion from yfinance,
> engineered 50+ return, momentum, volatility and volume indicators, and analysed stock
> performance against the NIFTY 50 benchmark. Built and evaluated regression and
> classification models using time-series walk-forward validation, then deployed
> interactive dashboards for risk analysis, model comparison and short-term forecasting.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built for Data Analyst portfolio demonstration. Educational use only.*
