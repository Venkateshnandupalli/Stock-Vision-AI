"""
StockVision AI — Centralized Configuration
==========================================
All application settings are loaded from environment variables (.env).
Import `settings` anywhere in the project.

Usage:
    from src.utils.config import settings
    print(settings.db_url)
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field, computed_field

# ── Project root ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Typed, validated application settings loaded from .env"""

    # ── Database ──────────────────────────────────────────────────────────
    db_host: str = Field(default="localhost", alias="DB_HOST")
    db_port: int = Field(default=5432, alias="DB_PORT")
    db_name: str = Field(default="stockvision_db", alias="DB_NAME")
    db_user: str = Field(default="postgres", alias="DB_USER")
    db_password: str = Field(default="", alias="DB_PASSWORD")

    # ── External APIs ─────────────────────────────────────────────────────
    alpha_vantage_api_key: str = Field(default="", alias="ALPHA_VANTAGE_API_KEY")

    # ── MLflow ────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = Field(
        default=f"sqlite:///{PROJECT_ROOT}/models/mlflow.db",
        alias="MLFLOW_TRACKING_URI",
    )
    mlflow_experiment_name: str = Field(
        default="stockvision_ai", alias="MLFLOW_EXPERIMENT_NAME"
    )

    # ── Application ───────────────────────────────────────────────────────
    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: str = Field(default="logs/stockvision.log", alias="LOG_FILE")

    # ── Data ──────────────────────────────────────────────────────────────
    historical_start_date: str = Field(
        default="2019-01-01", alias="HISTORICAL_START_DATE"
    )
    data_source: str = Field(default="yfinance", alias="DATA_SOURCE")

    # ── Servers ───────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    model_config = {
        "env_file": str(PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }

    @computed_field
    @property
    def db_url(self) -> str:
        """SQLAlchemy connection string for PostgreSQL"""
        return (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @computed_field
    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"


# ── Stock Universe ─────────────────────────────────────────────────────────
STOCK_UNIVERSE = {
    "Information Technology": ["TCS.NS", "INFY.NS", "WIPRO.NS"],
    "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"],
    "Energy": ["RELIANCE.NS", "ONGC.NS"],
    "Automobile": ["TATAMOTORS.NS", "MARUTI.NS"],
}

BENCHMARK_TICKER = "^NSEI"
BENCHMARK_NAME = "NIFTY 50"

ALL_TICKERS = [
    ticker
    for tickers in STOCK_UNIVERSE.values()
    for ticker in tickers
]

COMPANY_INFO = {
    "TCS.NS":        {"name": "Tata Consultancy Services",  "sector": "Information Technology", "exchange": "NSE"},
    "INFY.NS":       {"name": "Infosys Limited",             "sector": "Information Technology", "exchange": "NSE"},
    "WIPRO.NS":      {"name": "Wipro Limited",               "sector": "Information Technology", "exchange": "NSE"},
    "HDFCBANK.NS":   {"name": "HDFC Bank Limited",           "sector": "Banking",               "exchange": "NSE"},
    "ICICIBANK.NS":  {"name": "ICICI Bank Limited",          "sector": "Banking",               "exchange": "NSE"},
    "SBIN.NS":       {"name": "State Bank of India",         "sector": "Banking",               "exchange": "NSE"},
    "RELIANCE.NS":   {"name": "Reliance Industries Limited", "sector": "Energy",                "exchange": "NSE"},
    "ONGC.NS":       {"name": "Oil and Natural Gas Corp",    "sector": "Energy",                "exchange": "NSE"},
    "TATAMOTORS.NS": {"name": "Tata Motors Limited",         "sector": "Automobile",            "exchange": "NSE"},
    "MARUTI.NS":     {"name": "Maruti Suzuki India Limited", "sector": "Automobile",            "exchange": "NSE"},
    "^NSEI":         {"name": "NIFTY 50 Index",              "sector": "Benchmark",             "exchange": "NSE"},
}

# ── Feature Engineering Config ─────────────────────────────────────────────
SMA_WINDOWS = [5, 10, 20, 50]
EMA_WINDOWS = [20]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_WINDOW = 20
ATR_PERIOD = 14
VOLATILITY_WINDOW = 20
VOLUME_MA_WINDOW = 20

# ── Model Config ───────────────────────────────────────────────────────────
FORECAST_HORIZONS = [1, 5]          # next-day and next-5-day
TRAIN_END_DATE = "2023-12-31"
VALIDATION_END_DATE = "2024-12-31"
# Test: everything after VALIDATION_END_DATE

N_SPLITS_TIMESERIES = 4             # walk-forward folds

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_RAW_DIR       = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL_DIR  = PROJECT_ROOT / "data" / "external"
MODELS_DIR         = PROJECT_ROOT / "models"
REPORTS_DIR        = PROJECT_ROOT / "reports"
LOGS_DIR           = PROJECT_ROOT / "logs"
SQL_DIR            = PROJECT_ROOT / "sql"

# Create directories if they do not exist
for _dir in [DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_EXTERNAL_DIR,
             MODELS_DIR, REPORTS_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Singleton settings instance ────────────────────────────────────────────
settings = Settings()
