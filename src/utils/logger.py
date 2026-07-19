"""
StockVision AI — Structured Logger
===================================
Uses Loguru for structured, coloured, file-rotated logging.
Import `logger` anywhere in the project.

Usage:
    from src.utils.logger import logger
    logger.info("Pipeline started")
    logger.error("Fetch failed for {ticker}", ticker="TCS.NS")
"""

import sys
from pathlib import Path
from loguru import logger as _loguru_logger

from src.utils.config import settings, LOGS_DIR


def _configure_logger() -> None:
    """Set up Loguru with console + rotating file sink."""
    _loguru_logger.remove()

    # ── Console handler (coloured) ────────────────────────────────────────
    _loguru_logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # ── File handler (rotating, 10 MB, 30-day retention) ──────────────────
    log_path = LOGS_DIR / "stockvision.log"
    _loguru_logger.add(
        str(log_path),
        level=settings.log_level,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} | {message}"
        ),
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        enqueue=True,       # thread-safe
    )

    # ── Ingestion audit log (separate file) ───────────────────────────────
    ingestion_log = LOGS_DIR / "ingestion_audit.log"
    _loguru_logger.add(
        str(ingestion_log),
        level="INFO",
        filter=lambda record: "INGESTION" in record["extra"],
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        rotation="50 MB",
        retention="90 days",
    )


_configure_logger()

# Public logger — import this everywhere
logger = _loguru_logger

__all__ = ["logger"]
