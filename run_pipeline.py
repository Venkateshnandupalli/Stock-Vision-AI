"""
StockVision AI — Master Pipeline Orchestrator
=============================================
Single entry-point to run all pipeline phases in the correct order.

Usage:
    # Run full pipeline (ingest → features → predictions)
    python run_pipeline.py

    # Run specific phases
    python run_pipeline.py --phases ingest features

    # Full reload of historical data
    python run_pipeline.py --full-reload

    # Dry run (validate only, no DB writes)
    python run_pipeline.py --dry-run
"""

import argparse
import sys
import time
from datetime import datetime

from src.utils.logger import logger
from src.utils.config import ALL_TICKERS, BENCHMARK_TICKER


def phase_ingest(full_reload: bool = False, dry_run: bool = False) -> bool:
    """Phase 1: Ingest market data from yfinance."""
    logger.info("=" * 60)
    logger.info("PHASE: DATA INGESTION")
    logger.info("=" * 60)
    try:
        from src.ingestion.fetch_market_data import run_pipeline
        summary = run_pipeline(full_reload=full_reload, dry_run=dry_run)
        errors = (summary["status"] == "error").sum()
        success = (summary["status"] == "success").sum()
        logger.info("Ingestion complete — Success: {s} | Errors: {e}", s=success, e=errors)
        return errors == 0
    except Exception as exc:
        logger.error("Ingestion phase failed: {exc}", exc=exc)
        return False


def phase_features(tickers: list[str] | None = None) -> bool:
    """Phase 2: Build feature matrix for all tickers."""
    logger.info("=" * 60)
    logger.info("PHASE: FEATURE ENGINEERING")
    logger.info("=" * 60)
    try:
        from src.features.build_features import build_features_all_tickers
        results = build_features_all_tickers(tickers=tickers)
        logger.info("Features built for {n} tickers.", n=len(results))
        return len(results) > 0
    except Exception as exc:
        logger.error("Feature phase failed: {exc}", exc=exc)
        return False


def phase_train(tickers: list[str] | None = None) -> bool:
    """Phase 3: Train all ML models."""
    logger.info("=" * 60)
    logger.info("PHASE: MODEL TRAINING")
    logger.info("=" * 60)
    try:
        from src.models.train import train_all_tickers
        results = train_all_tickers(tickers=tickers, task="both", horizons=[1, 5])
        logger.info("Training complete for {n} tickers.", n=len(results))
        return True
    except Exception as exc:
        logger.error("Training phase failed: {exc}", exc=exc)
        return False


def phase_predict(tickers: list[str] | None = None) -> bool:
    """Phase 4: Generate and store predictions using the best model."""
    logger.info("=" * 60)
    logger.info("PHASE: PREDICTION GENERATION")
    logger.info("=" * 60)
    try:
        from src.models.predict import generate_all_predictions
        preds = generate_all_predictions(
            tickers=tickers or ALL_TICKERS,
            model_name="xgboost_regressor",
            horizon=1,
        )
        logger.info("Generated {n} predictions.", n=len(preds))
        return True
    except Exception as exc:
        logger.error("Prediction phase failed: {exc}", exc=exc)
        return False


def check_db_connection() -> bool:
    """Verify PostgreSQL connection before running pipeline."""
    try:
        from src.database.connection import test_connection
        ok = test_connection()
        if not ok:
            logger.error("Database connection failed. Check your .env credentials.")
        return ok
    except Exception as exc:
        logger.error("Cannot connect to database: {exc}", exc=exc)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="StockVision AI — Master Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                      # Full pipeline
  python run_pipeline.py --phases ingest      # Only ingest data
  python run_pipeline.py --full-reload        # Full historical reload
  python run_pipeline.py --dry-run            # Validate without writing
  python run_pipeline.py --skip-train         # Skip training (use existing models)
        """
    )
    parser.add_argument(
        "--phases", nargs="+",
        choices=["ingest", "features", "train", "predict"],
        default=["ingest", "features", "predict"],
        help="Phases to run (default: ingest features predict)"
    )
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Specific tickers (default: all configured)")
    parser.add_argument("--full-reload", action="store_true",
                        help="Force full historical data reload")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate without writing to database")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip model training (use existing saved models)")

    args = parser.parse_args()

    start_time = time.time()
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║         StockVision AI — Pipeline Starting               ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info("Start time: {t}", t=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("Phases:     {p}", p=", ".join(args.phases))
    logger.info("Tickers:    {t}", t=args.tickers or "ALL")
    logger.info("Full reload:{r}", r=args.full_reload)

    # Always check DB first
    if not check_db_connection():
        logger.error("Aborting — database not available.")
        sys.exit(1)

    phase_status = {}

    if "ingest" in args.phases:
        phase_status["ingest"] = phase_ingest(
            full_reload=args.full_reload,
            dry_run=args.dry_run,
        )

    if "features" in args.phases:
        phase_status["features"] = phase_features(tickers=args.tickers)

    if "train" in args.phases and not args.skip_train:
        phase_status["train"] = phase_train(tickers=args.tickers)

    if "predict" in args.phases:
        phase_status["predict"] = phase_predict(tickers=args.tickers)

    elapsed = time.time() - start_time

    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║                 PIPELINE SUMMARY                         ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    for phase, ok in phase_status.items():
        status = "✅ SUCCESS" if ok else "❌ FAILED"
        logger.info("  {phase:<12} {status}", phase=phase.upper(), status=status)
    logger.info("  Total time:  {t:.1f}s", t=elapsed)
    logger.info("═" * 60)

    if not all(phase_status.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
