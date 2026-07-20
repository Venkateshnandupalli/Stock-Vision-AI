"""
StockVision AI — Database Connection
=====================================
Provides SQLAlchemy engine and session factory.
Uses connection pooling for efficient DB access.

Usage:
    from src.database.connection import get_engine, get_session
    engine = get_engine()
    with get_session() as session:
        session.execute(...)
"""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase

from src.utils.config import settings
from src.utils.logger import logger


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all ORM models."""
    pass


def get_engine(pool_size: int = 5, max_overflow: int = 10) -> Engine:
    """
    Create and return a SQLAlchemy engine for PostgreSQL.

    Args:
        pool_size: Number of connections to maintain in pool.
        max_overflow: Extra connections allowed above pool_size.

    Returns:
        Configured SQLAlchemy Engine.
    """
    engine = create_engine(
        settings.db_url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,          # verify connections before use
        pool_recycle=3600,           # recycle after 1 hour
        echo=settings.app_env == "development",  # log SQL in dev
    )

    @event.listens_for(engine, "connect")
    def _set_search_path(dbapi_conn, connection_record):
        """Set timezone to UTC for all connections."""
        cursor = dbapi_conn.cursor()
        cursor.execute("SET timezone='UTC'")
        cursor.close()

    return engine


def create_session_factory(engine: Engine) -> sessionmaker:
    """Create a SQLAlchemy session factory bound to the given engine."""
    return sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )


# ── Module-level singletons ────────────────────────────────────────────────
_engine: Engine | None = None
_SessionFactory: sessionmaker | None = None


def get_engine_singleton() -> Engine:
    """Return the module-level engine singleton, creating it if needed."""
    global _engine
    if _engine is None:
        _engine = get_engine()
        logger.info("Database engine created: {host}:{port}/{db}",
                    host=settings.db_host,
                    port=settings.db_port,
                    db=settings.db_name)
    return _engine


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Context manager that yields a database session.
    Automatically commits on success or rolls back on exception.

    Usage:
        with get_session() as session:
            result = session.execute(text("SELECT 1"))
    """
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = create_session_factory(get_engine_singleton())

    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.error("Session rolled back due to: {exc}", exc=exc)
        raise
    finally:
        session.close()


def test_connection() -> bool:
    """
    Verify the database connection is healthy.

    Returns:
        True if connection succeeds, False otherwise.
    """
    try:
        engine = get_engine_singleton()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.scalar()
            logger.info("Database connection OK. PostgreSQL version: {v}", v=version)
        return True
    except Exception as exc:
        logger.error("Database connection FAILED: {exc}", exc=exc)
        return False


def init_database(drop_existing: bool = False) -> None:
    """
    Initialize database schema by running schema.sql.

    Args:
        drop_existing: If True, drops and recreates all tables (DANGEROUS).
    """
    from src.utils.config import SQL_DIR

    schema_file = SQL_DIR / "schema.sql"
    if not schema_file.exists():
        logger.warning("schema.sql not found at {path}", path=schema_file)
        return

    engine = get_engine_singleton()
    with engine.connect() as conn:
        sql = schema_file.read_text(encoding="utf-8")
        conn.execute(text(sql))
        conn.commit()
    logger.info("Database schema initialised successfully.")


if __name__ == "__main__":
    test_connection()
