"""
StockVision AI — Database Initialization & Setup
=================================================
Establishes the database connection, creates tables, loads seed data,
and registers analytical views on PostgreSQL / Supabase.

Usage:
    python setup_database.py
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

from src.database.connection import get_engine_singleton, test_connection, init_database
from src.utils.logger import logger
from sqlalchemy import text

def run_sql_file(file_path: Path, description: str) -> bool:
    """Executes a SQL file containing multiple statements separated by semicolons."""
    if not file_path.exists():
        logger.error(f"{description} file not found at {file_path}")
        return False
    
    logger.info(f"Running {description} script from {file_path.name}...")
    try:
        engine = get_engine_singleton()
        sql_content = file_path.read_text(encoding="utf-8")
        
        # Split statements by semicolon, ignoring comments and whitespace
        statements = []
        current_stmt = []
        for line in sql_content.splitlines():
            # Strip comments
            stripped = line.strip()
            if stripped.startswith("--") or not stripped:
                continue
            current_stmt.append(line)
            if ";" in line:
                statements.append("\n".join(current_stmt))
                current_stmt = []
        
        # Handle trailing statement without semicolon
        if current_stmt:
            statements.append("\n".join(current_stmt))
            
        with engine.connect() as conn:
            for stmt in statements:
                stmt_clean = stmt.strip()
                if stmt_clean:
                    # Remove trailing semicolon if SQLAlchemy objects
                    conn.execute(text(stmt_clean))
            conn.commit()
            
        logger.info(f"✅ {description} executed successfully.")
        return True
    except Exception as exc:
        logger.error(f"❌ Failed to run {description}: {exc}")
        return False

def main():
    logger.info("Starting StockVision AI Database Setup...")
    
    # 1. Verify connection
    if not test_connection():
        logger.error("❌ Could not establish database connection. Please verify database is running and configuration in .env is correct.")
        sys.exit(1)
        
    # 2. Run schema initialization
    logger.info("Initializing schema...")
    try:
        init_database(drop_existing=False)
        logger.info("✅ Database schema initialized.")
    except Exception as exc:
        logger.error(f"❌ Schema initialization failed: {exc}")
        sys.exit(1)
        
    # 3. Seed Companies reference table
    sql_dir = PROJECT_ROOT / "sql"
    seed_file = sql_dir / "seed_companies.sql"
    if not run_sql_file(seed_file, "Seeding Companies"):
        sys.exit(1)
        
    # 4. Create Analytical Views
    views_file = sql_dir / "analytical_views.sql"
    if not run_sql_file(views_file, "Creating Analytical Views"):
        sys.exit(1)
        
    logger.info("🎉 Database setup complete! All tables and views are configured and ready.")

if __name__ == "__main__":
    main()
