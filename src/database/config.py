#!/usr/bin/env python3
"""
Database Configuration
Purpose: Centralized database configuration and connection management
Usage: from database.config import get_database_url, get_session
"""

from __future__ import annotations
import os
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool


def get_database_url() -> str:
    """
    Get database URL from environment or use default.

    Returns:
        Database connection string
    """
    # Check for environment variable
    db_url = os.environ.get("DATABASE_URL")

    if db_url:
        # Handle Heroku-style postgres:// URLs
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        return db_url

    # Default to local PostgreSQL
    db_host = os.environ.get("DB_HOST", "localhost")
    db_port = os.environ.get("DB_PORT", "5432")
    db_name = os.environ.get("DB_NAME", "efab_erp")
    db_user = os.environ.get("DB_USER", "postgres")
    db_password = os.environ.get("DB_PASSWORD", "password")

    return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


def get_sqlite_url(db_path: str = "efab_erp.db") -> str:
    """
    Get SQLite database URL for testing.

    Args:
        db_path: Path to SQLite database file

    Returns:
        SQLite connection string
    """
    return f"sqlite:///{db_path}"


# Global engine and session factory
_engine = None
_SessionLocal = None


def init_database(database_url: Optional[str] = None, echo: bool = False) -> tuple:
    """
    Initialize database engine and session factory.

    Args:
        database_url: Database connection string (uses env if not provided)
        echo: Whether to echo SQL statements

    Returns:
        Tuple of (engine, SessionLocal)
    """
    global _engine, _SessionLocal

    if not database_url:
        database_url = get_database_url()

    # Create engine with appropriate settings
    if "sqlite" in database_url:
        # SQLite settings
        _engine = create_engine(
            database_url,
            echo=echo,
            connect_args={"check_same_thread": False}
        )
    else:
        # PostgreSQL settings
        _engine = create_engine(
            database_url,
            echo=echo,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600
        )

    # Create session factory
    _SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=_engine
    )

    return _engine, _SessionLocal


def get_engine():
    """Get database engine, initializing if needed."""
    global _engine
    if _engine is None:
        init_database()
    return _engine


def get_session() -> Session:
    """
    Get database session.

    Returns:
        SQLAlchemy session

    Usage:
        with get_session() as session:
            # Do database operations
            session.commit()
    """
    global _SessionLocal
    if _SessionLocal is None:
        init_database()
    return _SessionLocal()


def create_tables() -> None:
    """Create all database tables."""
    from database.models import Base

    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created")


def drop_tables() -> None:
    """Drop all database tables."""
    from database.models import Base

    engine = get_engine()
    Base.metadata.drop_all(bind=engine)
    print("✓ Database tables dropped")


def test_connection() -> bool:
    """
    Test database connection.

    Returns:
        True if connection successful
    """
    try:
        from sqlalchemy import text

        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False


if __name__ == "__main__":
    # Test configuration
    print("Database Configuration Test")
    print("-" * 40)

    # Show configuration
    db_url = get_database_url()
    print(f"Database URL: {db_url}")

    # Test connection
    if test_connection():
        print("✓ Connection successful")

        # Create tables
        create_tables()

        # Show table info
        from sqlalchemy import inspect

        engine = get_engine()
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print(f"\nTables created: {len(tables)}")
        for table in tables:
            print(f"  - {table}")
    else:
        print("✗ Connection failed")