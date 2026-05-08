"""
SmartKitchen AI X — Database Connection
SQLAlchemy engine, session factory, and Base model.
Supports both SQLite (dev) and PostgreSQL (production).
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import get_settings

settings = get_settings()

# ── Engine configuration ──
# SQLite doesn't support pool_size/max_overflow, so we detect and handle it
db_url = settings.DATABASE_URL
is_sqlite = db_url.startswith("sqlite")

if is_sqlite:
    engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False},  # Required for SQLite + FastAPI
        echo=settings.APP_DEBUG,
    )
else:
    engine = create_engine(
        db_url,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,  # reconnect on stale connections
        echo=settings.APP_DEBUG,
    )

# ── Session Factory ──
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ── Base Model ──
Base = declarative_base()


def get_db():
    """Dependency: yields a database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
