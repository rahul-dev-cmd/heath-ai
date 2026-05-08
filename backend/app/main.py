"""
SmartKitchen AI X — FastAPI Application Entry Point
Main application with middleware, routers, and startup events.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.config import get_settings
from app.database import engine, Base

# Import all models so they register with SQLAlchemy
from app.models.user import User
from app.models.kitchen import Kitchen, KitchenMember
from app.models.inventory import InventoryItem, StockLog
from app.models.prediction import Prediction, MealPlan
from app.models.waste import WasteLog
from app.models.sensor import Sensor, SensorReading
from app.models.recommendation import Recommendation, AuditLog

settings = get_settings()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    # ── Startup ──
    logger.info("🚀 SmartKitchen AI X starting up...")

    # Create database tables (dev only — use Alembic in production)
    if settings.APP_ENV == "development":
        logger.info("📦 Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables ready")

    yield

    # ── Shutdown ──
    logger.info("👋 SmartKitchen AI X shutting down...")


# ── Create FastAPI app ──
app = FastAPI(
    title="SmartKitchen AI X",
    description=(
        "AI + IoT powered smart kitchen intelligence platform for institutional kitchens. "
        "Provides demand forecasting, inventory monitoring, waste analytics, "
        "and sustainability tracking."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS Middleware ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health Check ──
@app.get("/", tags=["Health"])
async def root():
    return {
        "app": "SmartKitchen AI X",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "database": "connected"}


# ── Import and register routers ──
from app.api.v1 import auth, kitchens, inventory, forecasting, waste, analytics

app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(kitchens.router, prefix="/api/v1/kitchens", tags=["Kitchens"])
app.include_router(inventory.router, prefix="/api/v1", tags=["Inventory"])
app.include_router(forecasting.router, prefix="/api/v1", tags=["Forecasting"])
app.include_router(waste.router, prefix="/api/v1", tags=["Waste"])
app.include_router(analytics.router, prefix="/api/v1", tags=["Analytics"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
