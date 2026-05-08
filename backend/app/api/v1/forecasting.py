"""
SmartKitchen AI X — Forecasting API
AI-powered demand prediction endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import date, datetime, timedelta
from typing import List
import pandas as pd
from pathlib import Path

from app.database import get_db
from app.models.prediction import Prediction
from app.models.kitchen import KitchenMember
from app.models.user import User
from app.schemas.schemas import (
    ForecastResponse, FeedbackRequest, AccuracyResponse, MealType, PredictionItem
)
from app.api.v1.auth import get_current_user
from app.ml.predict import ForecastPredictor
from app.ml.generate_data import generate_synthetic_data

router = APIRouter()

# Initialize predictor (singleton)
predictor = ForecastPredictor()


def verify_kitchen_access(db: Session, kitchen_id: str, user_id: str):
    membership = (
        db.query(KitchenMember)
        .filter(KitchenMember.kitchen_id == kitchen_id, KitchenMember.user_id == user_id)
        .first()
    )
    if not membership:
        raise HTTPException(status_code=403, detail="Not a member of this kitchen")
    return membership


def get_historical_data(kitchen_id: str) -> pd.DataFrame:
    """
    Load historical data for a kitchen.
    Falls back to synthetic data for demo/development.
    """
    data_path = Path(__file__).parent.parent.parent / "ml" / "data" / "synthetic_meal_data.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    # Generate synthetic data on-the-fly for demo
    return generate_synthetic_data()


@router.get("/kitchens/{kitchen_id}/forecast", response_model=ForecastResponse)
async def get_forecast(
    kitchen_id: str,
    target_date: date = Query(default=None, alias="date"),
    meal_type: MealType = Query(default=MealType.lunch),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get AI demand forecast for a specific date and meal type.

    Returns predicted quantities for each food item with confidence intervals.
    """
    verify_kitchen_access(db, kitchen_id, current_user.id)

    if target_date is None:
        target_date = date.today() + timedelta(days=1)  # Default: tomorrow

    # Check cache (existing predictions in DB)
    existing = (
        db.query(Prediction)
        .filter(
            Prediction.kitchen_id == kitchen_id,
            Prediction.date == target_date,
            Prediction.meal_type == meal_type.value,
        )
        .all()
    )

    if existing:
        predictions = [
            PredictionItem(
                item=p.item,
                predicted_kg=p.predicted_quantity,
                confidence=p.confidence or 0.85,
                lower_bound=p.lower_bound or p.predicted_quantity * 0.8,
                upper_bound=p.upper_bound or p.predicted_quantity * 1.2,
            )
            for p in existing
        ]
        return ForecastResponse(
            kitchen_id=kitchen_id,
            date=target_date,
            meal_type=meal_type,
            predictions=predictions,
            factors=[],
            model_version=existing[0].model_version or "cached",
        )

    # Generate new predictions
    historical_df = get_historical_data(str(kitchen_id))

    try:
        result = predictor.predict_full_meal(
            historical_df=historical_df,
            target_date=target_date,
            meal_type=meal_type.value,
            kitchen_id="default",  # Use default models for now
            weather={"temperature": 30, "humidity": 60, "is_rainy": 0},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Store predictions in DB
    predictions = []
    for pred in result["predictions"]:
        if "error" not in pred:
            db_pred = Prediction(
                kitchen_id=kitchen_id,
                date=target_date,
                meal_type=meal_type.value,
                item=pred["item"],
                predicted_quantity=pred["predicted_kg"],
                confidence=pred["confidence"],
                lower_bound=pred["lower_bound"],
                upper_bound=pred["upper_bound"],
                model_version=pred.get("model_version", "v1"),
            )
            db.add(db_pred)

            predictions.append(PredictionItem(
                item=pred["item"],
                predicted_kg=pred["predicted_kg"],
                confidence=pred["confidence"],
                lower_bound=pred["lower_bound"],
                upper_bound=pred["upper_bound"],
            ))

    db.commit()

    return ForecastResponse(
        kitchen_id=kitchen_id,
        date=target_date,
        meal_type=meal_type,
        predictions=predictions,
        weather=result.get("weather"),
        factors=result.get("factors", []),
        model_version=result.get("model_version", "v1"),
    )


@router.post("/kitchens/{kitchen_id}/forecast/feedback")
async def submit_feedback(
    kitchen_id: str,
    data: FeedbackRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Submit actual consumption data for a past prediction.
    This feeds the model retraining loop.
    """
    verify_kitchen_access(db, kitchen_id, current_user.id)

    prediction = (
        db.query(Prediction)
        .filter(
            Prediction.kitchen_id == kitchen_id,
            Prediction.date == data.date,
            Prediction.meal_type == data.meal_type.value,
            Prediction.item == data.item,
        )
        .first()
    )

    if prediction:
        prediction.actual_quantity = data.actual_kg
        db.commit()
        error_pct = prediction.error_pct
        return {
            "status": "updated",
            "prediction_id": str(prediction.id),
            "predicted": prediction.predicted_quantity,
            "actual": data.actual_kg,
            "error_pct": round(error_pct, 1) if error_pct else None,
        }
    else:
        # Create prediction record with just the actual
        new_pred = Prediction(
            kitchen_id=kitchen_id,
            date=data.date,
            meal_type=data.meal_type.value,
            item=data.item,
            predicted_quantity=0,
            actual_quantity=data.actual_kg,
            model_version="feedback_only",
        )
        db.add(new_pred)
        db.commit()
        return {"status": "created", "message": "Actual logged (no prediction to compare)"}


@router.get("/kitchens/{kitchen_id}/forecast/accuracy")
async def get_forecast_accuracy(
    kitchen_id: str,
    period: int = Query(default=30, description="Number of days to evaluate"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get model accuracy metrics for the kitchen."""
    verify_kitchen_access(db, kitchen_id, current_user.id)

    cutoff = date.today() - timedelta(days=period)

    predictions = (
        db.query(Prediction)
        .filter(
            Prediction.kitchen_id == kitchen_id,
            Prediction.date >= cutoff,
            Prediction.actual_quantity.isnot(None),
            Prediction.predicted_quantity > 0,
        )
        .all()
    )

    if not predictions:
        return {
            "overall_mape": 0,
            "overall_rmse": 0,
            "per_item": [],
            "period_days": period,
            "total_predictions": 0,
            "message": "No feedback data available yet. Submit actual consumption data to see accuracy.",
        }

    # Calculate metrics
    import numpy as np

    actuals = [p.actual_quantity for p in predictions]
    preds_vals = [p.predicted_quantity for p in predictions]

    errors = [abs(a - p) / a * 100 for a, p in zip(actuals, preds_vals) if a > 0]
    overall_mape = np.mean(errors) if errors else 0
    overall_rmse = np.sqrt(np.mean([(a - p) ** 2 for a, p in zip(actuals, preds_vals)]))

    # Per-item breakdown
    items = set(p.item for p in predictions)
    per_item = []
    for item_name in items:
        item_preds = [p for p in predictions if p.item == item_name]
        item_actuals = [p.actual_quantity for p in item_preds]
        item_predicted = [p.predicted_quantity for p in item_preds]
        item_errors = [abs(a - p) / a * 100 for a, p in zip(item_actuals, item_predicted) if a > 0]
        per_item.append({
            "item": item_name,
            "mape": round(np.mean(item_errors), 1) if item_errors else 0,
            "rmse": round(np.sqrt(np.mean([(a - p) ** 2 for a, p in zip(item_actuals, item_predicted)])), 2),
            "n_predictions": len(item_preds),
        })

    per_item.sort(key=lambda x: x["mape"])

    return {
        "overall_mape": round(overall_mape, 1),
        "overall_rmse": round(overall_rmse, 2),
        "per_item": per_item,
        "period_days": period,
        "total_predictions": len(predictions),
    }
