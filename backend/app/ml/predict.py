"""
SmartKitchen AI X — Prediction / Inference Engine
Loads trained models and generates demand forecasts for future dates.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import date, datetime
from typing import Optional, List
import logging

from app.ml.features import FeatureEngineer

logger = logging.getLogger(__name__)


class ForecastPredictor:
    """
    Loads trained XGBoost models and generates predictions.

    Handles:
    - Loading models from disk
    - Building prediction features
    - Post-processing (bounds, rounding)
    - Confidence intervals
    - Batch predictions for full meal
    """

    def __init__(self, model_dir: str = None):
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).parent / "models"
        self.feature_engineer = FeatureEngineer()
        self._model_cache: dict = {}  # Cache loaded models in memory

    def _load_model(self, model_name: str) -> dict:
        """Load model from disk with caching."""
        if model_name in self._model_cache:
            return self._model_cache[model_name]

        model_path = self.model_dir / f"{model_name}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        artifact = joblib.load(model_path)
        self._model_cache[model_name] = artifact
        logger.info(f"Loaded model: {model_name} (v{artifact.get('version', '?')})")
        return artifact

    def predict_single(
        self,
        historical_df: pd.DataFrame,
        item: str,
        meal_type: str,
        target_date: date,
        kitchen_id: str = "default",
        weather: Optional[dict] = None,
        is_holiday: bool = False,
        is_exam: bool = False,
        is_vacation: bool = False,
    ) -> dict:
        """
        Predict demand for a single item + meal_type on a target date.

        Args:
            historical_df: Historical meal data
            item: Food item name (e.g. "Rice")
            meal_type: "lunch" or "dinner"
            target_date: Date to predict for
            kitchen_id: Kitchen ID for model lookup
            weather: Weather dict {"temperature": 34, "humidity": 65, "is_rainy": 0}
            is_holiday, is_exam, is_vacation: Event flags

        Returns:
            Dict with predicted_kg, confidence, lower_bound, upper_bound, factors
        """
        # Load model
        item_slug = item.lower().replace(" ", "_")
        model_name = f"{kitchen_id}_{item_slug}_{meal_type}"

        try:
            artifact = self._load_model(model_name)
        except FileNotFoundError:
            # Fallback to default kitchen model
            model_name = f"default_{item_slug}_{meal_type}"
            artifact = self._load_model(model_name)

        model = artifact["model"]
        feature_columns = artifact["feature_columns"]
        residual_std = artifact["metrics"].get("residual_std", 5.0)

        # Build features
        target_ts = pd.Timestamp(target_date)
        features = self.feature_engineer.build_prediction_features(
            historical_df=historical_df,
            item=item,
            meal_type=meal_type,
            target_date=target_ts,
            weather=weather,
            is_holiday=is_holiday,
            is_exam=is_exam,
            is_vacation=is_vacation,
        )

        # Align feature columns (handle missing columns)
        for col in feature_columns:
            if col not in features.columns:
                features[col] = 0
        features = features[feature_columns]

        # ── Predict ──
        raw_prediction = model.predict(features)[0]

        # ── Post-processing ──
        # 1. Never predict negative
        predicted_kg = max(0, raw_prediction)

        # 2. Round to practical units (0.5 kg increments)
        predicted_kg = round(predicted_kg * 2) / 2

        # 3. Confidence interval (using training residual std)
        confidence = max(0, min(1, 1 - (residual_std / predicted_kg))) if predicted_kg > 0 else 0.5
        lower_bound = max(0, round((predicted_kg - 1.96 * residual_std) * 2) / 2)
        upper_bound = round((predicted_kg + 1.96 * residual_std) * 2) / 2

        # 4. Identify influencing factors
        factors = self._identify_factors(
            target_date=target_date,
            weather=weather,
            is_holiday=is_holiday,
            is_exam=is_exam,
            is_vacation=is_vacation,
            features=features,
            feature_importance=artifact.get("feature_importance", {}),
        )

        return {
            "item": item,
            "meal_type": meal_type,
            "date": str(target_date),
            "predicted_kg": predicted_kg,
            "confidence": round(confidence, 2),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "model_version": artifact.get("version", "unknown"),
            "factors": factors,
        }

    def predict_full_meal(
        self,
        historical_df: pd.DataFrame,
        target_date: date,
        meal_type: str,
        kitchen_id: str = "default",
        items: List[str] = None,
        weather: Optional[dict] = None,
        is_holiday: bool = False,
        is_exam: bool = False,
        is_vacation: bool = False,
    ) -> dict:
        """
        Predict demand for ALL items for a specific meal on a date.

        Returns:
            Dict with date, meal_type, predictions list, weather, and factors
        """
        if items is None:
            items = historical_df["item"].unique().tolist()

        predictions = []
        all_factors = set()

        for item in items:
            try:
                pred = self.predict_single(
                    historical_df=historical_df,
                    item=item,
                    meal_type=meal_type,
                    target_date=target_date,
                    kitchen_id=kitchen_id,
                    weather=weather,
                    is_holiday=is_holiday,
                    is_exam=is_exam,
                    is_vacation=is_vacation,
                )
                predictions.append(pred)
                all_factors.update(pred.get("factors", []))
            except Exception as e:
                logger.error(f"Prediction failed for {item}/{meal_type}: {e}")
                predictions.append({
                    "item": item,
                    "meal_type": meal_type,
                    "date": str(target_date),
                    "predicted_kg": 0,
                    "confidence": 0,
                    "lower_bound": 0,
                    "upper_bound": 0,
                    "error": str(e),
                })

        return {
            "kitchen_id": kitchen_id,
            "date": str(target_date),
            "meal_type": meal_type,
            "predictions": predictions,
            "weather": weather,
            "factors": list(all_factors),
            "model_version": predictions[0].get("model_version", "unknown") if predictions else "none",
        }

    def predict_day(
        self,
        historical_df: pd.DataFrame,
        target_date: date,
        kitchen_id: str = "default",
        items: List[str] = None,
        weather: Optional[dict] = None,
        is_holiday: bool = False,
        is_exam: bool = False,
        is_vacation: bool = False,
    ) -> dict:
        """Predict for all meals in a full day."""
        meals = {}
        for meal_type in ["lunch", "dinner"]:
            meals[meal_type] = self.predict_full_meal(
                historical_df=historical_df,
                target_date=target_date,
                meal_type=meal_type,
                kitchen_id=kitchen_id,
                items=items,
                weather=weather,
                is_holiday=is_holiday,
                is_exam=is_exam,
                is_vacation=is_vacation,
            )

        # Aggregate
        total_predicted = sum(
            p["predicted_kg"]
            for meal in meals.values()
            for p in meal["predictions"]
            if "error" not in p
        )

        return {
            "kitchen_id": kitchen_id,
            "date": str(target_date),
            "meals": meals,
            "total_predicted_kg": round(total_predicted, 1),
            "weather": weather,
        }

    def _identify_factors(
        self,
        target_date: date,
        weather: Optional[dict],
        is_holiday: bool,
        is_exam: bool,
        is_vacation: bool,
        features: pd.DataFrame,
        feature_importance: dict,
    ) -> List[str]:
        """Generate human-readable explanation of prediction factors."""
        factors = []

        dow = target_date.weekday() if isinstance(target_date, date) else pd.Timestamp(target_date).weekday()
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        if dow >= 5:
            factors.append(f"Weekend effect ({day_names[dow]}): reduced attendance expected")

        if is_holiday:
            factors.append("Holiday: significant attendance drop expected")
        if is_exam:
            factors.append("Exam period: higher campus attendance expected")
        if is_vacation:
            factors.append("Vacation period: very low attendance expected")

        if weather:
            temp = weather.get("temperature", 30)
            if temp > 38:
                factors.append(f"Extreme heat ({temp}°C): reduced heavy food consumption")
            elif temp > 35:
                factors.append(f"Hot weather ({temp}°C): slightly reduced appetite")
            elif temp < 10:
                factors.append(f"Cold weather ({temp}°C): increased appetite expected")

            if weather.get("is_rainy"):
                factors.append("Rainy day: students tend to stay on campus")

        # Top feature importance
        top_features = list(feature_importance.keys())[:3]
        if top_features:
            factors.append(f"Key drivers: {', '.join(top_features)}")

        return factors

    def clear_cache(self):
        """Clear model cache (e.g. after retraining)."""
        self._model_cache.clear()
        logger.info("Model cache cleared")


# ── Convenience function ──
def quick_predict(
    data_path: str = None,
    target_date: date = None,
    kitchen_id: str = "default",
) -> dict:
    """
    Quick prediction for testing/demo purposes.

    Usage:
        from app.ml.predict import quick_predict
        result = quick_predict()
    """
    from app.ml.generate_data import generate_synthetic_data

    if target_date is None:
        target_date = date(2026, 5, 10)

    # Load historical data
    if data_path:
        df = pd.read_csv(data_path)
        df["date"] = pd.to_datetime(df["date"])
    else:
        df = generate_synthetic_data()

    predictor = ForecastPredictor()
    result = predictor.predict_day(
        historical_df=df,
        target_date=target_date,
        kitchen_id=kitchen_id,
        weather={"temperature": 34, "humidity": 65, "is_rainy": 0},
    )

    # Pretty print
    print(f"\n🔮 Forecast for {target_date}")
    print(f"{'='*60}")
    for meal_type, meal_data in result["meals"].items():
        print(f"\n🍽️  {meal_type.upper()}")
        for pred in meal_data["predictions"]:
            if "error" not in pred:
                print(
                    f"  {pred['item']:12s}: {pred['predicted_kg']:6.1f} kg "
                    f"(conf: {pred['confidence']:.0%}, "
                    f"range: {pred['lower_bound']:.1f}–{pred['upper_bound']:.1f})"
                )
    print(f"\n📦 Total for day: {result['total_predicted_kg']:.1f} kg")

    if result["meals"]["lunch"]["factors"]:
        print(f"\n📋 Factors:")
        for f in result["meals"]["lunch"]["factors"]:
            print(f"  • {f}")

    return result


if __name__ == "__main__":
    quick_predict()
