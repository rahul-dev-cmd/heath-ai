"""
SmartKitchen AI X — Model Evaluation
Comprehensive evaluation metrics and visualization for forecast models.
"""

import pandas as pd
import numpy as np
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)
import joblib
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates trained models against actual consumption data.
    
    Generates:
    - Per-item accuracy metrics (MAPE, RMSE, MAE, R²)
    - Overall kitchen accuracy
    - Prediction vs actual comparison data for charts
    - Worst-performing items identification
    - Accuracy trend over time
    """

    def __init__(self, model_dir: str = None):
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).parent / "models"

    def evaluate_predictions(
        self,
        predictions_df: pd.DataFrame,
    ) -> dict:
        """
        Evaluate predictions against actuals.

        Args:
            predictions_df: DataFrame with columns:
                date, item, meal_type, predicted_kg, actual_kg

        Returns:
            Comprehensive evaluation metrics
        """
        # Filter out rows where actual is missing
        df = predictions_df.dropna(subset=["actual_kg"]).copy()
        df = df[df["actual_kg"] > 0]

        if len(df) == 0:
            return {"error": "No actual data available for evaluation"}

        # ── Overall metrics ──
        overall = self._calc_metrics(df["actual_kg"], df["predicted_kg"])

        # ── Per-item metrics ──
        per_item = []
        for item in df["item"].unique():
            item_df = df[df["item"] == item]
            metrics = self._calc_metrics(item_df["actual_kg"], item_df["predicted_kg"])
            metrics["item"] = item
            metrics["n_predictions"] = len(item_df)
            per_item.append(metrics)

        per_item.sort(key=lambda x: x["mape"])

        # ── Per-meal metrics ──
        per_meal = []
        for meal in df["meal_type"].unique():
            meal_df = df[df["meal_type"] == meal]
            metrics = self._calc_metrics(meal_df["actual_kg"], meal_df["predicted_kg"])
            metrics["meal_type"] = meal
            per_meal.append(metrics)

        # ── Daily accuracy trend ──
        trend = []
        for d in sorted(df["date"].unique()):
            day_df = df[df["date"] == d]
            day_metrics = self._calc_metrics(day_df["actual_kg"], day_df["predicted_kg"])
            trend.append({
                "date": str(d),
                "mape": day_metrics["mape"],
                "rmse": day_metrics["rmse"],
                "n_items": len(day_df),
            })

        # ── Worst performers (highest MAPE) ──
        worst = sorted(per_item, key=lambda x: x["mape"], reverse=True)[:3]

        # ── Prediction vs actual pairs (for chart) ──
        comparison = df[["date", "item", "meal_type", "predicted_kg", "actual_kg"]].to_dict(orient="records")

        return {
            "overall": overall,
            "per_item": per_item,
            "per_meal": per_meal,
            "trend": trend,
            "worst_performers": worst,
            "comparison": comparison[:100],  # Limit for API response
            "total_predictions": len(df),
            "date_range": {
                "start": str(df["date"].min()),
                "end": str(df["date"].max()),
            },
        }

    def evaluate_model_file(
        self,
        model_name: str,
        test_data: pd.DataFrame,
    ) -> dict:
        """Evaluate a specific model file against test data."""
        from app.ml.features import FeatureEngineer

        model_path = self.model_dir / f"{model_name}.joblib"
        artifact = joblib.load(model_path)

        model = artifact["model"]
        item = artifact["item"]
        meal_type = artifact["meal_type"]
        feature_columns = artifact["feature_columns"]

        fe = FeatureEngineer()
        X, y = fe.build_features_for_item(test_data, item, meal_type)

        # Align columns
        for col in feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_columns]

        preds = model.predict(X)
        preds = np.maximum(preds, 0)

        metrics = self._calc_metrics(y, preds)
        metrics["model_name"] = model_name
        metrics["model_version"] = artifact.get("version", "unknown")
        metrics["training_mape"] = artifact["metrics"].get("cv_mape", None)

        return metrics

    def _calc_metrics(self, actual: pd.Series, predicted: pd.Series) -> dict:
        """Calculate regression metrics."""
        actual = np.array(actual)
        predicted = np.array(predicted)

        mask = actual > 0
        mape = mean_absolute_percentage_error(actual[mask], predicted[mask]) * 100 if mask.sum() > 0 else 0

        return {
            "mape": round(mape, 2),
            "rmse": round(np.sqrt(mean_squared_error(actual, predicted)), 2),
            "mae": round(mean_absolute_error(actual, predicted), 2),
            "r2": round(r2_score(actual, predicted), 3),
            "mean_error": round(np.mean(predicted - actual), 2),  # positive = overprediction
            "median_abs_error": round(np.median(np.abs(predicted - actual)), 2),
        }

    def generate_report(self, evaluation: dict) -> str:
        """Generate a text report from evaluation results."""
        lines = []
        lines.append("=" * 60)
        lines.append("📊 SmartKitchen AI X — Model Evaluation Report")
        lines.append("=" * 60)
        lines.append("")

        o = evaluation["overall"]
        lines.append(f"Overall MAPE:  {o['mape']:.1f}%")
        lines.append(f"Overall RMSE:  {o['rmse']:.2f} kg")
        lines.append(f"Overall MAE:   {o['mae']:.2f} kg")
        lines.append(f"Overall R²:    {o['r2']:.3f}")
        lines.append(f"Mean Error:    {o['mean_error']:+.2f} kg ({'over' if o['mean_error'] > 0 else 'under'}prediction)")
        lines.append(f"Total Predictions: {evaluation['total_predictions']}")
        lines.append("")

        lines.append("── Per Item ──")
        for item in evaluation["per_item"]:
            lines.append(
                f"  {item['item']:12s}: MAPE={item['mape']:5.1f}%, "
                f"RMSE={item['rmse']:5.2f}, R²={item['r2']:.3f} "
                f"({item['n_predictions']} predictions)"
            )
        lines.append("")

        if evaluation["worst_performers"]:
            lines.append("── Worst Performers (need attention) ──")
            for item in evaluation["worst_performers"]:
                lines.append(f"  ⚠️  {item['item']}: MAPE={item['mape']:.1f}%")

        return "\n".join(lines)
