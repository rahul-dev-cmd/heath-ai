"""
SmartKitchen AI X — Model Training Pipeline
Trains XGBoost models for food demand forecasting using time-series cross-validation.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
def timeseries_split(n_samples, n_splits):
    fold_size = n_samples // (n_splits + 1)
    for i in range(1, n_splits + 1):
        train_end = i * fold_size
        test_end = train_end + fold_size
        if i == n_splits:
            test_end = n_samples
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, test_end)
        yield train_idx, val_idx

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
from datetime import datetime
from typing import Optional, List
import json
import logging

from app.ml.features import FeatureEngineer

logger = logging.getLogger(__name__)


class ForecastTrainer:
    """
    Trains XGBoost models for food demand forecasting.

    One model per (kitchen, item, meal_type) combination, or a shared
    model across items if data is insufficient for per-item models.
    """

    DEFAULT_PARAMS = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }

    def __init__(self, model_dir: str = None):
        self.feature_engineer = FeatureEngineer()
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).parent / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def train_for_item(
        self,
        df: pd.DataFrame,
        item: str,
        meal_type: str,
        kitchen_id: str = "default",
        params: dict = None,
    ) -> dict:
        """
        Train an XGBoost model for a specific item + meal combination.

        Args:
            df: Full dataset
            item: Item name (e.g. "Rice")
            meal_type: "lunch" or "dinner"
            kitchen_id: Kitchen identifier for model naming
            params: XGBoost hyperparameters (overrides defaults)

        Returns:
            Dict with metrics, feature importance, and model path
        """
        logger.info(f"Training model for {item}/{meal_type} (kitchen: {kitchen_id})")

        # Build features
        X, y = self.feature_engineer.build_features_for_item(df, item, meal_type)
        logger.info(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")

        # Use provided params or defaults
        model_params = {**self.DEFAULT_PARAMS, **(params or {})}

        # ── Time-series cross-validation ──
        n_splits = min(5, len(X) // 20)  # at least 20 samples per fold
        n_splits = max(2, n_splits)
        n_splits = max(2, n_splits)

        cv_metrics = {
            "mape": [],
            "rmse": [],
            "mae": [],
            "r2": [],
        }

        for fold, (train_idx, val_idx) in enumerate(timeseries_split(len(X), n_splits)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = xgb.XGBRegressor(**model_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            preds = model.predict(X_val)
            preds = np.maximum(preds, 0)  # Never predict negative

            # Calculate metrics (avoid division by zero)
            mask = y_val > 0
            if mask.sum() > 0:
                cv_metrics["mape"].append(
                    mean_absolute_percentage_error(y_val[mask], preds[mask]) * 100
                )
            cv_metrics["rmse"].append(np.sqrt(mean_squared_error(y_val, preds)))
            cv_metrics["mae"].append(mean_absolute_error(y_val, preds))
            cv_metrics["r2"].append(r2_score(y_val, preds))

            logger.info(
                f"  Fold {fold + 1}: MAPE={cv_metrics['mape'][-1]:.1f}%, "
                f"RMSE={cv_metrics['rmse'][-1]:.2f}"
            )

        # ── Train final model on ALL data ──
        final_model = xgb.XGBRegressor(**model_params)
        final_model.fit(X, y, verbose=False)

        # ── Feature importance ──
        importance = dict(zip(X.columns, final_model.feature_importances_))
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        # ── Calculate prediction confidence (based on CV residuals) ──
        # Re-predict on validation set of last fold for confidence bounds
        last_train_idx, last_val_idx = list(timeseries_split(len(X), n_splits))[-1]
        last_preds = final_model.predict(X.iloc[last_val_idx])
        residuals = y.iloc[last_val_idx].values - last_preds
        residual_std = np.std(residuals)

        # ── Save model ──
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        item_slug = item.lower().replace(" ", "_")
        model_name = f"{kitchen_id}_{item_slug}_{meal_type}"
        model_path = self.model_dir / f"{model_name}.joblib"

        model_artifact = {
            "model": final_model,
            "feature_columns": X.columns.tolist(),
            "item": item,
            "meal_type": meal_type,
            "kitchen_id": kitchen_id,
            "metrics": {
                "cv_mape": round(np.mean(cv_metrics["mape"]), 2),
                "cv_rmse": round(np.mean(cv_metrics["rmse"]), 2),
                "cv_mae": round(np.mean(cv_metrics["mae"]), 2),
                "cv_r2": round(np.mean(cv_metrics["r2"]), 3),
                "residual_std": round(residual_std, 2),
            },
            "feature_importance": {k: round(v, 4) for k, v in list(importance.items())[:15]},
            "n_samples": len(X),
            "n_features": X.shape[1],
            "trained_at": timestamp,
            "model_params": model_params,
            "version": f"v1_{timestamp}",
        }

        joblib.dump(model_artifact, model_path)
        logger.info(f"✅ Model saved: {model_path}")

        # ── Return results ──
        return {
            "model_path": str(model_path),
            "model_name": model_name,
            "metrics": model_artifact["metrics"],
            "feature_importance": model_artifact["feature_importance"],
            "n_samples": len(X),
            "version": model_artifact["version"],
        }

    def train_all_items(
        self,
        df: pd.DataFrame,
        kitchen_id: str = "default",
        items: List[str] = None,
        meal_types: List[str] = None,
    ) -> dict:
        """
        Train models for all item + meal_type combinations.

        Returns:
            Dict of {model_name: training_results}
        """
        if items is None:
            items = df["item"].unique().tolist()
        if meal_types is None:
            meal_types = df["meal_type"].unique().tolist()

        results = {}
        total = len(items) * len(meal_types)
        count = 0

        for item in items:
            for meal_type in meal_types:
                count += 1
                logger.info(f"\n{'='*60}")
                logger.info(f"Training {count}/{total}: {item} / {meal_type}")
                logger.info(f"{'='*60}")

                try:
                    result = self.train_for_item(df, item, meal_type, kitchen_id)
                    results[result["model_name"]] = result
                    print(
                        f"  ✅ {item}/{meal_type}: "
                        f"MAPE={result['metrics']['cv_mape']:.1f}%, "
                        f"RMSE={result['metrics']['cv_rmse']:.2f}, "
                        f"R²={result['metrics']['cv_r2']:.3f}"
                    )
                except Exception as e:
                    logger.error(f"  ❌ Failed: {item}/{meal_type}: {e}")
                    results[f"{kitchen_id}_{item}_{meal_type}"] = {"error": str(e)}

        # ── Save summary ──
        summary_path = self.model_dir / f"{kitchen_id}_training_summary.json"
        summary = {
            "kitchen_id": kitchen_id,
            "trained_at": datetime.utcnow().isoformat(),
            "total_models": sum(1 for r in results.values() if "error" not in r),
            "failed": sum(1 for r in results.values() if "error" in r),
            "models": {
                k: {key: val for key, val in v.items() if key != "feature_importance"}
                for k, v in results.items()
            },
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        avg_mape = np.mean([
            r["metrics"]["cv_mape"]
            for r in results.values()
            if "metrics" in r
        ])
        print(f"\n{'='*60}")
        print(f"📊 Training Complete: {summary['total_models']} models, avg MAPE: {avg_mape:.1f}%")
        print(f"📁 Models saved to: {self.model_dir}")
        print(f"{'='*60}")

        return results


def run_training_pipeline(data_path: str = None, kitchen_id: str = "default"):
    """
    Full training pipeline: load data → engineer features → train → save.

    Can be called from CLI or Celery task.
    """
    logging.basicConfig(level=logging.INFO)

    # Load data
    if data_path is None:
        data_path = str(Path(__file__).parent / "data" / "synthetic_meal_data.csv")

    print(f"📂 Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    print(f"📊 Loaded {len(df)} records, {df['date'].dt.date.nunique()} days")

    # Train all models
    trainer = ForecastTrainer()
    results = trainer.train_all_items(df, kitchen_id=kitchen_id)

    return results


if __name__ == "__main__":
    run_training_pipeline()
