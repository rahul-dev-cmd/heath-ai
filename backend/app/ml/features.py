"""
SmartKitchen AI X — Feature Engineering Pipeline
Transforms raw meal/weather data into ML-ready feature vectors.

This is the core intelligence layer — the quality of features
directly determines forecast accuracy.
"""

import pandas as pd
import numpy as np
from typing import Optional


class FeatureEngineer:
    """
    Builds rich feature vectors for demand forecasting.

    Feature categories:
    1. Time features (day, week, month, season)
    2. Lag features (consumption t-1, t-7, t-14, t-30)
    3. Rolling features (7-day mean, std, min, max)
    4. Weather features (temperature, humidity, rain)
    5. Event features (holiday, exam, vacation, weekend)
    6. Item-specific features (meal type encoding)
    """

    # All feature columns used by the model
    FEATURE_COLUMNS = [
        # Time features
        "day_of_week", "day_of_month", "month", "week_of_year",
        "is_weekend", "quarter", "season_encoded",
        "day_of_week_sin", "day_of_week_cos",
        "month_sin", "month_cos",
        # Lag features
        "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_21", "lag_30",
        # Rolling features
        "rolling_mean_3", "rolling_mean_7", "rolling_mean_14", "rolling_mean_30",
        "rolling_std_7", "rolling_std_14",
        "rolling_min_7", "rolling_max_7",
        "rolling_median_7",
        # Trend features
        "trend_7d",  # 7-day linear trend slope
        "pct_change_7d",
        # Weather features
        "temperature", "humidity", "is_rainy",
        "temp_rolling_mean_3",
        # Event features
        "is_holiday", "is_exam", "is_vacation",
        # Meal type
        "is_dinner",
    ]

    SEASON_MAP = {"winter": 0, "summer": 1, "monsoon": 2, "autumn": 3}

    def build_features_for_item(
        self,
        df: pd.DataFrame,
        item: str,
        meal_type: str,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Build features for a specific item + meal type combination.

        Args:
            df: Full dataset with all items/meals
            item: Item name (e.g. "Rice")
            meal_type: "lunch" or "dinner"

        Returns:
            X: Feature DataFrame
            y: Target Series (consumed_kg)
        """
        # Filter to specific item + meal
        mask = (df["item"] == item) & (df["meal_type"] == meal_type)
        item_df = df[mask].copy().sort_values("date").reset_index(drop=True)

        if len(item_df) < 30:
            raise ValueError(f"Need at least 30 data points for {item}/{meal_type}, got {len(item_df)}")

        # ── Time Features ──
        item_df["day_of_month"] = item_df["date"].dt.day
        item_df["quarter"] = item_df["date"].dt.quarter
        item_df["season_encoded"] = item_df["season"].map(self.SEASON_MAP).fillna(0)

        # Cyclical encoding (captures Mon→Sun→Mon continuity)
        item_df["day_of_week_sin"] = np.sin(2 * np.pi * item_df["day_of_week"] / 7)
        item_df["day_of_week_cos"] = np.cos(2 * np.pi * item_df["day_of_week"] / 7)
        item_df["month_sin"] = np.sin(2 * np.pi * item_df["month"] / 12)
        item_df["month_cos"] = np.cos(2 * np.pi * item_df["month"] / 12)

        # ── Lag Features ── (most important for time series)
        for lag in [1, 2, 3, 7, 14, 21, 30]:
            item_df[f"lag_{lag}"] = item_df["consumed_kg"].shift(lag)

        # ── Rolling Features ──
        for window in [3, 7, 14, 30]:
            item_df[f"rolling_mean_{window}"] = (
                item_df["consumed_kg"].shift(1).rolling(window, min_periods=1).mean()
            )

        for window in [7, 14]:
            item_df[f"rolling_std_{window}"] = (
                item_df["consumed_kg"].shift(1).rolling(window, min_periods=1).std()
            )

        item_df["rolling_min_7"] = item_df["consumed_kg"].shift(1).rolling(7, min_periods=1).min()
        item_df["rolling_max_7"] = item_df["consumed_kg"].shift(1).rolling(7, min_periods=1).max()
        item_df["rolling_median_7"] = item_df["consumed_kg"].shift(1).rolling(7, min_periods=1).median()

        # ── Trend Features ──
        item_df["trend_7d"] = (
            item_df["consumed_kg"].shift(1).rolling(7, min_periods=2)
            .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0)
        )

        item_df["pct_change_7d"] = item_df["rolling_mean_7"].pct_change(periods=7).fillna(0)

        # ── Weather Rolling ──
        item_df["temp_rolling_mean_3"] = item_df["temperature"].rolling(3, min_periods=1).mean()

        # ── Meal Type Feature ──
        item_df["is_dinner"] = 1 if meal_type == "dinner" else 0

        # ── Drop rows with NaN lags (first 30 days) ──
        item_df = item_df.dropna(subset=["lag_30"]).reset_index(drop=True)

        # ── Select features and target ──
        available_features = [c for c in self.FEATURE_COLUMNS if c in item_df.columns]
        X = item_df[available_features].copy()
        y = item_df["consumed_kg"].copy()

        # Fill any remaining NaN with 0
        X = X.fillna(0)

        return X, y

    def build_prediction_features(
        self,
        historical_df: pd.DataFrame,
        item: str,
        meal_type: str,
        target_date: pd.Timestamp,
        weather: Optional[dict] = None,
        is_holiday: bool = False,
        is_exam: bool = False,
        is_vacation: bool = False,
    ) -> pd.DataFrame:
        """
        Build a single feature vector for a future prediction.

        Args:
            historical_df: Past data for this item/meal
            item: Item name
            meal_type: "lunch" or "dinner"
            target_date: Date to predict for
            weather: {"temperature": 34, "humidity": 65, "is_rainy": 0}
            is_holiday, is_exam, is_vacation: Event flags

        Returns:
            Single-row DataFrame with feature columns
        """
        # Filter historical data for this item/meal
        mask = (historical_df["item"] == item) & (historical_df["meal_type"] == meal_type)
        hist = historical_df[mask].copy().sort_values("date").reset_index(drop=True)

        if len(hist) < 7:
            raise ValueError(f"Need at least 7 days of history for {item}/{meal_type}")

        consumed = hist["consumed_kg"].values

        features = {}

        # ── Time features ──
        features["day_of_week"] = target_date.weekday()
        features["day_of_month"] = target_date.day
        features["month"] = target_date.month
        features["week_of_year"] = target_date.isocalendar()[1]
        features["is_weekend"] = 1 if target_date.weekday() >= 5 else 0
        features["quarter"] = (target_date.month - 1) // 3 + 1

        season_month = target_date.month
        if season_month in [11, 12, 1, 2]:
            features["season_encoded"] = 0
        elif season_month in [3, 4, 5]:
            features["season_encoded"] = 1
        elif season_month in [6, 7, 8, 9]:
            features["season_encoded"] = 2
        else:
            features["season_encoded"] = 3

        features["day_of_week_sin"] = np.sin(2 * np.pi * features["day_of_week"] / 7)
        features["day_of_week_cos"] = np.cos(2 * np.pi * features["day_of_week"] / 7)
        features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
        features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)

        # ── Lag features ──
        for lag in [1, 2, 3, 7, 14, 21, 30]:
            idx = -lag
            features[f"lag_{lag}"] = consumed[idx] if abs(idx) <= len(consumed) else consumed[-1]

        # ── Rolling features ──
        for window in [3, 7, 14, 30]:
            w = min(window, len(consumed))
            features[f"rolling_mean_{window}"] = np.mean(consumed[-w:])

        for window in [7, 14]:
            w = min(window, len(consumed))
            features[f"rolling_std_{window}"] = np.std(consumed[-w:]) if w > 1 else 0

        w7 = min(7, len(consumed))
        features["rolling_min_7"] = np.min(consumed[-w7:])
        features["rolling_max_7"] = np.max(consumed[-w7:])
        features["rolling_median_7"] = np.median(consumed[-w7:])

        # Trend
        if len(consumed) >= 7:
            recent = consumed[-7:]
            features["trend_7d"] = np.polyfit(range(7), recent, 1)[0]
        else:
            features["trend_7d"] = 0

        mean_7 = np.mean(consumed[-7:]) if len(consumed) >= 7 else consumed[-1]
        mean_14 = np.mean(consumed[-14:]) if len(consumed) >= 14 else mean_7
        features["pct_change_7d"] = (mean_7 - mean_14) / mean_14 if mean_14 > 0 else 0

        # ── Weather features ──
        if weather:
            features["temperature"] = weather.get("temperature", 30)
            features["humidity"] = weather.get("humidity", 60)
            features["is_rainy"] = weather.get("is_rainy", 0)
        else:
            features["temperature"] = 30
            features["humidity"] = 60
            features["is_rainy"] = 0

        # Rolling temp from history
        if "temperature" in hist.columns:
            features["temp_rolling_mean_3"] = hist["temperature"].tail(3).mean()
        else:
            features["temp_rolling_mean_3"] = features["temperature"]

        # ── Event features ──
        features["is_holiday"] = 1 if is_holiday else 0
        features["is_exam"] = 1 if is_exam else 0
        features["is_vacation"] = 1 if is_vacation else 0

        # ── Meal type ──
        features["is_dinner"] = 1 if meal_type == "dinner" else 0

        # Build DataFrame with correct column order
        available = [c for c in self.FEATURE_COLUMNS if c in features]
        return pd.DataFrame([{c: features.get(c, 0) for c in available}])
