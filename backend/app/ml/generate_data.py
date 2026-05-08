"""
SmartKitchen AI X — Synthetic Data Generator
Generates realistic institutional kitchen meal data for model training.

This simulates 6 months of daily meal data for a college hostel,
including weather effects, weekend patterns, holidays, and seasonal trends.
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path
import json


# ── Indian holiday calendar (simplified) ──
HOLIDAYS_2025_2026 = [
    date(2025, 1, 26),   # Republic Day
    date(2025, 3, 14),   # Holi
    date(2025, 4, 14),   # Ambedkar Jayanti
    date(2025, 5, 1),    # May Day
    date(2025, 8, 15),   # Independence Day
    date(2025, 10, 2),   # Gandhi Jayanti
    date(2025, 10, 20),  # Dussehra
    date(2025, 11, 1),   # Diwali
    date(2025, 12, 25),  # Christmas
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 4),    # Holi
    date(2026, 4, 14),   # Ambedkar Jayanti
    date(2026, 5, 1),    # May Day
]

# ── Exam periods (students stay on campus, higher attendance) ──
EXAM_PERIODS = [
    (date(2025, 4, 20), date(2025, 5, 10)),   # End-sem 1
    (date(2025, 9, 15), date(2025, 9, 25)),    # Mid-sem 2
    (date(2025, 11, 15), date(2025, 12, 5)),   # End-sem 2
    (date(2026, 4, 20), date(2026, 5, 10)),    # End-sem 3
]

# ── Vacation periods (very low attendance) ──
VACATION_PERIODS = [
    (date(2025, 5, 15), date(2025, 7, 15)),    # Summer vacation
    (date(2025, 12, 20), date(2026, 1, 5)),     # Winter break
]

# ── Menu items with base consumption patterns ──
MENU_ITEMS = {
    "Rice": {
        "base_kg": 75,
        "std_dev": 8,
        "weekend_factor": 0.68,
        "summer_factor": 0.85,    # less heavy food in summer
        "monsoon_factor": 1.10,   # students stay in, eat more
        "dinner_factor": 0.75,
        "cost_per_kg": 45,
    },
    "Dal": {
        "base_kg": 28,
        "std_dev": 4,
        "weekend_factor": 0.72,
        "summer_factor": 0.90,
        "monsoon_factor": 1.05,
        "dinner_factor": 0.80,
        "cost_per_kg": 90,
    },
    "Roti": {
        "base_kg": 22,
        "std_dev": 3.5,
        "weekend_factor": 0.65,
        "summer_factor": 0.82,
        "monsoon_factor": 1.08,
        "dinner_factor": 1.10,   # more roti at dinner
        "cost_per_kg": 35,
    },
    "Sabzi": {
        "base_kg": 32,
        "std_dev": 5,
        "weekend_factor": 0.70,
        "summer_factor": 0.92,
        "monsoon_factor": 1.03,
        "dinner_factor": 0.85,
        "cost_per_kg": 60,
    },
    "Salad": {
        "base_kg": 12,
        "std_dev": 2.5,
        "weekend_factor": 0.75,
        "summer_factor": 1.20,   # more salad in summer
        "monsoon_factor": 0.90,
        "dinner_factor": 0.70,
        "cost_per_kg": 40,
    },
    "Curd": {
        "base_kg": 18,
        "std_dev": 2.8,
        "weekend_factor": 0.78,
        "summer_factor": 1.25,   # much more curd in summer
        "monsoon_factor": 0.95,
        "dinner_factor": 0.60,
        "cost_per_kg": 55,
    },
}


def get_season(month: int) -> str:
    """Return Indian season name."""
    if month in [11, 12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "summer"
    elif month in [6, 7, 8, 9]:
        return "monsoon"
    else:
        return "autumn"


def get_temperature(d: date) -> float:
    """Simulate realistic temperature for Delhi NCR region."""
    # Base temp follows a sinusoidal pattern across the year
    day_of_year = d.timetuple().tm_yday
    base_temp = 25 + 12 * np.sin(2 * np.pi * (day_of_year - 60) / 365)
    noise = np.random.normal(0, 2.5)
    return round(base_temp + noise, 1)


def get_humidity(d: date) -> float:
    """Simulate realistic humidity."""
    month = d.month
    if month in [7, 8, 9]:      # monsoon
        base = 80
    elif month in [11, 12, 1]:   # winter
        base = 45
    else:                         # summer
        base = 35
    return round(base + np.random.normal(0, 8), 1)


def is_rainy(d: date) -> bool:
    """Simulate rain probability based on season."""
    month = d.month
    if month in [7, 8, 9]:
        return np.random.random() < 0.55
    elif month in [6, 10]:
        return np.random.random() < 0.25
    else:
        return np.random.random() < 0.05


def is_holiday(d: date) -> bool:
    return d in HOLIDAYS_2025_2026


def is_exam_period(d: date) -> bool:
    return any(start <= d <= end for start, end in EXAM_PERIODS)


def is_vacation(d: date) -> bool:
    return any(start <= d <= end for start, end in VACATION_PERIODS)


def generate_synthetic_data(
    start_date: date = date(2025, 8, 1),
    days: int = 270,
    kitchen_capacity: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic synthetic meal data for a college hostel.

    Returns DataFrame with columns:
        date, meal_type, item, prepared_kg, consumed_kg, wasted_kg,
        waste_pct, day_of_week, is_weekend, month, season,
        temperature, humidity, is_rainy, is_holiday, is_exam, is_vacation
    """
    np.random.seed(seed)
    records = []

    for day_offset in range(days):
        d = start_date + timedelta(days=day_offset)
        temp = get_temperature(d)
        humidity = get_humidity(d)
        rainy = is_rainy(d)
        holiday = is_holiday(d)
        exam = is_exam_period(d)
        vacation = is_vacation(d)
        season = get_season(d.month)

        for meal_type in ["lunch", "dinner"]:
            for item_name, params in MENU_ITEMS.items():
                base = params["base_kg"]

                # ── Meal type adjustment ──
                if meal_type == "dinner":
                    base *= params["dinner_factor"]

                # ── Weekend effect (students leave campus) ──
                if d.weekday() >= 5:
                    base *= params["weekend_factor"]

                # ── Season effects ──
                if season == "summer":
                    base *= params["summer_factor"]
                elif season == "monsoon":
                    base *= params["monsoon_factor"]

                # ── Holiday effect (most students leave) ──
                if holiday:
                    base *= np.random.uniform(0.35, 0.55)

                # ── Vacation effect (very few students) ──
                if vacation:
                    base *= np.random.uniform(0.15, 0.30)

                # ── Exam effect (more students stay on campus) ──
                if exam:
                    base *= np.random.uniform(1.05, 1.15)

                # ── Weather effects ──
                if temp > 38:
                    base *= 0.88   # very hot → less appetite
                elif temp > 35:
                    base *= 0.93
                elif temp < 10:
                    base *= 1.08   # cold → more appetite

                if rainy:
                    base *= 1.05   # rainy → students stay, eat more

                # ── Random daily noise ──
                consumed = max(1, base + np.random.normal(0, params["std_dev"]))

                # ── Overproduction simulation (5-30% over actual need) ──
                overproduction_pct = np.random.uniform(0.05, 0.30)
                prepared = consumed * (1 + overproduction_pct)
                wasted = prepared - consumed

                # ── Occasional spoilage events ──
                if np.random.random() < 0.03:  # 3% chance of spoilage
                    spoilage = np.random.uniform(1, 5)
                    wasted += spoilage
                    prepared += spoilage

                records.append({
                    "date": d,
                    "meal_type": meal_type,
                    "item": item_name,
                    "prepared_kg": round(prepared, 1),
                    "consumed_kg": round(consumed, 1),
                    "wasted_kg": round(max(0, wasted), 1),
                    "waste_pct": round(max(0, wasted) / prepared * 100, 1) if prepared > 0 else 0,
                    "cost_per_kg": params["cost_per_kg"],
                    "cost_wasted": round(max(0, wasted) * params["cost_per_kg"], 1),
                    "day_of_week": d.weekday(),
                    "day_name": d.strftime("%A"),
                    "is_weekend": 1 if d.weekday() >= 5 else 0,
                    "month": d.month,
                    "week_of_year": d.isocalendar()[1],
                    "season": season,
                    "temperature": temp,
                    "humidity": humidity,
                    "is_rainy": 1 if rainy else 0,
                    "is_holiday": 1 if holiday else 0,
                    "is_exam": 1 if exam else 0,
                    "is_vacation": 1 if vacation else 0,
                })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


def save_synthetic_data(output_dir: str = None) -> str:
    """Generate and save synthetic data to CSV."""
    if output_dir is None:
        output_dir = str(Path(__file__).parent / "data")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_data()

    csv_path = str(Path(output_dir) / "synthetic_meal_data.csv")
    df.to_csv(csv_path, index=False)

    # Also save summary stats
    stats = {
        "total_records": len(df),
        "date_range": f"{df['date'].min().date()} to {df['date'].max().date()}",
        "total_days": df["date"].dt.date.nunique(),
        "items": list(df["item"].unique()),
        "meal_types": list(df["meal_type"].unique()),
        "total_consumed_kg": round(df["consumed_kg"].sum(), 1),
        "total_wasted_kg": round(df["wasted_kg"].sum(), 1),
        "avg_waste_pct": round(df["waste_pct"].mean(), 1),
        "total_cost_wasted": round(df["cost_wasted"].sum(), 1),
    }

    stats_path = str(Path(output_dir) / "data_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    print(f"✅ Generated {len(df)} records spanning {stats['total_days']} days")
    print(f"📁 Saved to: {csv_path}")
    print(f"📊 Avg waste: {stats['avg_waste_pct']}%")
    print(f"💰 Total cost wasted: ₹{stats['total_cost_wasted']:,.0f}")

    return csv_path


if __name__ == "__main__":
    save_synthetic_data()
