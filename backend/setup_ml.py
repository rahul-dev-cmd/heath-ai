"""
SmartKitchen AI X — Quick Setup Script
Run this to generate synthetic data and train all models.

Usage:
    cd backend
    python setup_ml.py
"""

import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.ml.generate_data import save_synthetic_data
from app.ml.train import run_training_pipeline


def main():
    print("=" * 60)
    print("🍳 SmartKitchen AI X — ML Setup")
    print("=" * 60)
    print()

    # Step 1: Generate synthetic data
    print("📊 Step 1: Generating synthetic meal data...")
    print("-" * 40)
    csv_path = save_synthetic_data()
    print()

    # Step 2: Train models
    print("🤖 Step 2: Training XGBoost models...")
    print("-" * 40)
    results = run_training_pipeline(data_path=csv_path)
    print()

    # Step 3: Summary
    print("=" * 60)
    print("✅ Setup Complete!")
    print("=" * 60)

    successful = sum(1 for r in results.values() if "error" not in r)
    failed = sum(1 for r in results.values() if "error" in r)

    print(f"  Models trained: {successful}")
    print(f"  Models failed:  {failed}")
    print()
    print("  Next steps:")
    print("  1. Start PostgreSQL (or use SQLite for dev)")
    print("  2. Run: uvicorn app.main:app --reload")
    print("  3. Open: http://localhost:8000/docs")
    print()

    # Step 4: Quick prediction test
    print("🔮 Step 3: Running test prediction...")
    print("-" * 40)
    from app.ml.predict import quick_predict
    quick_predict()


if __name__ == "__main__":
    main()
