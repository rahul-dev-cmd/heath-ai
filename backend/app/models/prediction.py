"""
SmartKitchen AI X — Prediction Model
Stores AI-generated demand forecasts and actual consumption feedback.
"""

import uuid
from datetime import datetime, date
from typing import Optional
from sqlalchemy import Column, String, Float, Date, DateTime, Text, ForeignKey, UniqueConstraint, Enum as SAEnum
from sqlalchemy.orm import relationship
from app.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    kitchen_id = Column(String(36), ForeignKey("kitchens.id", ondelete="CASCADE"), nullable=False, index=True)
    date = Column(Date, nullable=False)
    meal_type = Column(
        SAEnum("breakfast", "lunch", "dinner", "snacks", name="meal_type_enum", create_constraint=True),
        nullable=False,
    )
    item = Column(String(200), nullable=False)
    predicted_quantity = Column(Float, nullable=False)  # kg
    confidence = Column(Float, nullable=True)  # 0.0 – 1.0
    lower_bound = Column(Float, nullable=True)
    upper_bound = Column(Float, nullable=True)
    actual_quantity = Column(Float, nullable=True)  # filled when actuals are logged
    model_version = Column(String(50), nullable=True)
    features_used = Column(Text, nullable=True)  # JSON string of input features
    created_at = Column(DateTime, default=datetime.utcnow)

    # ── Unique: one prediction per kitchen/date/meal/item ──
    __table_args__ = (
        UniqueConstraint("kitchen_id", "date", "meal_type", "item", name="uq_prediction_key"),
    )

    # ── Relationships ──
    kitchen = relationship("Kitchen", back_populates="predictions")

    @property
    def error(self) -> Optional[float]:
        """Absolute error between predicted and actual."""
        if self.actual_quantity is None:
            return None
        return abs(self.predicted_quantity - self.actual_quantity)

    @property
    def error_pct(self) -> Optional[float]:
        """Percentage error (MAPE-style)."""
        if self.actual_quantity is None or self.actual_quantity == 0:
            return None
        return abs(self.predicted_quantity - self.actual_quantity) / self.actual_quantity * 100

    def __repr__(self):
        return f"<Prediction {self.item} on {self.date}: {self.predicted_quantity:.1f}kg>"


class MealPlan(Base):
    """Kitchen meal plan with menu items and expected headcount."""
    __tablename__ = "meal_plans"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    kitchen_id = Column(String(36), ForeignKey("kitchens.id", ondelete="CASCADE"), nullable=False, index=True)
    date = Column(Date, nullable=False)
    meal_type = Column(
        SAEnum("breakfast", "lunch", "dinner", "snacks", name="meal_type_enum", create_constraint=True),
        nullable=False,
    )
    menu_items = Column(Text, nullable=False)  # JSON string: [{"item": "Rice", "planned_kg": 50}, ...]
    expected_headcount = Column(Float, nullable=True)
    actual_headcount = Column(Float, nullable=True)
    notes = Column(String(500), nullable=True)
    created_by = Column(String(36), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("kitchen_id", "date", "meal_type", name="uq_meal_plan_key"),
    )

    # ── Relationships ──
    kitchen = relationship("Kitchen", back_populates="meal_plans")

    def __repr__(self):
        return f"<MealPlan {self.date} {self.meal_type}>"
