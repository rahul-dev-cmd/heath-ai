"""
SmartKitchen AI X — Waste Model
Tracks food waste events with categorized reasons and cost impact.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Date, DateTime, Text, ForeignKey, Enum as SAEnum
from sqlalchemy.orm import relationship
from app.database import Base


class WasteLog(Base):
    __tablename__ = "waste_logs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    kitchen_id = Column(String(36), ForeignKey("kitchens.id", ondelete="CASCADE"), nullable=False, index=True)
    item = Column(String(200), nullable=False)
    quantity_kg = Column(Float, nullable=False)
    meal_type = Column(
        SAEnum("breakfast", "lunch", "dinner", "snacks", name="meal_type_enum", create_constraint=True),
        nullable=True,
    )
    reason = Column(
        SAEnum(
            "overproduction", "spoilage", "plate_waste",
            "preparation_loss", "expired", "other",
            name="waste_reason", create_constraint=True,
        ),
        nullable=False,
    )
    cost_impact = Column(Float, nullable=True)  # ₹ — auto-calculated from inventory cost
    date = Column(Date, nullable=False)
    notes = Column(Text, nullable=True)
    logged_by = Column(String(36), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # ── Relationships ──
    kitchen = relationship("Kitchen", back_populates="waste_logs")
    logged_by_user = relationship("User", back_populates="waste_logs")

    def __repr__(self):
        return f"<WasteLog {self.item}: {self.quantity_kg}kg ({self.reason})>"
