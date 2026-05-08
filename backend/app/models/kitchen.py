"""
SmartKitchen AI X — Kitchen Model
Represents an institutional kitchen with its configuration.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Boolean, DateTime, Float, Text, ForeignKey, Enum as SAEnum
from sqlalchemy.orm import relationship
from app.database import Base


class Kitchen(Base):
    __tablename__ = "kitchens"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(200), nullable=False)
    type = Column(
        SAEnum("college", "hospital", "corporate", "hotel", "military", "catering", "other", name="kitchen_type", create_constraint=True),
        nullable=False,
    )
    location = Column(String(300), nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    capacity = Column(Integer, nullable=True)  # max meals per service
    meal_times = Column(
        Text,  # JSON string: {"breakfast": "07:30", "lunch": "12:30", "dinner": "19:30"}
        default='{"breakfast": "07:30", "lunch": "12:30", "dinner": "19:30"}',
    )
    timezone = Column(String(50), default="Asia/Kolkata")
    owner_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # ── Relationships ──
    members = relationship("KitchenMember", back_populates="kitchen", cascade="all, delete-orphan")
    inventory_items = relationship("InventoryItem", back_populates="kitchen", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="kitchen", cascade="all, delete-orphan")
    waste_logs = relationship("WasteLog", back_populates="kitchen", cascade="all, delete-orphan")
    sensors = relationship("Sensor", back_populates="kitchen", cascade="all, delete-orphan")
    meal_plans = relationship("MealPlan", back_populates="kitchen", cascade="all, delete-orphan")
    recommendations = relationship("Recommendation", back_populates="kitchen", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Kitchen {self.name} ({self.type})>"


class KitchenMember(Base):
    """Many-to-many: Users belong to Kitchens with specific roles."""
    __tablename__ = "kitchen_members"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    kitchen_id = Column(String(36), ForeignKey("kitchens.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    role = Column(
        SAEnum("admin", "manager", "staff", name="member_role", create_constraint=True),
        default="staff",
        nullable=False,
    )
    joined_at = Column(DateTime, default=datetime.utcnow)

    # ── Relationships ──
    kitchen = relationship("Kitchen", back_populates="members")
    user = relationship("User", back_populates="kitchen_memberships")

    def __repr__(self):
        return f"<KitchenMember user={self.user_id} kitchen={self.kitchen_id}>"
