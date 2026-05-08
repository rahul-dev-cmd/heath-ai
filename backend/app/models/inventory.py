"""
SmartKitchen AI X — Inventory Models
Tracks food items, stock levels, and stock change audit trail.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, Text, ForeignKey, Enum as SAEnum
from sqlalchemy.orm import relationship
from app.database import Base


class InventoryItem(Base):
    __tablename__ = "inventory_items"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    kitchen_id = Column(String(36), ForeignKey("kitchens.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(200), nullable=False)
    category = Column(
        SAEnum(
            "grains", "pulses", "vegetables", "dairy", "spices",
            "oils", "meat", "fruits", "beverages", "other",
            name="item_category", create_constraint=True,
        ),
        nullable=False,
    )
    quantity = Column(Float, nullable=False, default=0.0)
    unit = Column(String(20), nullable=False, default="kg")
    min_threshold = Column(Float, nullable=True)
    max_capacity = Column(Float, nullable=True)
    cost_per_unit = Column(Float, nullable=True)  # ₹ per kg/liter/piece
    supplier = Column(String(200), nullable=True)
    shelf_life_days = Column(Integer, nullable=True)
    sensor_id = Column(String(36), ForeignKey("sensors.id"), nullable=True)
    last_restocked = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # ── Relationships ──
    kitchen = relationship("Kitchen", back_populates="inventory_items")
    stock_logs = relationship("StockLog", back_populates="item", cascade="all, delete-orphan")

    @property
    def is_low_stock(self) -> bool:
        if self.min_threshold is None:
            return False
        return self.quantity <= self.min_threshold

    @property
    def stock_status(self) -> str:
        if self.min_threshold is None:
            return "unknown"
        if self.quantity <= 0:
            return "out_of_stock"
        if self.quantity <= self.min_threshold:
            return "low"
        if self.max_capacity and self.quantity >= self.max_capacity * 0.9:
            return "overstocked"
        return "normal"

    def __repr__(self):
        return f"<InventoryItem {self.name}: {self.quantity} {self.unit}>"


class StockLog(Base):
    """Audit trail for every inventory change."""
    __tablename__ = "stock_logs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    item_id = Column(String(36), ForeignKey("inventory_items.id", ondelete="CASCADE"), nullable=False, index=True)
    action = Column(
        SAEnum("restock", "consume", "waste", "adjust", "iot_update", name="stock_action", create_constraint=True),
        nullable=False,
    )
    quantity_change = Column(Float, nullable=False)  # positive = add, negative = remove
    new_quantity = Column(Float, nullable=False)
    cost = Column(Float, nullable=True)
    note = Column(Text, nullable=True)
    performed_by = Column(String(36), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # ── Relationships ──
    item = relationship("InventoryItem", back_populates="stock_logs")

    def __repr__(self):
        return f"<StockLog {self.action}: {self.quantity_change:+.1f}>"
