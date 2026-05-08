"""
SmartKitchen AI X — Recommendation & Audit Models
AI-generated recommendations and user activity audit trail.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Boolean, DateTime, Text, ForeignKey, Enum as SAEnum
from sqlalchemy.orm import relationship
from app.database import Base


class Recommendation(Base):
    __tablename__ = "recommendations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    kitchen_id = Column(String(36), ForeignKey("kitchens.id", ondelete="CASCADE"), nullable=False, index=True)
    type = Column(
        SAEnum("production", "inventory", "menu", "sustainability", "cost", name="recommendation_type", create_constraint=True),
        nullable=False,
    )
    priority = Column(
        SAEnum("critical", "high", "medium", "low", name="recommendation_priority", create_constraint=True),
        nullable=False,
    )
    title = Column(String(300), nullable=False)
    description = Column(Text, nullable=True)
    estimated_savings = Column(Float, nullable=True)  # ₹
    is_read = Column(Boolean, default=False)
    is_acted_on = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)

    # ── Relationships ──
    kitchen = relationship("Kitchen", back_populates="recommendations")

    def __repr__(self):
        return f"<Recommendation [{self.priority}] {self.title}>"


class AuditLog(Base):
    """Immutable audit trail of all user actions."""
    __tablename__ = "audit_logs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    kitchen_id = Column(String(36), ForeignKey("kitchens.id"), nullable=True)
    action = Column(String(50), nullable=False)  # e.g. "create_inventory", "log_waste"
    entity_type = Column(String(50), nullable=True)  # e.g. "inventory_item", "waste_log"
    entity_id = Column(String(36), nullable=True)
    changes = Column(Text, nullable=True)  # JSON string: {"field": {"old": x, "new": y}}
    ip_address = Column(String(45), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # ── Relationships ──
    user = relationship("User", back_populates="audit_logs")

    def __repr__(self):
        return f"<AuditLog {self.action} by {self.user_id}>"
