"""
SmartKitchen AI X — Sensor Model
IoT sensor registration and time-series readings.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, ForeignKey, Enum as SAEnum
from sqlalchemy.orm import relationship
from app.database import Base


class Sensor(Base):
    __tablename__ = "sensors"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    kitchen_id = Column(String(36), ForeignKey("kitchens.id", ondelete="CASCADE"), nullable=False, index=True)
    type = Column(
        SAEnum("weight", "temperature", "humidity", name="sensor_type", create_constraint=True),
        nullable=False,
    )
    location_label = Column(String(200), nullable=True)  # e.g. "Rice Storage Bin"
    mac_address = Column(String(17), nullable=True)
    device_id = Column(String(50), unique=True, nullable=False)
    firmware_version = Column(String(20), nullable=True)
    battery_level = Column(Integer, nullable=True)
    is_active = Column(Boolean, default=True)
    last_reading_at = Column(DateTime, nullable=True)
    calibration_factor = Column(Float, nullable=True)
    calibration_offset = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    # ── Relationships ──
    kitchen = relationship("Kitchen", back_populates="sensors")
    readings = relationship("SensorReading", back_populates="sensor", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Sensor {self.device_id} ({self.type}) at {self.location_label}>"


class SensorReading(Base):
    """Time-series sensor data. In production, use TimescaleDB hypertable."""
    __tablename__ = "sensor_readings"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    sensor_id = Column(String(36), ForeignKey("sensors.id", ondelete="CASCADE"), nullable=False, index=True)
    time = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    value = Column(Float, nullable=False)
    unit = Column(String(10), nullable=True)
    battery = Column(Integer, nullable=True)

    # ── Relationships ──
    sensor = relationship("Sensor", back_populates="readings")

    def __repr__(self):
        return f"<SensorReading {self.value} at {self.time}>"
