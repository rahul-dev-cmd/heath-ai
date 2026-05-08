"""
SmartKitchen AI X — Pydantic Schemas
Request/response validation models for all API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import date, datetime
from uuid import UUID
from enum import Enum


# ═══════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════

class UserRole(str, Enum):
    super_admin = "super_admin"
    admin = "admin"
    manager = "manager"
    staff = "staff"


class KitchenType(str, Enum):
    college = "college"
    hospital = "hospital"
    corporate = "corporate"
    hotel = "hotel"
    military = "military"
    catering = "catering"
    other = "other"


class ItemCategory(str, Enum):
    grains = "grains"
    pulses = "pulses"
    vegetables = "vegetables"
    dairy = "dairy"
    spices = "spices"
    oils = "oils"
    meat = "meat"
    fruits = "fruits"
    beverages = "beverages"
    other = "other"


class MealType(str, Enum):
    breakfast = "breakfast"
    lunch = "lunch"
    dinner = "dinner"
    snacks = "snacks"


class WasteReason(str, Enum):
    overproduction = "overproduction"
    spoilage = "spoilage"
    plate_waste = "plate_waste"
    preparation_loss = "preparation_loss"
    expired = "expired"
    other = "other"


class StockAction(str, Enum):
    restock = "restock"
    consume = "consume"
    waste = "waste"
    adjust = "adjust"
    iot_update = "iot_update"


# ═══════════════════════════════════════
# AUTH SCHEMAS
# ═══════════════════════════════════════

class UserRegister(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: str = Field(..., max_length=255)
    password: str = Field(..., min_length=8, max_length=128)
    phone: Optional[str] = None
    kitchen_name: Optional[str] = None
    kitchen_type: Optional[KitchenType] = None


class UserLogin(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    phone: Optional[str] = None
    role: UserRole
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ═══════════════════════════════════════
# KITCHEN SCHEMAS
# ═══════════════════════════════════════

class KitchenCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=200)
    type: KitchenType
    location: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    capacity: Optional[int] = Field(None, ge=1)
    meal_times: Optional[dict] = None


class KitchenUpdate(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    capacity: Optional[int] = None
    meal_times: Optional[dict] = None


class KitchenResponse(BaseModel):
    id: str
    name: str
    type: str
    location: Optional[str] = None
    capacity: Optional[int] = None
    meal_times: Optional[Any] = None
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ═══════════════════════════════════════
# INVENTORY SCHEMAS
# ═══════════════════════════════════════

class InventoryItemCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    category: ItemCategory
    quantity: float = Field(..., ge=0)
    unit: str = Field(default="kg", max_length=20)
    min_threshold: Optional[float] = Field(None, ge=0)
    max_capacity: Optional[float] = Field(None, ge=0)
    cost_per_unit: Optional[float] = Field(None, ge=0)
    supplier: Optional[str] = None
    shelf_life_days: Optional[int] = None


class InventoryItemUpdate(BaseModel):
    name: Optional[str] = None
    quantity: Optional[float] = Field(None, ge=0)
    min_threshold: Optional[float] = None
    cost_per_unit: Optional[float] = None
    supplier: Optional[str] = None


class InventoryItemResponse(BaseModel):
    id: str
    kitchen_id: str
    name: str
    category: ItemCategory
    quantity: float
    unit: str
    min_threshold: Optional[float] = None
    max_capacity: Optional[float] = None
    cost_per_unit: Optional[float] = None
    supplier: Optional[str] = None
    stock_status: str
    last_restocked: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


class RestockRequest(BaseModel):
    quantity_added: float = Field(..., gt=0)
    cost: Optional[float] = None
    note: Optional[str] = None


# ═══════════════════════════════════════
# WASTE SCHEMAS
# ═══════════════════════════════════════

class WasteLogCreate(BaseModel):
    item: str = Field(..., min_length=1, max_length=200)
    quantity_kg: float = Field(..., gt=0)
    meal_type: Optional[MealType] = None
    reason: WasteReason
    date: date
    notes: Optional[str] = None


class WasteLogResponse(BaseModel):
    id: str
    kitchen_id: str
    item: str
    quantity_kg: float
    meal_type: Optional[MealType] = None
    reason: WasteReason
    cost_impact: Optional[float] = None
    date: date
    notes: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class WasteAnalytics(BaseModel):
    total_waste_kg: float
    waste_percentage: float
    cost_impact: float
    by_category: List[dict]
    by_reason: List[dict]
    by_meal: List[dict]
    trend: List[dict]
    period_days: int


# ═══════════════════════════════════════
# PREDICTION SCHEMAS
# ═══════════════════════════════════════

class ForecastRequest(BaseModel):
    date: date
    meal_type: Optional[MealType] = None


class PredictionItem(BaseModel):
    item: str
    predicted_kg: float
    confidence: float
    lower_bound: float
    upper_bound: float


class ForecastResponse(BaseModel):
    kitchen_id: str
    date: date
    meal_type: MealType
    predictions: List[PredictionItem]
    weather: Optional[dict] = None
    factors: List[str] = []
    model_version: str


class FeedbackRequest(BaseModel):
    date: date
    meal_type: MealType
    item: str
    actual_kg: float = Field(..., ge=0)


class AccuracyResponse(BaseModel):
    overall_mape: float
    overall_rmse: float
    per_item: List[dict]
    period_days: int
    total_predictions: int


# ═══════════════════════════════════════
# SENSOR SCHEMAS
# ═══════════════════════════════════════

class SensorCreate(BaseModel):
    type: str  # weight, temperature, humidity
    location_label: Optional[str] = None
    device_id: str = Field(..., min_length=1)


class SensorResponse(BaseModel):
    id: str
    kitchen_id: str
    type: str
    location_label: Optional[str] = None
    device_id: str
    battery_level: Optional[int] = None
    is_active: bool
    last_reading_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class SensorReadingCreate(BaseModel):
    sensor_id: str
    kitchen_id: str
    type: str
    value: float
    unit: str
    timestamp: datetime
    battery: Optional[int] = None


# ═══════════════════════════════════════
# RECOMMENDATION SCHEMAS
# ═══════════════════════════════════════

class RecommendationResponse(BaseModel):
    id: str
    type: str
    priority: str
    title: str
    description: Optional[str] = None
    estimated_savings: Optional[float] = None
    is_read: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ═══════════════════════════════════════
# DASHBOARD / ANALYTICS SCHEMAS
# ═══════════════════════════════════════

class DashboardStats(BaseModel):
    total_meals_today: int
    waste_percentage: float
    inventory_health: float  # 0-100%
    monthly_savings: float  # ₹
    active_sensors: int
    total_sensors: int
    low_stock_items: int
    predictions_accuracy: float
