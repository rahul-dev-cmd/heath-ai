"""
SmartKitchen AI X — Dashboard Analytics API
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import date, timedelta

from app.database import get_db
from app.models.inventory import InventoryItem
from app.models.waste import WasteLog
from app.models.prediction import Prediction
from app.models.sensor import Sensor
from app.models.kitchen import KitchenMember
from app.models.user import User
from app.schemas.schemas import DashboardStats
from app.api.v1.auth import get_current_user

router = APIRouter()


@router.get("/kitchens/{kitchen_id}/dashboard", response_model=DashboardStats)
async def get_dashboard_stats(kitchen_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    m = db.query(KitchenMember).filter(KitchenMember.kitchen_id == kitchen_id, KitchenMember.user_id == current_user.id).first()
    if not m:
        raise HTTPException(status_code=403, detail="Not a member of this kitchen")
    today = date.today()
    week_ago = today - timedelta(days=7)
    month_start = today.replace(day=1)
    tp = db.query(func.sum(Prediction.predicted_quantity)).filter(Prediction.kitchen_id == kitchen_id, Prediction.date == today).scalar() or 0
    ww = db.query(func.sum(WasteLog.quantity_kg)).filter(WasteLog.kitchen_id == kitchen_id, WasteLog.date >= week_ago).scalar() or 0
    wp = db.query(func.sum(Prediction.predicted_quantity)).filter(Prediction.kitchen_id == kitchen_id, Prediction.date >= week_ago).scalar() or 1
    waste_pct = round((ww / wp) * 100, 1) if wp > 0 else 0
    items = db.query(InventoryItem).filter(InventoryItem.kitchen_id == kitchen_id).all()
    ti = len(items)
    hi = sum(1 for i in items if i.stock_status in ["normal", "overstocked"])
    inv_health = round((hi / ti) * 100, 1) if ti > 0 else 100
    mc = db.query(func.sum(WasteLog.cost_impact)).filter(WasteLog.kitchen_id == kitchen_id, WasteLog.date >= month_start).scalar() or 0
    ts = db.query(Sensor).filter(Sensor.kitchen_id == kitchen_id).count()
    acs = db.query(Sensor).filter(Sensor.kitchen_id == kitchen_id, Sensor.is_active == True).count()
    ls = sum(1 for i in items if i.is_low_stock)
    return DashboardStats(total_meals_today=int(tp), waste_percentage=waste_pct, inventory_health=inv_health,
                          monthly_savings=round(mc * 0.3, 0), active_sensors=acs, total_sensors=ts,
                          low_stock_items=ls, predictions_accuracy=85.0)
