"""
SmartKitchen AI X — Waste Analytics API
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List
from datetime import date, timedelta

from app.database import get_db
from app.models.waste import WasteLog
from app.models.inventory import InventoryItem
from app.models.kitchen import KitchenMember
from app.models.user import User
from app.schemas.schemas import WasteLogCreate, WasteLogResponse
from app.api.v1.auth import get_current_user

router = APIRouter()


def verify_kitchen_access(db, kitchen_id, user_id):
    m = db.query(KitchenMember).filter(
        KitchenMember.kitchen_id == kitchen_id, KitchenMember.user_id == user_id
    ).first()
    if not m:
        raise HTTPException(status_code=403, detail="Not a member of this kitchen")


@router.post("/kitchens/{kitchen_id}/waste", response_model=WasteLogResponse, status_code=201)
async def log_waste(kitchen_id: str, data: WasteLogCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    verify_kitchen_access(db, kitchen_id, current_user.id)
    cost_impact = None
    inv = db.query(InventoryItem).filter(InventoryItem.kitchen_id == kitchen_id, InventoryItem.name == data.item).first()
    if inv and inv.cost_per_unit:
        cost_impact = round(data.quantity_kg * inv.cost_per_unit, 2)
    wl = WasteLog(kitchen_id=kitchen_id, item=data.item, quantity_kg=data.quantity_kg,
                  meal_type=data.meal_type.value if data.meal_type else None,
                  reason=data.reason.value, cost_impact=cost_impact, date=data.date,
                  notes=data.notes, logged_by=current_user.id)
    db.add(wl)
    db.commit()
    db.refresh(wl)
    return wl


@router.get("/kitchens/{kitchen_id}/waste", response_model=List[WasteLogResponse])
async def list_waste_logs(kitchen_id: str, days: int = Query(default=30, le=365), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    verify_kitchen_access(db, kitchen_id, current_user.id)
    cutoff = date.today() - timedelta(days=days)
    return db.query(WasteLog).filter(WasteLog.kitchen_id == kitchen_id, WasteLog.date >= cutoff).order_by(WasteLog.date.desc()).limit(200).all()


@router.get("/kitchens/{kitchen_id}/waste/analytics")
async def get_waste_analytics(kitchen_id: str, days: int = Query(default=30, le=365), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    verify_kitchen_access(db, kitchen_id, current_user.id)
    cutoff = date.today() - timedelta(days=days)
    logs = db.query(WasteLog).filter(WasteLog.kitchen_id == kitchen_id, WasteLog.date >= cutoff).all()
    if not logs:
        return {"total_waste_kg": 0, "cost_impact": 0, "by_reason": [], "by_item": [], "trend": [], "period_days": days}
    tw = sum(l.quantity_kg for l in logs)
    tc = sum(l.cost_impact or 0 for l in logs)
    rm = {}
    for l in logs:
        r = l.reason or "unknown"
        rm[r] = rm.get(r, 0) + l.quantity_kg
    by_reason = [{"reason": k, "kg": round(v, 1), "pct": round(v / tw * 100, 1)} for k, v in sorted(rm.items(), key=lambda x: x[1], reverse=True)]
    im = {}
    for l in logs:
        im[l.item] = im.get(l.item, 0) + l.quantity_kg
    by_item = [{"item": k, "kg": round(v, 1)} for k, v in sorted(im.items(), key=lambda x: x[1], reverse=True)[:10]]
    dm = {}
    for l in logs:
        d = str(l.date)
        dm.setdefault(d, 0)
        dm[d] += l.quantity_kg
    trend = [{"date": k, "waste_kg": round(v, 1)} for k, v in sorted(dm.items())]
    return {"total_waste_kg": round(tw, 1), "avg_daily_kg": round(tw / days, 1), "cost_impact": round(tc, 1), "by_reason": by_reason, "by_item": by_item, "trend": trend, "period_days": days}
