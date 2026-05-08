"""
SmartKitchen AI X — Kitchen Management API
CRUD operations for kitchen profiles.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import json

from app.database import get_db
from app.models.kitchen import Kitchen, KitchenMember
from app.models.user import User
from app.schemas.schemas import KitchenCreate, KitchenUpdate, KitchenResponse
from app.api.v1.auth import get_current_user

router = APIRouter()


@router.post("/", response_model=KitchenResponse, status_code=201)
async def create_kitchen(
    data: KitchenCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new kitchen."""
    meal_times_json = json.dumps(data.meal_times) if data.meal_times else '{"breakfast": "07:30", "lunch": "12:30", "dinner": "19:30"}'
    kitchen = Kitchen(
        name=data.name,
        type=data.type.value,
        location=data.location,
        latitude=data.latitude,
        longitude=data.longitude,
        capacity=data.capacity,
        meal_times=meal_times_json,
        owner_id=current_user.id,
    )
    db.add(kitchen)
    db.flush()

    # Add creator as admin
    membership = KitchenMember(
        kitchen_id=kitchen.id,
        user_id=current_user.id,
        role="admin",
    )
    db.add(membership)
    db.commit()
    db.refresh(kitchen)
    return kitchen


@router.get("/", response_model=List[KitchenResponse])
async def list_kitchens(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all kitchens the current user belongs to."""
    kitchen_ids = (
        db.query(KitchenMember.kitchen_id)
        .filter(KitchenMember.user_id == current_user.id)
        .subquery()
    )
    kitchens = db.query(Kitchen).filter(Kitchen.id.in_(kitchen_ids)).all()
    return kitchens


@router.get("/{kitchen_id}", response_model=KitchenResponse)
async def get_kitchen(
    kitchen_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get kitchen details."""
    kitchen = db.query(Kitchen).filter(Kitchen.id == kitchen_id).first()
    if not kitchen:
        raise HTTPException(status_code=404, detail="Kitchen not found")

    # Verify membership
    membership = (
        db.query(KitchenMember)
        .filter(KitchenMember.kitchen_id == kitchen_id, KitchenMember.user_id == current_user.id)
        .first()
    )
    if not membership:
        raise HTTPException(status_code=403, detail="Not a member of this kitchen")

    return kitchen


@router.put("/{kitchen_id}", response_model=KitchenResponse)
async def update_kitchen(
    kitchen_id: str,
    data: KitchenUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update kitchen details."""
    kitchen = db.query(Kitchen).filter(Kitchen.id == kitchen_id).first()
    if not kitchen:
        raise HTTPException(status_code=404, detail="Kitchen not found")

    # Check admin role
    membership = (
        db.query(KitchenMember)
        .filter(
            KitchenMember.kitchen_id == kitchen_id,
            KitchenMember.user_id == current_user.id,
            KitchenMember.role.in_(["admin", "manager"]),
        )
        .first()
    )
    if not membership:
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    update_data = data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(kitchen, key, value)

    db.commit()
    db.refresh(kitchen)
    return kitchen
