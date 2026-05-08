"""
SmartKitchen AI X — Inventory Management API
CRUD operations for food inventory with stock tracking.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional, List
from datetime import datetime

from app.database import get_db
from app.models.inventory import InventoryItem, StockLog
from app.models.kitchen import KitchenMember
from app.models.user import User
from app.schemas.schemas import (
    InventoryItemCreate, InventoryItemUpdate, InventoryItemResponse, RestockRequest
)
from app.api.v1.auth import get_current_user

router = APIRouter()


def verify_kitchen_access(db: Session, kitchen_id: str, user_id: str):
    """Verify user has access to the kitchen."""
    membership = (
        db.query(KitchenMember)
        .filter(KitchenMember.kitchen_id == kitchen_id, KitchenMember.user_id == user_id)
        .first()
    )
    if not membership:
        raise HTTPException(status_code=403, detail="Not a member of this kitchen")
    return membership


@router.get("/kitchens/{kitchen_id}/inventory", response_model=List[InventoryItemResponse])
async def list_inventory(
    kitchen_id: str,
    category: Optional[str] = None,
    status: Optional[str] = Query(None, description="Filter: low, normal, out_of_stock"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all inventory items for a kitchen."""
    verify_kitchen_access(db, kitchen_id, current_user.id)

    query = db.query(InventoryItem).filter(InventoryItem.kitchen_id == kitchen_id)

    if category:
        query = query.filter(InventoryItem.category == category)

    items = query.order_by(InventoryItem.name).all()

    # Filter by status if requested
    if status:
        items = [item for item in items if item.stock_status == status]

    return items


@router.post("/kitchens/{kitchen_id}/inventory", response_model=InventoryItemResponse, status_code=201)
async def create_inventory_item(
    kitchen_id: str,
    data: InventoryItemCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Add a new inventory item."""
    verify_kitchen_access(db, kitchen_id, current_user.id)

    item = InventoryItem(
        kitchen_id=kitchen_id,
        name=data.name,
        category=data.category.value,
        quantity=data.quantity,
        unit=data.unit,
        min_threshold=data.min_threshold,
        max_capacity=data.max_capacity,
        cost_per_unit=data.cost_per_unit,
        supplier=data.supplier,
        shelf_life_days=data.shelf_life_days,
        last_restocked=datetime.utcnow() if data.quantity > 0 else None,
    )
    db.add(item)
    db.flush()

    # Log initial stock
    if data.quantity > 0:
        log = StockLog(
            item_id=item.id,
            action="restock",
            quantity_change=data.quantity,
            new_quantity=data.quantity,
            performed_by=current_user.id,
            note="Initial stock entry",
        )
        db.add(log)

    db.commit()
    db.refresh(item)
    return item


@router.put("/inventory/{item_id}", response_model=InventoryItemResponse)
async def update_inventory_item(
    item_id: str,
    data: InventoryItemUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update an inventory item."""
    item = db.query(InventoryItem).filter(InventoryItem.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    verify_kitchen_access(db, item.kitchen_id, current_user.id)

    # Track quantity change
    old_quantity = item.quantity

    update_data = data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(item, key, value)

    # Log if quantity changed
    if "quantity" in update_data and update_data["quantity"] != old_quantity:
        change = update_data["quantity"] - old_quantity
        log = StockLog(
            item_id=item.id,
            action="adjust",
            quantity_change=change,
            new_quantity=update_data["quantity"],
            performed_by=current_user.id,
            note="Manual adjustment",
        )
        db.add(log)

    db.commit()
    db.refresh(item)
    return item


@router.delete("/inventory/{item_id}", status_code=204)
async def delete_inventory_item(
    item_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete an inventory item."""
    item = db.query(InventoryItem).filter(InventoryItem.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    verify_kitchen_access(db, item.kitchen_id, current_user.id)
    db.delete(item)
    db.commit()


@router.post("/inventory/{item_id}/restock", response_model=InventoryItemResponse)
async def restock_item(
    item_id: str,
    data: RestockRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Add stock to an inventory item."""
    item = db.query(InventoryItem).filter(InventoryItem.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    verify_kitchen_access(db, item.kitchen_id, current_user.id)

    item.quantity += data.quantity_added
    item.last_restocked = datetime.utcnow()

    log = StockLog(
        item_id=item.id,
        action="restock",
        quantity_change=data.quantity_added,
        new_quantity=item.quantity,
        cost=data.cost,
        performed_by=current_user.id,
        note=data.note,
    )
    db.add(log)
    db.commit()
    db.refresh(item)
    return item


@router.get("/kitchens/{kitchen_id}/inventory/alerts")
async def get_low_stock_alerts(
    kitchen_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get items with low stock levels."""
    verify_kitchen_access(db, kitchen_id, current_user.id)

    items = (
        db.query(InventoryItem)
        .filter(InventoryItem.kitchen_id == kitchen_id)
        .all()
    )

    alerts = []
    for item in items:
        if item.is_low_stock:
            deficit = item.min_threshold - item.quantity if item.min_threshold else 0
            alerts.append({
                "item_id": str(item.id),
                "name": item.name,
                "category": item.category,
                "current_quantity": item.quantity,
                "min_threshold": item.min_threshold,
                "unit": item.unit,
                "deficit": round(deficit, 1),
                "status": item.stock_status,
            })

    return sorted(alerts, key=lambda x: x["deficit"], reverse=True)
