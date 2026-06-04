from __future__ import annotations

from datetime import date

from fastapi import APIRouter, HTTPException, Query

from app.services.calendar_service import CalendarService
from app.services.model_runtime import PortablePredictionRuntime


router = APIRouter()
runtime = PortablePredictionRuntime()
calendar_service = CalendarService(runtime=runtime)


@router.get("/health")
def health() -> dict:
    return runtime.health()


@router.get("/calendar/month")
def calendar_month(
    year: int = Query(..., ge=2025, le=2100),
    month: int = Query(..., ge=1, le=12),
    selectedDate: str | None = Query(None),
    riskMode: bool = Query(False),
) -> dict:
    selected = date.fromisoformat(selectedDate) if selectedDate else None
    return calendar_service.month_payload(year=year, month=month, selected_date=selected, risk_mode=riskMode)


@router.get("/predictions/{target_date}")
def prediction(target_date: str) -> dict:
    try:
        return runtime.predict_for_date(date.fromisoformat(target_date))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/risk-window/{target_date}")
def risk_window(target_date: str) -> dict:
    parsed_date = date.fromisoformat(target_date)
    selectable, reason = runtime.is_selectable(parsed_date)
    if not selectable:
        raise HTTPException(status_code=400, detail=reason)
    return {
        "selectedDate": parsed_date.isoformat(),
        "days": runtime.list_predictions_for_window(parsed_date),
    }
