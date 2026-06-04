from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date

from app.config import MONTH_NAMES_KO, TODAY
from app.domain.commentary import color_for_prediction
from app.services.holiday_service import holiday_info, is_holiday
from app.services.model_runtime import PortablePredictionRuntime


@dataclass
class CalendarService:
    runtime: PortablePredictionRuntime

    def month_payload(self, year: int, month: int, selected_date: date | None, risk_mode: bool) -> dict:
        first_weekday, days_in_month = calendar.monthrange(year, month)
        weeks = []
        week = [None] * ((first_weekday + 1) % 7)
        risk_lookup = {}
        if risk_mode and selected_date:
            risk_lookup = {
                item["date"]: item
                for item in self.runtime.list_predictions_for_window(selected_date)
            }

        for day in range(1, days_in_month + 1):
            current_date = date(year, month, day)
            selectable, reason = self.runtime.is_selectable(current_date)
            iso_date = current_date.isoformat()
            visitors = None
            risk_color = None
            congestion_level = None
            if iso_date in risk_lookup:
                visitors = risk_lookup[iso_date]["predictedVisitors"]
                risk_color = risk_lookup[iso_date]["riskColor"]
                congestion_level = risk_lookup[iso_date]["congestionLevel"]
            holiday_items = holiday_info(iso_date)

            week.append(
                {
                    "date": iso_date,
                    "day": day,
                    "isToday": current_date == TODAY,
                    "isSelected": selected_date == current_date,
                    "isWeekend": current_date.weekday() >= 5,
                    "isHoliday": bool(is_holiday(iso_date)),
                    "holidayNames": [item["name"] for item in holiday_items],
                    "holidayIcons": holiday_items,
                    "selectable": selectable,
                    "reason": reason,
                    "riskColor": risk_color,
                    "riskTextColor": color_for_prediction(visitors) if visitors is not None else None,
                    "congestionLevel": congestion_level,
                }
            )
            if len(week) == 7:
                weeks.append(week)
                week = []

        if week:
            week.extend([None] * (7 - len(week)))
            weeks.append(week)

        return {
            "year": year,
            "month": month,
            "monthLabel": MONTH_NAMES_KO[month - 1],
            "weeks": weeks,
            "selectedDate": selected_date.isoformat() if selected_date else None,
            "riskMode": risk_mode,
            "today": TODAY.isoformat(),
            "maxDate": self.runtime.max_supported_date.isoformat(),
        }
