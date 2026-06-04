from datetime import timedelta

from app.config import TODAY
from app.services.holiday_service import holiday_icon_key, holiday_info, holiday_names
from app.services.model_runtime import PortablePredictionRuntime


def test_past_dates_are_blocked() -> None:
    runtime = PortablePredictionRuntime()
    selectable, reason = runtime.is_selectable(TODAY - timedelta(days=1))
    assert selectable is False
    assert reason is not None


def test_holiday_names_and_icons_are_available() -> None:
    assert "어린이날" in holiday_names("2026-05-05")
    assert holiday_icon_key("어린이날") == "children"
    assert holiday_info("2026-09-25")[0]["icon"] == "chuseok"
