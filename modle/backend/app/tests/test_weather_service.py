from __future__ import annotations

from datetime import datetime

from app.config import SEOUL_TIMEZONE
from app.services.weather_service import WeatherService


def test_mid_forecast_base_datetime_uses_latest_kst_release() -> None:
    service = WeatherService.__new__(WeatherService)

    morning = datetime(2026, 4, 22, 7, 0, tzinfo=SEOUL_TIMEZONE)
    evening = datetime(2026, 4, 22, 21, 0, tzinfo=SEOUL_TIMEZONE)
    before_first_release = datetime(2026, 4, 22, 5, 0, tzinfo=SEOUL_TIMEZONE)

    assert service._mid_forecast_base_datetime(morning).strftime("%Y%m%d%H%M") == "202604220600"
    assert service._mid_forecast_base_datetime(evening).strftime("%Y%m%d%H%M") == "202604221800"
    assert service._mid_forecast_base_datetime(before_first_release).strftime("%Y%m%d%H%M") == "202604211800"


def test_mid_precipitation_mapping_uses_probability_and_weather_text() -> None:
    service = WeatherService.__new__(WeatherService)

    assert service._precipitation_from_mid_forecast(40, 60, "흐림", "흐리고 비") == 2.5
    assert service._precipitation_from_mid_forecast(20, 20, "구름많음", "구름많음") == 0.0
    assert service._precipitation_from_mid_forecast(60, 60, "흐림", "흐림") == 0.5


def test_extract_items_rejects_error_response() -> None:
    service = WeatherService.__new__(WeatherService)
    body = {"response": {"header": {"resultCode": "03", "resultMsg": "NO_DATA"}}}

    try:
        service._extract_items(body)
    except ValueError as exc:
        assert "NO_DATA" in str(exc)
    else:
        raise AssertionError("ValueError가 발생해야 합니다.")
