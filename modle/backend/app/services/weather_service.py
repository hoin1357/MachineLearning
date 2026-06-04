from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd

from app.config import (
    MID_LAND_REGION_ID,
    MID_TEMPERATURE_REGION_ID,
    MID_WEATHER_API_BASE_URL,
    SEOUL_TIMEZONE,
    WEATHER_API_BASE_URL,
    WEATHER_API_SERVICE_KEY,
    WEATHER_CACHE_FILE,
    WEATHER_GRID_X,
    WEATHER_GRID_Y,
    WEATHER_STRATEGY_VERSION,
    current_seoul_date,
)


@dataclass
class DailyWeatherFeature:
    average_temperature: float
    average_precipitation: float
    source: str


class WeatherService:
    def __init__(self, weather_history_dir: Path) -> None:
        self.weather_history_dir = weather_history_dir
        self.weather_history = self._load_history()
        self.by_month_day = self.weather_history.groupby("month_day", as_index=False).agg(
            average_temperature=("평균기온", "mean"),
            average_precipitation=("일강수량", "mean"),
        )
        self.by_month = self.weather_history.groupby("month", as_index=False).agg(
            average_temperature=("평균기온", "mean"),
            average_precipitation=("일강수량", "mean"),
        )
        self.api_cache = self._load_api_cache()

    def _load_history(self) -> pd.DataFrame:
        frames = []
        for csv_path in sorted(self.weather_history_dir.glob("*.csv")):
            frame = pd.read_csv(csv_path, encoding="cp949")
            frame["일자"] = pd.to_datetime(frame["일시"])
            frame["평균기온"] = pd.to_numeric(frame["평균기온(°C)"], errors="coerce")
            frame["일강수량"] = pd.to_numeric(frame["일강수량(mm)"], errors="coerce").fillna(0.0)
            frames.append(frame[["일자", "평균기온", "일강수량"]])
        weather = pd.concat(frames, ignore_index=True).sort_values("일자").reset_index(drop=True)
        weather["평균기온"] = weather["평균기온"].interpolate(limit_direction="both")
        weather["month_day"] = weather["일자"].dt.strftime("%m-%d")
        weather["month"] = weather["일자"].dt.month
        return weather

    def _load_api_cache(self) -> dict[str, dict[str, float | str]]:
        if WEATHER_CACHE_FILE.exists():
            return json.loads(WEATHER_CACHE_FILE.read_text(encoding="utf-8"))
        return {}

    def _save_api_cache(self) -> None:
        WEATHER_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        WEATHER_CACHE_FILE.write_text(json.dumps(self.api_cache, ensure_ascii=False, indent=2), encoding="utf-8")

    def _forecast_base_datetime(self, now: datetime) -> tuple[str, str]:
        base_times = ["0200", "0500", "0800", "1100", "1400", "1700", "2000", "2300"]
        hhmm = int(now.strftime("%H%M"))
        base_time = "2300"
        base_date = now.date()
        for candidate in base_times:
            if hhmm >= int(candidate) + 10:
                base_time = candidate
        if hhmm < 210:
            base_date = now.date() - timedelta(days=1)
            base_time = "2300"
        return base_date.strftime("%Y%m%d"), base_time

    def _mid_forecast_base_datetime(self, now: datetime) -> datetime:
        seoul_now = now.astimezone(SEOUL_TIMEZONE) if now.tzinfo else now.replace(tzinfo=SEOUL_TIMEZONE)
        if seoul_now.hour >= 18:
            return seoul_now.replace(hour=18, minute=0, second=0, microsecond=0)
        if seoul_now.hour >= 6:
            return seoul_now.replace(hour=6, minute=0, second=0, microsecond=0)
        return (seoul_now - timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)

    def _parse_precipitation(self, raw_value: str) -> float:
        value = raw_value.strip()
        if value in {"강수없음", "0", "0.0"}:
            return 0.0
        if "1mm 미만" in value:
            return 0.5
        if "~" in value:
            parts = value.replace("mm", "").split("~")
            numbers = [float(part.strip()) for part in parts if part.strip()]
            if len(numbers) == 2:
                return sum(numbers) / 2
        cleaned = value.replace("mm", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return 0.0

    def _extract_items(self, body: dict) -> list[dict]:
        response = body.get("response", {})
        header = response.get("header", {})
        if header.get("resultCode") != "00":
            raise ValueError(header.get("resultMsg") or "날씨 API 응답이 정상 상태가 아닙니다.")
        items = response.get("body", {}).get("items", {}).get("item", [])
        if isinstance(items, dict):
            return [items]
        if not isinstance(items, list):
            raise ValueError("날씨 API 응답 item 형식이 올바르지 않습니다.")
        return items

    def _precipitation_from_mid_forecast(
        self,
        morning_probability: float,
        afternoon_probability: float,
        *weather_texts: str,
    ) -> float:
        probability = (morning_probability + afternoon_probability) / 2
        weather_text = " ".join(text for text in weather_texts if text)
        has_wet_weather = any(keyword in weather_text for keyword in ("비", "눈", "소나기"))
        if not has_wet_weather:
            return 0.0 if probability < 50 else 0.5

        base_amount = 5.0
        if "눈" in weather_text and "비" not in weather_text:
            base_amount = 3.0
        if "많" in weather_text or "강" in weather_text:
            base_amount = 8.0
        return round(base_amount * (probability / 100), 2)

    def _fetch_short_forecast_block(self) -> dict[str, DailyWeatherFeature]:
        now = datetime.now(SEOUL_TIMEZONE)
        base_date, base_time = self._forecast_base_datetime(now)
        params = {
            "serviceKey": WEATHER_API_SERVICE_KEY,
            "pageNo": 1,
            "numOfRows": 1000,
            "dataType": "JSON",
            "base_date": base_date,
            "base_time": base_time,
            "nx": WEATHER_GRID_X,
            "ny": WEATHER_GRID_Y,
        }
        url = f"{WEATHER_API_BASE_URL}/getVilageFcst"

        forecast_by_date: dict[str, dict[str, list[float]]] = {}
        with httpx.Client(timeout=8.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            items = self._extract_items(response.json())
        for item in items:
            fcst_date = str(item["fcstDate"])
            if fcst_date not in forecast_by_date:
                forecast_by_date[fcst_date] = {"TMP": [], "PCP": []}
            if item["category"] == "TMP":
                forecast_by_date[fcst_date]["TMP"].append(float(item["fcstValue"]))
            elif item["category"] == "PCP":
                forecast_by_date[fcst_date]["PCP"].append(self._parse_precipitation(str(item["fcstValue"])))

        transformed = {}
        for key, values in forecast_by_date.items():
            if not values["TMP"]:
                continue
            transformed[pd.to_datetime(key, format="%Y%m%d").date().isoformat()] = DailyWeatherFeature(
                average_temperature=float(sum(values["TMP"]) / len(values["TMP"])),
                average_precipitation=float(sum(values["PCP"]) / len(values["PCP"])) if values["PCP"] else 0.0,
                source="forecast",
            )
        return transformed

    def _fetch_mid_forecast_block(self) -> dict[str, DailyWeatherFeature]:
        base_datetime = self._mid_forecast_base_datetime(datetime.now(SEOUL_TIMEZONE))
        common_params = {
            "serviceKey": WEATHER_API_SERVICE_KEY,
            "pageNo": 1,
            "numOfRows": 10,
            "dataType": "JSON",
            "tmFc": base_datetime.strftime("%Y%m%d%H%M"),
        }
        with httpx.Client(timeout=8.0) as client:
            temperature_response = client.get(
                f"{MID_WEATHER_API_BASE_URL}/getMidTa",
                params={**common_params, "regId": MID_TEMPERATURE_REGION_ID},
            )
            temperature_response.raise_for_status()
            temperature_items = self._extract_items(temperature_response.json())

            land_response = client.get(
                f"{MID_WEATHER_API_BASE_URL}/getMidLandFcst",
                params={**common_params, "regId": MID_LAND_REGION_ID},
            )
            land_response.raise_for_status()
            land_items = self._extract_items(land_response.json())

        if not temperature_items or not land_items:
            return {}

        temperature_item = temperature_items[0]
        land_item = land_items[0]
        transformed = {}
        for day_offset in (4, 5):
            min_key = f"taMin{day_offset}"
            max_key = f"taMax{day_offset}"
            if min_key not in temperature_item or max_key not in temperature_item:
                continue
            rn_am = float(land_item.get(f"rnSt{day_offset}Am", 0))
            rn_pm = float(land_item.get(f"rnSt{day_offset}Pm", 0))
            target_date = base_datetime.date() + timedelta(days=day_offset)
            transformed[target_date.isoformat()] = DailyWeatherFeature(
                average_temperature=(float(temperature_item[min_key]) + float(temperature_item[max_key])) / 2,
                average_precipitation=self._precipitation_from_mid_forecast(
                    rn_am,
                    rn_pm,
                    str(land_item.get(f"wf{day_offset}Am", "")),
                    str(land_item.get(f"wf{day_offset}Pm", "")),
                ),
                source="mid_forecast",
            )
        return transformed

    def _fetch_forecast_block(self) -> dict[str, DailyWeatherFeature]:
        if not WEATHER_API_SERVICE_KEY:
            raise ValueError("WEATHER_API_SERVICE_KEY 환경변수가 설정되지 않았습니다.")

        cache_key = f"{current_seoul_date().isoformat()}:{WEATHER_STRATEGY_VERSION}"
        if cache_key in self.api_cache:
            return {
                key: DailyWeatherFeature(
                    average_temperature=value["average_temperature"],
                    average_precipitation=value["average_precipitation"],
                    source=value["source"],
                )
                for key, value in self.api_cache[cache_key].items()
            }

        transformed = {}
        for fetcher in (self._fetch_mid_forecast_block, self._fetch_short_forecast_block):
            try:
                transformed.update(fetcher())
            except Exception:
                continue
        if not transformed:
            raise ValueError("사용 가능한 날씨 API 예보가 없습니다.")

        self.api_cache[cache_key] = {
            key: {
                "average_temperature": value.average_temperature,
                "average_precipitation": value.average_precipitation,
                "source": value.source,
            }
            for key, value in transformed.items()
        }
        self._save_api_cache()
        return transformed

    def _climatology_for_date(self, target_date: date) -> DailyWeatherFeature:
        month_day = target_date.strftime("%m-%d")
        match = self.by_month_day.loc[self.by_month_day["month_day"] == month_day]
        if not match.empty:
            row = match.iloc[0]
            return DailyWeatherFeature(
                average_temperature=float(row["average_temperature"]),
                average_precipitation=float(row["average_precipitation"]),
                source="climatology",
            )

        month_match = self.by_month.loc[self.by_month["month"] == target_date.month]
        row = month_match.iloc[0]
        return DailyWeatherFeature(
            average_temperature=float(row["average_temperature"]),
            average_precipitation=float(row["average_precipitation"]),
            source="climatology",
        )

    def feature_for_date(self, target_date: date) -> DailyWeatherFeature:
        today = current_seoul_date()
        if target_date < today:
            row = self.weather_history.loc[self.weather_history["일자"] == pd.Timestamp(target_date)]
            if not row.empty:
                row = row.iloc[0]
                return DailyWeatherFeature(
                    average_temperature=float(row["평균기온"]),
                    average_precipitation=float(row["일강수량"]),
                    source="historical",
                )

        if target_date <= today + timedelta(days=5):
            try:
                forecast_block = self._fetch_forecast_block()
                if target_date.isoformat() in forecast_block:
                    return forecast_block[target_date.isoformat()]
            except Exception:
                pass

        return self._climatology_for_date(target_date)
