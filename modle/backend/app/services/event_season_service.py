from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from os import PathLike

import pandas as pd


SEASON_COLUMNS = [
    "벚꽃행사시즌",
    "야간개장시즌",
    "어린이행사시즌",
    "가을축제시즌",
]


def parse_event_period(event_text: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    matches = re.findall(r"(\d{4}\.\d{2}\.\d{2})", event_text)
    if not matches:
        raise ValueError(f"행사 날짜를 파싱할 수 없습니다: {event_text}")
    start_date = pd.to_datetime(matches[0], format="%Y.%m.%d")
    end_date = pd.to_datetime(matches[1], format="%Y.%m.%d") if len(matches) > 1 else start_date
    return start_date, end_date


def classify_event_name(event_text: str) -> str | None:
    lowered = event_text.replace(" ", "")
    if any(keyword in lowered for keyword in ["벚꽃", "봄꽃", "장미원", "꽃의숲", "정원", "식물원축제", "장미,정원을품다"]):
        return "벚꽃행사시즌"
    if any(keyword in lowered for keyword in ["어린이", "어린이날", "주(zoo)인공"]):
        return "어린이행사시즌"
    if any(keyword in lowered for keyword in ["영화제", "호숫가영화제", "야간", "하이킹"]):
        return "야간개장시즌"
    if any(keyword in lowered for keyword in ["한가위", "가을", "미술관", "동물원밖동물원", "숲속콘서트", "동물서식지문화축제", "동물원큰잔치"]):
        return "가을축제시즌"
    return None


@dataclass
class EventSeasonService:
    historical_daily_flags: pd.DataFrame
    recurring_day_of_year: dict[str, set[int]]

    @classmethod
    def from_csv(cls, events_file: str | PathLike[str]) -> "EventSeasonService":
        events = pd.read_csv(events_file)
        daily_records: list[dict[str, object]] = []
        recurring_day_of_year = {season: set() for season in SEASON_COLUMNS}

        for event_text in events["행사이름 - 날짜"].astype(str):
            season = classify_event_name(event_text)
            if not season:
                continue
            start_date, end_date = parse_event_period(event_text)
            for current_date in pd.date_range(start_date, end_date, freq="D"):
                daily_records.append({"일자": current_date, season: 1})
                recurring_day_of_year[season].add(int(current_date.strftime("%j")))

        if daily_records:
            daily_flags = pd.DataFrame(daily_records).groupby("일자", as_index=False).max()
        else:
            daily_flags = pd.DataFrame({"일자": []})

        for season in SEASON_COLUMNS:
            if season not in daily_flags.columns:
                daily_flags[season] = 0

        daily_flags["season_active_count"] = daily_flags[SEASON_COLUMNS].sum(axis=1)
        return cls(
            historical_daily_flags=daily_flags.sort_values("일자").reset_index(drop=True),
            recurring_day_of_year=recurring_day_of_year,
        )

    def merge_historical_flags(self, frame: pd.DataFrame) -> pd.DataFrame:
        merged = frame.merge(self.historical_daily_flags, on="일자", how="left")
        for season in SEASON_COLUMNS:
            merged[season] = merged[season].fillna(0).astype(int)
        merged["season_active_count"] = merged["season_active_count"].fillna(0).astype(int)
        return merged

    def recurring_flags_for_date(self, target_date: date) -> dict[str, int]:
        day_of_year = int(target_date.strftime("%j"))
        flags = {
            season: int(day_of_year in self.recurring_day_of_year[season])
            for season in SEASON_COLUMNS
        }
        flags["season_active_count"] = sum(flags.values())
        return flags
