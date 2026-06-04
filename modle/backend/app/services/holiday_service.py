from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from types import MappingProxyType

from korean_lunar_calendar import KoreanLunarCalendar


FIXED_SOLAR_HOLIDAYS = {
    (1, 1): "신정",
    (3, 1): "삼일절",
    (5, 5): "어린이날",
    (6, 6): "현충일",
    (8, 15): "광복절",
    (10, 3): "개천절",
    (10, 9): "한글날",
    (12, 25): "기독탄신일",
}

HOLIDAY_ICON_KEYWORDS = [
    ("설날", "seollal"),
    ("추석", "chuseok"),
    ("어린이", "children"),
    ("부처", "buddha"),
    ("현충", "memorial"),
    ("한글", "hangeul"),
    ("기독", "christmas"),
    ("성탄", "christmas"),
    ("신정", "new-year"),
    ("삼일", "korea"),
    ("광복", "korea"),
    ("개천", "korea"),
    ("선거", "election"),
    ("대체", "substitute"),
    ("임시", "special"),
]


@lru_cache(maxsize=None)
def _lunar_to_solar(year: int, month: int, day: int) -> date:
    calendar = KoreanLunarCalendar()
    calendar.setLunarDate(year, month, day, False)
    return date.fromisoformat(calendar.SolarIsoFormat())


@lru_cache(maxsize=None)
def holiday_names_for_year(year: int) -> MappingProxyType:
    holidays: dict[date, list[str]] = {}

    def add_holiday(target: date, name: str) -> None:
        holidays.setdefault(target, [])
        if name not in holidays[target]:
            holidays[target].append(name)

    seollal = _lunar_to_solar(year, 1, 1)
    add_holiday(seollal - timedelta(days=1), "설날 연휴")
    add_holiday(seollal, "설날")
    add_holiday(seollal + timedelta(days=1), "설날 연휴")

    buddha_birthday = _lunar_to_solar(year, 4, 8)
    add_holiday(buddha_birthday, "부처님오신날")

    chuseok = _lunar_to_solar(year, 8, 15)
    add_holiday(chuseok - timedelta(days=1), "추석 연휴")
    add_holiday(chuseok, "추석")
    add_holiday(chuseok + timedelta(days=1), "추석 연휴")

    for (month, day), name in FIXED_SOLAR_HOLIDAYS.items():
        add_holiday(date(year, month, day), name)

    for substitute, source_names in _substitute_holidays(holidays).items():
        for source_name in source_names:
            add_holiday(substitute, f"{source_name} 대체공휴일")

    return MappingProxyType(holidays)


@lru_cache(maxsize=None)
def holidays_for_year(year: int) -> frozenset[date]:
    return frozenset(holiday_names_for_year(year).keys())


def _substitute_holidays(base_holidays: dict[date, list[str]]) -> dict[date, list[str]]:
    substitutes: dict[date, list[str]] = {}
    base_dates = set(base_holidays)
    for holiday in sorted(base_dates):
        if holiday.weekday() < 5:
            continue
        substitute = holiday + timedelta(days=1)
        while substitute.weekday() >= 5 or substitute in base_dates or substitute in substitutes:
            substitute += timedelta(days=1)
        substitutes[substitute] = list(base_holidays[holiday])
    return substitutes


def holiday_names(value: date | str) -> list[str]:
    target = date.fromisoformat(value) if isinstance(value, str) else value
    names: list[str] = []
    for year in (target.year - 1, target.year, target.year + 1):
        for name in holiday_names_for_year(year).get(target, []):
            if name not in names:
                names.append(name)
    return names


def holiday_icon_key(holiday_name: str) -> str:
    for keyword, icon_key in HOLIDAY_ICON_KEYWORDS:
        if keyword in holiday_name:
            return icon_key
    return "special"


def holiday_info(value: date | str) -> list[dict[str, str]]:
    return [
        {
            "name": name,
            "icon": holiday_icon_key(name),
        }
        for name in holiday_names(value)
    ]


def is_holiday(value: date | str) -> bool:
    return bool(holiday_names(value))
