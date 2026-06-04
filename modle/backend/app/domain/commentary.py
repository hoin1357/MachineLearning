from __future__ import annotations

import random
from datetime import date

from app.config import WEEKDAY_NAMES_KO


CONGESTION_THRESHOLDS = {
    "한산함": (0, 2999),
    "보통": (3000, 7999),
    "붐빔": (8000, 14999),
    "매우 붐빔": (15000, 30000),
}

COLOR_STOPS = {
    "한산함": ("#fff7cc", "#ffe680"),
    "보통": ("#ffd166", "#ffb347"),
    "붐빔": ("#ff8a3d", "#ff4d2e"),
    "매우 붐빔": ("#ff2d2d", "#e60000"),
}

LEVEL_COMMENTS = {
    "한산함": [
        "산책하듯 여유롭게 둘러보기 좋은 흐름이에요.",
        "사진 찍고 쉬어 가기에도 부담이 적은 날이에요.",
        "아이와 천천히 코스를 돌기 좋은 한산한 편이에요.",
        "대기와 이동 스트레스가 크지 않을 가능성이 높아요.",
        "유모차나 짐이 있어도 비교적 수월한 날로 보여요.",
        "편하게 들어가서 원하는 동선을 잡기 좋은 날이에요.",
        "붐비는 구간을 피해 다닐 필요가 크지 않은 수준이에요.",
        "가볍게 나들이 가기 좋은 차분한 흐름으로 예상돼요.",
    ],
    "보통": [
        "적당히 활기 있지만 크게 답답하지는 않을 가능성이 높아요.",
        "무난한 방문일로, 주요 구역은 약간의 대기가 생길 수 있어요.",
        "점심 전후만 잘 피해도 꽤 편하게 돌아볼 수 있는 수준이에요.",
        "가족 단위 방문객이 조금씩 늘어나는 정도로 보여요.",
        "크게 붐비진 않지만 인기 구역은 체크해 두는 편이 좋아요.",
        "보통 수준이라 일정만 잘 짜면 안정적으로 다녀오기 좋겠어요.",
        "한산하진 않지만 과하게 북적이는 흐름은 아닐 가능성이 커요.",
        "평균적인 주말/행사 수요 정도로 받아들이면 무난한 날이에요.",
    ],
    "붐빔": [
        "주요 동선은 서둘러 움직이는 편이 유리해 보여요.",
        "피크 시간대에는 체감 혼잡이 꽤 커질 수 있어요.",
        "입장과 주차, 식사 동선을 미리 정해 두는 편이 좋아요.",
        "인기 구역은 대기와 밀집이 생길 가능성이 높아요.",
        "아이와 함께라면 쉬는 지점을 중간중간 확보해 두는 편이 안전해요.",
        "조금 서둘러 출발해야 만족도가 올라갈 가능성이 커요.",
        "붐비는 날에 가까워 보여서 이동 계획을 짧고 선명하게 잡는 게 좋아요.",
        "오후로 갈수록 혼잡 체감이 더 커질 수 있는 패턴이에요.",
    ],
    "매우 붐빔": [
        "상당히 붐빌 가능성이 커서 일정 여유를 넉넉히 잡아야 해요.",
        "가족 나들이라면 입장 시간과 휴식 포인트를 꼭 먼저 정해 두세요.",
        "주차와 인기 구역 대기 부담이 크게 느껴질 수 있는 날이에요.",
        "특히 정오 이후에는 체감 혼잡이 매우 높아질 가능성이 있어요.",
        "혼잡 스트레스가 커질 수 있어 대체 날짜 검토도 충분히 추천돼요.",
        "동선이 꼬이기 쉬운 수준이라 방문 목적을 좁혀 움직이는 편이 좋아요.",
        "빠른 입장, 빠른 이동이 아니면 예상보다 훨씬 지칠 수 있는 날이에요.",
        "가능하면 오전 초반에 핵심 코스를 먼저 소화하는 편이 유리해 보여요.",
    ],
}


def congestion_level_from_visitors(visitors: int) -> str:
    if visitors < 3000:
        return "한산함"
    if visitors < 8000:
        return "보통"
    if visitors < 15000:
        return "붐빔"
    return "매우 붐빔"


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.removeprefix("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _interpolate_color(start_hex: str, end_hex: str, ratio: float) -> str:
    start_rgb = _hex_to_rgb(start_hex)
    end_rgb = _hex_to_rgb(end_hex)
    bounded = max(0.0, min(1.0, ratio))
    current = tuple(round(start + (end - start) * bounded) for start, end in zip(start_rgb, end_rgb))
    return _rgb_to_hex(current)


def color_for_prediction(visitors: int) -> str:
    level = congestion_level_from_visitors(visitors)
    min_value, max_value = CONGESTION_THRESHOLDS[level]
    if max_value <= min_value:
        ratio = 1.0
    else:
        ratio = (visitors - min_value) / (max_value - min_value)
    start_hex, end_hex = COLOR_STOPS[level]
    return _interpolate_color(start_hex, end_hex, ratio)


def random_comment_for_level(level: str, seed_key: str) -> str:
    rng = random.Random(seed_key)
    return rng.choice(LEVEL_COMMENTS[level])


def build_date_comment(
    target_date: date,
    visitors: int,
    is_weekend: bool,
    is_holiday: bool,
    weather_source: str,
    active_seasons: list[str],
) -> str:
    weekday_name = WEEKDAY_NAMES_KO[target_date.weekday()]
    day_type = "주말" if is_weekend else "평일"
    if is_holiday and not is_weekend:
        day_type = "공휴일"

    season_phrase = (
        f"{', '.join(active_seasons)} 분위기가 살짝 겹쳐서 공원에 생기가 있을 것 같아요."
        if active_seasons
        else "특별한 시즌 느낌은 적어서 차분하게 둘러보기 좋아 보여요."
    )
    level = congestion_level_from_visitors(visitors)
    day_phrase = "주말이라 나들이 발걸음이 조금 더 있을 수 있어요."
    if day_type == "평일":
        day_phrase = "평일이라 비교적 움직이기 편한 날이에요."
    if day_type == "공휴일":
        day_phrase = "공휴일이라 여유 있게 출발하면 더 좋겠어요."

    return (
        f"{weekday_name}인 {target_date:%Y-%m-%d}을 선택하셨네요. "
        f"{day_phrase} {season_phrase} "
        f"전체적으로는 {level} 정도의 분위기를 기대해볼 만해요."
    )
