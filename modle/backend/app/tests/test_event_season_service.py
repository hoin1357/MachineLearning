from datetime import date
from pathlib import Path

from app.services.event_season_service import EventSeasonService, classify_event_name


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def test_event_names_map_to_expected_seasons() -> None:
    assert classify_event_name("서울대공원 벚꽃축제 - 2024.04.05 ~ 2024.04.07") == "벚꽃행사시즌"
    assert classify_event_name("어린이날 기념행사 「안녕! 대공원」 - 2024.05.04 ~ 2024.05.06") == "어린이행사시즌"
    assert classify_event_name("서울대공원 영화제 「무비 인사이드」 - 2024.10.05 ~ 2024.10.06") == "야간개장시즌"
    assert classify_event_name("가을축제 「동물원 속 미술관」 - 2018.10.09 ~ 2018.10.21") == "가을축제시즌"


def test_recurring_flags_work_for_future_dates() -> None:
    service = EventSeasonService.from_csv(DATA_DIR / "events.csv")
    flags = service.recurring_flags_for_date(date(2026, 4, 5))
    assert flags["벚꽃행사시즌"] == 1
