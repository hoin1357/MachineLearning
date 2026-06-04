from app.domain.commentary import color_for_prediction, congestion_level_from_visitors


def test_congestion_thresholds_follow_spec() -> None:
    assert congestion_level_from_visitors(2999) == "한산함"
    assert congestion_level_from_visitors(3000) == "보통"
    assert congestion_level_from_visitors(7999) == "보통"
    assert congestion_level_from_visitors(8000) == "붐빔"
    assert congestion_level_from_visitors(14999) == "붐빔"
    assert congestion_level_from_visitors(15000) == "매우 붐빔"


def test_color_gets_deeper_inside_same_band() -> None:
    assert color_for_prediction(6500) != color_for_prediction(10000)
