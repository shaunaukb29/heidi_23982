from carithm_inspector.domain import DamageType, Detection
from carithm_inspector.estimates import estimate


def test_broken_lamp_is_not_marked_driveable() -> None:
    result = estimate(Detection(DamageType.LAMP_BROKEN, 0.84, (0, 0, 100, 100), 1000, 800))
    assert result.driveable is False
    assert result.low_usd < result.high_usd


def test_larger_scratch_has_a_higher_estimate() -> None:
    small = estimate(Detection(DamageType.SCRATCH, 0.9, (0, 0, 20, 20), 1000, 800))
    large = estimate(Detection(DamageType.SCRATCH, 0.9, (0, 0, 300, 300), 1000, 800))
    assert large.low_usd > small.low_usd
