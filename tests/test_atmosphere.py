
import pytest

from formulas.atmosphere import (
    EARTH_RADIUS_M,
    calculate_atmosphere,
)


def geometric_altitude(geopotential_m: float) -> float:
    """Convert ISA geopotential altitude to geometric altitude."""
    return (
        EARTH_RADIUS_M * geopotential_m
        / (EARTH_RADIUS_M - geopotential_m)
    )


def test_sea_level():
    state = calculate_atmosphere(0.0, 0.0)

    assert state.temperature_k == pytest.approx(288.15)
    assert state.pressure_pa == pytest.approx(101325.0)
    assert state.density_kg_m3 == pytest.approx(1.225, rel=1e-3)
    assert state.speed_of_sound_mps == pytest.approx(340.29, rel=1e-3)
    assert state.mach == 0.0


def test_troposphere_5km():
    state = calculate_atmosphere(
        geometric_altitude(5000.0), 200.0
    )

    assert state.temperature_k == pytest.approx(255.65)
    assert state.pressure_pa == pytest.approx(54019.9, rel=1e-3)
    assert state.density_kg_m3 == pytest.approx(0.7361, rel=1e-3)


def test_tropopause_11km():
    state = calculate_atmosphere(
        geometric_altitude(11000.0), 250.0
    )

    assert state.temperature_k == pytest.approx(216.65)
    assert state.pressure_pa == pytest.approx(22632.0, rel=1e-3)


def test_stratosphere_20km():
    state = calculate_atmosphere(
        geometric_altitude(20000.0), 300.0
    )

    assert state.temperature_k == pytest.approx(216.65)
    assert state.pressure_pa == pytest.approx(5474.9, rel=1e-3)


def test_mach_number():
    state = calculate_atmosphere(0.0, 340.29)

    assert state.mach == pytest.approx(1.0, rel=1e-3)


@pytest.mark.parametrize(
    "altitude,speed",
    [
        (-100.0, 100.0),
        (0.0, -20.0),
        (30000.0, 100.0),
        (float("nan"), 100.0),
        (0.0, float("inf")),
    ],
)
def test_invalid_inputs(altitude, speed):
    with pytest.raises(ValueError):
        calculate_atmosphere(altitude, speed)

def test_isa_upper_boundary():
    maximum = geometric_altitude(20_000.0)

    state = calculate_atmosphere(maximum, 120.0)

    assert state.temperature_k == pytest.approx(216.65)
    assert state.pressure_pa == pytest.approx(5474.9, rel=1e-3)

    with pytest.raises(ValueError):
        calculate_atmosphere(maximum + 10.0, 120.0)