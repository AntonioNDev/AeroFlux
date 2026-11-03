
import numpy as np
import pytest

from formulas.pressure import (
    dynamic_pressure,
    pressure_from_cp,
    cp_from_pressure,
)


def test_dynamic_pressure():
    q = dynamic_pressure(1.225, 20.0)

    assert q == pytest.approx(245.0)


def test_pressure_from_cp():
    pressure = pressure_from_cp(
        cp=np.array([1.0, 0.0, -1.0]),
        ambient_pressure_pa=101325.0,
        density_kg_m3=1.225,
        speed_mps=20.0,
    )

    np.testing.assert_allclose(
        pressure,
        [101570.0, 101325.0, 101080.0],
    )


def test_pressure_cp_roundtrip():
    original_cp = np.array([-0.5, 0.0, 0.8])

    pressure = pressure_from_cp(
        original_cp, 101325.0, 1.225, 20.0
    )

    recovered_cp = cp_from_pressure(
        pressure, 101325.0, 1.225, 20.0
    )

    np.testing.assert_allclose(
        recovered_cp,
        original_cp,
        atol=1e-12,
    )


def test_cp_undefined_at_zero_speed():
    with pytest.raises(ValueError):
        cp_from_pressure(
            np.array([101325.0]),
            101325.0,
            1.225,
            0.0,
        )


def test_zero_speed_produces_ambient_pressure():
    pressure = pressure_from_cp(
        np.array([-1.0, 0.0, 1.0]),
        101325.0,
        1.225,
        0.0,
    )

    np.testing.assert_allclose(
        pressure,
        np.full(3, 101325.0),
    )
