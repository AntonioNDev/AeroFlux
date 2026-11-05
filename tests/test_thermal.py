import numpy as np
import pytest

from formulas.mesh import MeshModel
from formulas.pipeline import TestSettings as ResultSettings, run_test
from formulas.prescribed_solver import PrescribedCpSolver
from formulas.atmosphere import calculate_atmosphere
from formulas.thermal import stagnation_temperature, recovery_factor, adiabatic_wall_temperature, convective_heat_flux, convective_heat_transfer_rate, calculate_thermal_references


def test_stagnation_temperature_at_zero_mach():
    assert stagnation_temperature(288.15, 0.0) == pytest.approx(288.15)


def test_stagnation_temperature_at_mach_one():
    assert stagnation_temperature(300.0, 1.0) == pytest.approx(360.0)


def test_stagnation_temperature_at_mach_two():
    assert stagnation_temperature(300.0, 2.0) == pytest.approx(540.0)


def test_stagnation_temperature_with_atmosphere():
    atmosphere = calculate_atmosphere(
        altitude_m=5000.0,
        speed_mps=300.0,
    )

    result = stagnation_temperature(
        atmosphere.temperature_k,
        atmosphere.mach,
    )

    assert result > atmosphere.temperature_k

@pytest.mark.parametrize(
    "temperature,mach,gamma",
    [
        (0.0, 1.0, 1.4),
        (-10.0, 1.0, 1.4),
        (300.0, -1.0, 1.4),
        (300.0, 1.0, 1.0),
        (float("nan"), 1.0, 1.4),
        (300.0, float("inf"), 1.4),
    ],
)
def test_stagnation_temperature_rejects_invalid_inputs(
    temperature, mach, gamma
):
    with pytest.raises(ValueError):
        stagnation_temperature(temperature, mach, gamma)


def test_laminar_recovery_factor():
    result = recovery_factor(0.71, "laminar")
    assert result == pytest.approx(0.71**0.5)


def test_turbulent_recovery_factor():
    result = recovery_factor(0.71, "turbulent")
    assert result == pytest.approx(0.71 ** (1.0 / 3.0))


def test_adiabatic_wall_temperature_zero_mach():
    result = adiabatic_wall_temperature(
        ambient_temperature_k=300.0,
        mach=0.0,
        recovery=0.9,
    )
    assert result == pytest.approx(300.0)


def test_adiabatic_wall_temperature_mach_two():
    result = adiabatic_wall_temperature(
        ambient_temperature_k=300.0,
        mach=2.0,
        recovery=0.9,
    )
    assert result == pytest.approx(516.0)


def test_recovery_temperature_below_stagnation_temperature():
    ambient = 300.0
    mach = 2.0
    recovery = recovery_factor(0.71, "turbulent")

    wall = adiabatic_wall_temperature(
        ambient, mach, recovery
    )
    stagnation = stagnation_temperature(
        ambient, mach
    )

    assert ambient < wall < stagnation


@pytest.mark.parametrize(
    "prandtl,boundary_layer",
    [
        (0.0, "laminar"),
        (-0.5, "turbulent"),
        (float("nan"), "laminar"),
        (0.71, "invalid"),
    ],
)

def test_recovery_factor_invalid_inputs(prandtl, boundary_layer):
    with pytest.raises(ValueError):
        recovery_factor(prandtl, boundary_layer)

@pytest.mark.parametrize(
    "ambient,mach,recovery,gamma",
    [
        (0.0, 1.0, 0.9, 1.4),
        (300.0, -1.0, 0.9, 1.4),
        (300.0, 1.0, -0.1, 1.4),
        (300.0, 1.0, 1.1, 1.4),
        (300.0, 1.0, 0.9, 1.0),
        (300.0, 1.0, float("nan"), 1.4),
    ],
)
def test_adiabatic_wall_temperature_invalid_inputs(
    ambient, mach, recovery, gamma
):
    with pytest.raises(ValueError):
        adiabatic_wall_temperature(
            ambient, mach, recovery, gamma
        )


def test_convective_heat_flux_heating():
    result = convective_heat_flux(
        heat_transfer_coefficient=50.0,
        fluid_temperature_k=350.0,
        surface_temperature_k=300.0,
    )

    assert result == pytest.approx(2500.0)


def test_convective_heat_flux_cooling():
    result = convective_heat_flux(
        heat_transfer_coefficient=50.0,
        fluid_temperature_k=280.0,
        surface_temperature_k=300.0,
    )

    assert result == pytest.approx(-1000.0)


def test_convective_heat_flux_equal_temperatures():
    result = convective_heat_flux(50.0, 300.0, 300.0)

    assert result == pytest.approx(0.0)


def test_convective_heat_flux_zero_coefficient():
    result = convective_heat_flux(0.0, 350.0, 300.0)

    assert result == pytest.approx(0.0)


def test_convective_heat_transfer_rate():
    result = convective_heat_transfer_rate(
        heat_flux_w_m2=2500.0,
        surface_area_m2=2.0,
    )

    assert result == pytest.approx(5000.0)


def test_convective_heat_transfer_rate_cooling():
    result = convective_heat_transfer_rate(-1000.0, 2.0)

    assert result == pytest.approx(-2000.0)


def test_convective_heat_transfer_rate_zero_area():
    result = convective_heat_transfer_rate(2500.0, 0.0)

    assert result == pytest.approx(0.0)


@pytest.mark.parametrize(
    "h,fluid,surface",
    [
        (-1.0, 300.0, 290.0),
        (float("nan"), 300.0, 290.0),
        (50.0, 0.0, 290.0),
        (50.0, 300.0, -10.0),
        (50.0, float("inf"), 290.0),
        (50.0, 300.0, float("nan")),
    ],
)
def test_convective_heat_flux_invalid_inputs(h, fluid, surface):
    with pytest.raises(ValueError):
        convective_heat_flux(h, fluid, surface)


@pytest.mark.parametrize(
    "heat_flux,area",
    [
        (float("nan"), 2.0),
        (float("inf"), 2.0),
        (2500.0, -1.0),
        (2500.0, float("nan")),
    ],
)
def test_convective_heat_transfer_rate_invalid_inputs(
    heat_flux, area
):
    with pytest.raises(ValueError):
        convective_heat_transfer_rate(heat_flux, area)

from formulas.thermal import calculate_thermal_references


def test_thermal_references_at_zero_speed():
    atmosphere = calculate_atmosphere(
        altitude_m=0.0,
        speed_mps=0.0,
    )

    result = calculate_thermal_references(atmosphere)

    assert result["stagnation_temperature_k"] == pytest.approx(
        atmosphere.temperature_k
    )
    assert result["adiabatic_wall_reference_k"] == pytest.approx(
        atmosphere.temperature_k
    )


def test_thermal_references_at_positive_speed():
    atmosphere = calculate_atmosphere(
        altitude_m=1000.0,
        speed_mps=200.0,
    )

    result = calculate_thermal_references(atmosphere)

    assert (
        atmosphere.temperature_k
        < result["adiabatic_wall_reference_k"]
        < result["stagnation_temperature_k"]
    )


def test_pipeline_includes_thermal_references():
    model = MeshModel.demo()

    settings = ResultSettings(
        speed=200.0,
        altitude=1000.0,
        model_length=4.0,
        mesh_detail=1200,
    )

    prepared = model.scaled_to(
        settings.model_length
    ).simplified(settings.mesh_detail)

    cp = np.zeros(len(prepared.faces))

    solver = PrescribedCpSolver(
        cp=cp,
        reference_area_m2=1.0,
    )

    result = run_test(
        solver=solver,
        model=model,
        settings=settings,
        progress=lambda fraction, message: None,
    )

    atmosphere = result.atmosphere
    assert atmosphere is not None

    expected_stagnation = stagnation_temperature(
        atmosphere.temperature_k,
        atmosphere.mach,
    )

    assert result.summary[
        "stagnation_temperature_k"
    ] == pytest.approx(expected_stagnation)

    expected_recovery = recovery_factor(
        0.71,
        "turbulent",
    )

    assert result.summary[
        "recovery_factor"
    ] == pytest.approx(expected_recovery)

    expected_wall_reference = adiabatic_wall_temperature(
        atmosphere.temperature_k,
        atmosphere.mach,
        expected_recovery,
    )

    assert result.summary[
        "adiabatic_wall_reference_k"
    ] == pytest.approx(expected_wall_reference)
