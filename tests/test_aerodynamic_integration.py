import numpy as np
import pytest

from formulas.atmosphere import calculate_atmosphere
from formulas.mesh import MeshModel
from formulas.pressure import pressure_from_cp
from formulas.forces import (
    integrate_pressure_forces,
    calculate_aerodynamic_loads,
)


def make_test_triangle():
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    faces = np.array([[0, 1, 2]])

    return MeshModel(vertices, faces)


def test_complete_aerodynamic_calculation():
    model = make_test_triangle()

    atmosphere = calculate_atmosphere(
        altitude_m=0.0,
        speed_mps=20.0,
    )

    # Prescribed test coefficient, NOT a predicted Cp.
    cp = np.array([-1.0])

    pressure = pressure_from_cp(
        cp=cp,
        ambient_pressure_pa=atmosphere.pressure_pa,
        density_kg_m3=atmosphere.density_kg_m3,
        speed_mps=20.0,
    )

    forces = integrate_pressure_forces(
        model=model,
        pressure_pa=pressure,
        ambient_pressure_pa=atmosphere.pressure_pa,
    )

    loads = calculate_aerodynamic_loads(
        total_force_n=forces.total_force_n,
        flow_direction=np.array([0.0, 0.0, 1.0]),
        density_kg_m3=atmosphere.density_kg_m3,
        speed_mps=20.0,
        reference_area_m2=0.5,
        reference_up=np.array([0.0, 1.0, 0.0]),
    )

    # q ≈ 245 Pa, triangle area = 0.5 m²
    # Cp = -1 produces +122.5 N along +Z.
    assert loads.drag_n == pytest.approx(122.5, rel=1e-3)
    assert loads.lift_n == pytest.approx(0.0, abs=1e-10)
    assert loads.cd == pytest.approx(1.0, rel=1e-10)


def test_force_scales_with_speed_squared():
    model = make_test_triangle()

    def calculate_force(speed):
        atmosphere = calculate_atmosphere(0.0, speed)

        pressure = pressure_from_cp(
            cp=np.array([-1.0]),
            ambient_pressure_pa=atmosphere.pressure_pa,
            density_kg_m3=atmosphere.density_kg_m3,
            speed_mps=speed,
        )

        return integrate_pressure_forces(
            model,
            pressure,
            atmosphere.pressure_pa,
        ).total_force_n

    slow = calculate_force(20.0)
    fast = calculate_force(40.0)

    np.testing.assert_allclose(
        fast,
        4.0 * slow,
        rtol=1e-10,
        atol=1e-10,
    )
