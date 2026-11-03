
import numpy as np
import pytest

from formulas.mesh import MeshModel
from formulas.forces import integrate_pressure_forces, calculate_aerodynamic_loads


def test_pressure_force_on_single_triangle():
    # Triangle area = 0.5 m², normal = +Z.
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    faces = np.array([[0, 1, 2]])
    model = MeshModel(vertices, faces)

    result = integrate_pressure_forces(
        model=model,
        pressure_pa=np.array([110.0]),
        ambient_pressure_pa=100.0,
    )

    # Force = -(110 - 100) * 0.5 * [0, 0, 1]
    np.testing.assert_allclose(
        result.total_force_n,
        [0.0, 0.0, -5.0],
        atol=1e-12,
    )


def test_uniform_pressure_on_closed_tetrahedron():
    # A closed tetrahedron with outward-facing normals.
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    faces = np.array([
        [0, 2, 1],
        [0, 1, 3],
        [0, 3, 2],
        [1, 2, 3],
    ])

    model = MeshModel(vertices, faces)

    result = integrate_pressure_forces(
        model=model,
        pressure_pa=np.full(4, 120000.0),
        ambient_pressure_pa=101325.0,
    )

    # Uniform pressure on a closed surface
    # produces zero net force.
    np.testing.assert_allclose(
        result.total_force_n,
        np.zeros(3),
        atol=1e-9,
    )


def test_invalid_pressure_array():
    model = MeshModel.demo()

    with pytest.raises(ValueError):
        integrate_pressure_forces(
            model,
            pressure_pa=np.array([100.0]),
            ambient_pressure_pa=101325.0,
        )


def test_nonfinite_pressure():
    model = MeshModel.demo()

    pressure = np.full(len(model.faces), 101325.0)
    pressure[0] = np.nan

    with pytest.raises(ValueError):
        integrate_pressure_forces(
            model,
            pressure,
            ambient_pressure_pa=101325.0,
        )

def test_drag_and_lift_projection():
    result = calculate_aerodynamic_loads(
        total_force_n=np.array([100.0, 0.0, 50.0]),
        flow_direction=np.array([1.0, 0.0, 0.0]),
        density_kg_m3=1.225,
        speed_mps=20.0,
        reference_area_m2=2.0,
    )

    assert result.drag_n == pytest.approx(100.0)
    assert result.lift_n == pytest.approx(50.0)
    assert result.side_force_n == pytest.approx(0.0)
    assert result.cd == pytest.approx(100.0 / 490.0)
    assert result.cl == pytest.approx(50.0 / 490.0)


def test_rotated_flow_direction():
    result = calculate_aerodynamic_loads(
        total_force_n=np.array([0.0, 100.0, 0.0]),
        flow_direction=np.array([0.0, 1.0, 0.0]),
        density_kg_m3=1.225,
        speed_mps=20.0,
        reference_area_m2=2.0,
    )

    assert result.drag_n == pytest.approx(100.0)
    assert result.lift_n == pytest.approx(0.0)


def test_invalid_reference_area():
    with pytest.raises(ValueError):
        calculate_aerodynamic_loads(
            total_force_n=np.array([100.0, 0.0, 0.0]),
            flow_direction=np.array([1.0, 0.0, 0.0]),
            density_kg_m3=1.225,
            speed_mps=20.0,
            reference_area_m2=0.0,
        )


def test_parallel_reference_up():
    with pytest.raises(ValueError):
        calculate_aerodynamic_loads(
            total_force_n=np.array([100.0, 0.0, 0.0]),
            flow_direction=np.array([1.0, 0.0, 0.0]),
            density_kg_m3=1.225,
            speed_mps=20.0,
            reference_area_m2=2.0,
            reference_up=np.array([1.0, 0.0, 0.0]),
        )

