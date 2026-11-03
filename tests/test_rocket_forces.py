
import numpy as np
import pytest

from formulas.mesh import MeshModel
from formulas.pipeline import (TestSettings as SimulationSettings, run_test)
from formulas.prescribed_solver import PrescribedCpSolver
from formulas.atmosphere import calculate_atmosphere

def progress(value, message):
    pass

def test_demo_rocket_pressure_forces():
    """Verify pressure loads on the demo rocket."""

    model = MeshModel.demo()

    settings = SimulationSettings(
        speed=100.0,
        altitude=1000.0,
        angle_of_attack=0.0,
    )

    cp = np.zeros(len(model.faces))

    face_index = int(
        np.argmax(np.abs(model.face_normals @ settings.wind))
    )
    cp[face_index] = -0.5

    solver = PrescribedCpSolver(
        cp=cp,
        reference_area_m2=1.0,
    )

    result = solver.run(model, settings, progress)

    assert len(result.pressure) == len(model.faces)
    assert len(result.cp) == len(model.faces)

    assert np.all(np.isfinite(result.pressure))
    assert np.isfinite(result.summary["drag_n"])
    assert np.isfinite(result.summary["lift_n"])
    assert np.isfinite(result.summary["cd"])

    atmosphere = calculate_atmosphere(
        settings.altitude, settings.speed
    )

    q = 0.5 * atmosphere.density_kg_m3 * settings.speed**2

    expected_force = (
        -cp[face_index]
        * q
        * model.face_areas[face_index]
        * model.face_normals[face_index]
    )

    expected_drag = np.dot(expected_force, settings.wind)

    assert result.summary["drag_n"] == pytest.approx(expected_drag)

def test_rocket_solver_pipeline():
    """Verify integration with run_test()."""

    model = MeshModel.demo()

    settings = SimulationSettings(
        speed=120.0,
        altitude=2000.0,
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
        progress=progress,
    )

    assert result.solver == solver.name
    assert len(result.pressure) == len(result.model.faces)
    assert len(result.cp) == len(result.model.faces)

    assert result.summary["drag_n"] == pytest.approx(
        0.0, abs=1e-8
    )
    assert result.summary["lift_n"] == pytest.approx(
        0.0, abs=1e-8
    )
    assert result.summary["cd"] == pytest.approx(
        0.0, abs=1e-8
    )

    assert result.elapsed >= 0.0
