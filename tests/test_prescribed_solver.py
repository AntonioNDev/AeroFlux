
import numpy as np
import pytest

from formulas.mesh import MeshModel
from formulas.pipeline import TestSettings as SimulationSettings
from formulas.prescribed_solver import PrescribedCpSolver


def make_triangle():
    return MeshModel(
        vertices=np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]),
        faces=np.array([[0, 1, 2]]),
    )


def progress(value, message):
    pass


def test_prescribed_solver_calculates_aerodynamic_loads():
    model = make_triangle()

    settings = SimulationSettings(
        speed=20.0,
        altitude=0.0,
        angle_of_attack=90.0,
    )

    solver = PrescribedCpSolver(
        cp=np.array([-1.0]),
        reference_area_m2=0.5,
    )

    result = solver.run(model, settings, progress)

    assert result.summary["drag_n"] == pytest.approx(
        122.5, rel=1e-3
    )
    assert result.summary["cd"] == pytest.approx(1.0)
    assert result.summary["lift_n"] == pytest.approx(
        0.0, abs=1e-10
    )

    assert result.pressure.shape == (1,)
    assert result.cp.shape == (1,)
    assert result.streamlines.shape == (0, 0, 3)


def test_prescribed_solver_rejects_wrong_cp_shape():
    solver = PrescribedCpSolver(
        cp=np.array([0.2, 0.5]),
        reference_area_m2=0.5,
    )

    with pytest.raises(ValueError, match="Cp array"):
        solver.run(
            make_triangle(),
            SimulationSettings(speed=20.0),
            progress,
        )


def test_prescribed_solver_rejects_zero_speed():
    solver = PrescribedCpSolver(
        cp=np.array([-1.0]),
        reference_area_m2=0.5,
    )

    with pytest.raises(ValueError, match="positive airflow speed"):
        solver.run(
            make_triangle(),
            SimulationSettings(speed=0.0),
            progress,
        )
