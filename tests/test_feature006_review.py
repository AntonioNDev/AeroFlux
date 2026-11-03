
import numpy as np
import pytest

from formulas.mesh import MeshModel
from formulas.pipeline import TestSettings as SimulationSettings, run_test
from formulas.prescribed_solver import PrescribedCpSolver
from formulas.placeholder import PlaceholderSolver


@pytest.mark.parametrize(
    "angle_of_attack,sideslip",
    [
        (0.0, 0.0),
        (45.0, 0.0),
        (90.0, 0.0),
        (-90.0, 0.0),
        (0.0, 90.0),
    ],
)
def test_prescribed_solver_supports_flow_orientations(
    angle_of_attack, sideslip
):
    model = MeshModel.demo()

    settings = SimulationSettings(
        speed=100.0,
        angle_of_attack=angle_of_attack,
        sideslip=sideslip,
    )

    solver = PrescribedCpSolver(
        cp=np.zeros(len(model.faces)),
        reference_area_m2=1.0,
    )

    result = solver.run(
        model,
        settings,
        lambda value, message: None,
    )

    for key in ("drag_n", "lift_n", "side_force_n", "cd", "cl", "cy"):
        assert np.isfinite(result.summary[key])
        assert result.summary[key] == pytest.approx(0.0, abs=1e-8)


def test_prescribed_solver_reports_its_limitations():
    model = MeshModel.demo()

    solver = PrescribedCpSolver(
        cp=np.zeros(len(model.faces)),
        reference_area_m2=1.0,
    )

    result = solver.run(
        model,
        SimulationSettings(speed=100.0),
        lambda value, message: None,
    )

    assert solver.placeholder is True
    assert result.streamlines.size == 0
    assert any("supplied" in msg.lower() for msg in result.advice)
    assert any("not calculated" in msg.lower() for msg in result.advice)


def test_placeholder_solver_remains_compatible():
    model = MeshModel.demo()

    settings = SimulationSettings(
        speed=100.0,
        altitude=1000.0,
        angle_of_attack=0.0,
    )

    result = run_test(
        solver=PlaceholderSolver(),
        model=model,
        settings=settings,
        progress=lambda value, message: None,
    )

    assert result.solver == PlaceholderSolver.name

    assert result.pressure.shape == (len(result.model.faces),)
    assert result.cp.shape == (len(result.model.faces),)
    assert result.wall_temperature.shape == (len(result.model.faces),)

    assert result.streamlines.ndim == 3
    assert result.streamlines.shape[0] > 0
    assert result.streamlines.shape[2] == 3

    assert np.all(np.isfinite(result.pressure))
    assert np.all(np.isfinite(result.streamlines))
