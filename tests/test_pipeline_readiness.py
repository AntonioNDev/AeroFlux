import pytest

from formulas.mesh import MeshModel
from formulas.pipeline import (
    Solver,
    TestSettings as SimulationSettings,
    run_test,
)

class TrackingSolver(Solver):
    name = "tracking"

    def __init__(self):
        self.called = False

    def run(self, model, settings, progress):
        self.called = True
        return None


def test_pipeline_rejects_invalid_settings():
    solver = TrackingSolver()
    model = MeshModel.demo()
    settings = SimulationSettings(speed=-100.0)

    with pytest.raises(ValueError, match="Airspeed"):
        run_test(
            solver,
            model,
            settings,
            lambda fraction, message: None,
        )

    assert not solver.called


def test_pipeline_rejects_missing_mesh():
    solver = TrackingSolver()
    settings = SimulationSettings()

    with pytest.raises(ValueError, match="MeshModel"):
        run_test(
            solver,
            None,
            settings,
            lambda fraction, message: None,
        )

    assert not solver.called