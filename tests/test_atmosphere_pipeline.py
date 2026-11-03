
import numpy as np
import pytest

from formulas.mesh import MeshModel
from formulas.pipeline import (
    Solver,
    TestResult as SimulationResults,
    TestSettings as SimulationSettings,
    run_test,
)


class AtmosphereTestSolver(Solver):
    name = "atmosphere-test-solver"

    def run(self, model, settings, progress):
        count = len(model.faces)

        return SimulationResults(
            model=model,
            settings=settings,
            pressure=np.zeros(count),
            cp=np.zeros(count),
            wall_temperature=np.zeros(count),
            streamlines=np.empty((0, 0, 3)),
        )


def test_pipeline_attaches_atmosphere():
    model = MeshModel.demo()
    settings = SimulationSettings(
        altitude=3000.0,
        speed=100.0,
    )

    result = run_test(
        solver=AtmosphereTestSolver(),
        model=model,
        settings=settings,
        progress=lambda value, message: None,
    )

    assert result.atmosphere is not None
    assert result.atmosphere.altitude_m == 3000.0

    assert result.atmosphere.temperature_k == pytest.approx(
        268.7, abs=0.1
    )

    assert result.summary["mach"] == pytest.approx(
        result.atmosphere.mach
    )

    assert result.summary["air_density_kg_m3"] == pytest.approx(
        result.atmosphere.density_kg_m3
    )

    assert result.summary["ambient_pressure_pa"] == pytest.approx(
        result.atmosphere.pressure_pa
    )


def test_pipeline_uses_selected_speed():
    model = MeshModel.demo()

    result_slow = run_test(
        AtmosphereTestSolver(),
        model,
        SimulationSettings(speed=100.0),
        lambda value, message: None,
    )

    result_fast = run_test(
        AtmosphereTestSolver(),
        model,
        SimulationSettings(speed=200.0),
        lambda value, message: None,
    )

    assert result_fast.atmosphere.mach == pytest.approx(
        2.0 * result_slow.atmosphere.mach
    )

    assert result_fast.atmosphere.density_kg_m3 == pytest.approx(
        result_slow.atmosphere.density_kg_m3
    )
