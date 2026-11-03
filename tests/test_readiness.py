from formulas.pipeline import TestSettings as SimulationSettings
from formulas.readiness import validate_settings
from formulas.readiness import (
    validate_geometry,
    check_readiness,
)
from formulas.mesh import MeshModel

import numpy as np


def test_default_settings_are_valid():
    report = validate_settings(SimulationSettings())

    assert report.can_run
    assert report.errors == []


def test_negative_speed_is_invalid():
    settings = SimulationSettings(speed=-50.0)

    report = validate_settings(settings)

    assert not report.can_run
    assert any("Airspeed" in error for error in report.errors)


def test_zero_model_length_is_invalid():
    settings = SimulationSettings(model_length=0.0)

    report = validate_settings(settings)

    assert not report.can_run


def test_invalid_mesh_detail():
    settings = SimulationSettings(mesh_detail=-10)

    report = validate_settings(settings)

    assert not report.can_run


def test_invalid_line_count():
    settings = SimulationSettings(line_count=0)

    report = validate_settings(settings)

    assert not report.can_run


def test_non_finite_angle():
    settings = SimulationSettings(angle_of_attack=float("nan"))

    report = validate_settings(settings)

    assert not report.can_run


def test_multiple_invalid_settings():
    settings = SimulationSettings(
        speed=-20.0,
        model_length=0.0,
        line_count=-3,
    )

    report = validate_settings(settings)

    assert len(report.errors) == 3
    assert not report.can_run


def test_missing_geometry_is_rejected():
    report = validate_geometry(None)

    assert not report.can_run
    assert report.errors


def test_open_geometry_produces_warning():
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=float)

    model = MeshModel(vertices, [[0, 1, 2]])

    report = validate_geometry(model)

    assert report.can_run
    assert any(
        "boundary" in warning.lower()
        for warning in report.warnings
    )


def test_readiness_combines_errors_and_warnings():
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=float)

    model = MeshModel(vertices, [[0, 1, 2]])

    settings = SimulationSettings(speed=-10.0)

    report = check_readiness(model, settings)

    assert not report.can_run
    assert report.errors
    assert report.warnings