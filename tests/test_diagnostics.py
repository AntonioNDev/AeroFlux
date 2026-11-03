import numpy as np

from formulas.mesh import MeshModel
from formulas.diagnostics import analyze_mesh


def test_closed_mesh_diagnostics():
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=float)

    faces = np.array([
        [0, 2, 1],
        [0, 1, 3],
        [0, 3, 2],
        [1, 2, 3],
    ])

    model = MeshModel(vertices, faces)

    report = analyze_mesh(model)

    assert report.vertices == 4
    assert report.faces == 4
    assert report.connected_components == 1
    assert report.boundary_edges == 0
    assert report.non_manifold_edges == 0
    assert report.inconsistent_edges == 0
    assert report.poor_quality_faces == 0
    assert report.warnings == []

def test_open_mesh_diagnostics():
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=float)

    faces = np.array([[0, 1, 2]])

    model = MeshModel(vertices, faces)

    report = analyze_mesh(model)

    assert report.boundary_edges == 3
    assert len(report.warnings) == 1
    assert "boundary" in report.warnings[0].lower()

def test_poor_quality_mesh_diagnostics():
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.001, 0.0],
    ])

    faces = np.array([[0, 1, 2]])

    model = MeshModel(vertices, faces)

    report = analyze_mesh(model)

    assert report.poor_quality_faces == 1
    assert report.minimum_angle < 10.0
    assert any(
        "quality" in warning.lower()
        or "angle" in warning.lower()
        for warning in report.warnings
    )

