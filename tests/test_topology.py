import numpy as np

from formulas.mesh import MeshModel


def test_two_triangles_share_one_edge():
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ])

    mesh = MeshModel(vertices, faces)
    result = mesh.analyze_edges()

    assert result["total_edges"] == 5
    assert len(result["boundary_edges"]) == 4
    assert len(result["interior_edges"]) == 1
    assert len(result["non_manifold_edges"]) == 0

def test_consistent_winding():
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
    ])

    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ])

    mesh = MeshModel(vertices, faces)
    result = mesh.analyze_winding()

    assert result["count"] == 0
    assert result["inconsistent_edges"] == []


def test_inconsistent_winding():
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
    ])

    faces = np.array([
        [0, 1, 2],
        [0, 3, 2],
    ])

    mesh = MeshModel(vertices, faces)
    result = mesh.analyze_winding()

    assert result["count"] == 1
    assert result["inconsistent_edges"] == [(0, 2)]

def test_equilateral_triangle_quality():
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, np.sqrt(3) / 2, 0.0],
    ])

    mesh = MeshModel(vertices, [[0, 1, 2]])
    result = mesh.analyze_quality()

    assert np.isclose(result["min_angle"], 60.0)
    assert np.isclose(result["max_aspect_ratio"], 1.0)
    assert result["poor_quality_count"] == 0


def test_right_triangle_quality():
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    mesh = MeshModel(vertices, [[0, 1, 2]])
    result = mesh.analyze_quality()

    assert np.isclose(result["min_angle"], 45.0)
    assert np.isclose(
        result["max_aspect_ratio"],
        np.sqrt(2)
    )


def test_skinny_triangle_quality():
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.001, 0.0],
    ])

    mesh = MeshModel(vertices, [[0, 1, 2]])
    result = mesh.analyze_quality()

    assert result["min_angle"] < 10.0
    assert result["poor_quality_count"] == 1