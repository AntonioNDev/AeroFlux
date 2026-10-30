"""Tests for AeroFlux Feature #001: mesh validation."""

import numpy as np
import pytest
import trimesh

from formulas.mesh import MeshModel


@pytest.fixture
def triangle():
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    faces = np.array([[0, 1, 2]], dtype=int)
    return vertices, faces


@pytest.fixture
def tetrahedron():
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
    ], dtype=int)
    return vertices, faces


def test_valid_triangle_geometry(triangle):
    vertices, faces = triangle
    mesh = MeshModel(vertices, faces)
    assert mesh.validate() == []
    assert mesh.face_areas[0] == pytest.approx(0.5)
    np.testing.assert_allclose(mesh.face_centroids[0], [1 / 3, 1 / 3, 0])
    np.testing.assert_allclose(mesh.face_normals[0], [0, 0, 1])


def test_valid_closed_tetrahedron(tetrahedron):
    mesh = MeshModel(*tetrahedron)
    assert mesh.validate() == []
    assert mesh.is_watertight
    assert len(mesh.face_areas) == 4
    np.testing.assert_allclose(np.linalg.norm(mesh.face_normals, axis=1), 1.0)


@pytest.mark.parametrize("vertices,faces,match", [
    ([], [[0, 1, 2]], "[Vv]ertices"),
    ([[0, 0, 0]], [], "[Ff]aces"),
    ([[0, 0], [1, 0], [0, 1]], [[0, 1, 2]], "[Vv]ertices"),
    ([0, 1, 2], [[0, 1, 2]], "[Vv]ertices"),
    ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [0, 1, 2], "[Ff]aces"),
    ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2, 0]], "[Ff]aces"),
    ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 99]], "[Ff]ace"),
    ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[-1, 1, 2]], "[Ff]ace"),
    ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0.0, 1.5, 2.0]], "[Ff]ace"),
    ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [["0", "1", "2"]], "[Ff]ace"),
    ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 1]], "degenerate"),
    ([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 1, 2]], "degenerate"),
    ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 1, 2]], "zero spatial extent"),
])
def test_invalid_mesh_is_rejected(vertices, faces, match):
    with pytest.raises(ValueError, match=match):
        MeshModel(vertices, faces)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_coordinates_rejected(triangle, bad_value):
    vertices, faces = triangle
    vertices[0, 0] = bad_value
    with pytest.raises(ValueError, match="NaN or infinite"):
        MeshModel(vertices, faces)


def test_tiny_valid_triangle_is_scale_invariant(triangle):
    vertices, faces = triangle
    mesh = MeshModel(vertices * 1e-6, faces)
    assert mesh.validate() == []
    assert mesh.face_areas[0] == pytest.approx(0.5e-12)


def test_bounds_extent_and_length(triangle):
    mesh = MeshModel(*triangle)
    np.testing.assert_allclose(mesh.bounds, [[0, 0, 0], [1, 1, 0]])
    np.testing.assert_allclose(mesh.extent, [1, 1, 0])
    assert mesh.length == pytest.approx(1)

