import numpy as np

from formulas.mesh import MeshModel


def test_disconnected_components():
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [5, 0, 0],
        [6, 0, 0],
        [5, 1, 0],
    ])

    faces = np.array([
        [0, 1, 2],
        [3, 4, 5],
    ])

    mesh = MeshModel(vertices, faces)

    components = mesh.connected_components()

    assert len(components) == 2
    assert {frozenset(c) for c in components} == {
        frozenset({0}),
        frozenset({1}),
    }