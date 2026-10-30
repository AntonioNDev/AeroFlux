from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

EXTENSIONS = (".stl", ".obj", ".ply", ".off")


class MeshModel:
    def __init__(self, vertices, faces, name: str = "model", watertight: bool | None = None):
        self.vertices = np.asarray(vertices, dtype=float)
        self.faces = np.asarray(faces, dtype=int)
        self.name = name
        self._watertight = watertight

        tri = self.vertices[self.faces]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        norm = np.linalg.norm(cross, axis=1)
        self.face_areas = norm / 2.0
        self.face_normals = cross / np.maximum(norm, 1e-300)[:, None]
        self.face_centroids = tri.mean(axis=1)

    @classmethod
    def from_file(cls, path) -> MeshModel:
        path = Path(path)
        if path.suffix.lower() not in EXTENSIONS:
            raise ValueError(f"Unsupported file type {path.suffix!r}. Use {', '.join(EXTENSIONS)}")
        mesh = trimesh.load(path, force="mesh")
        if len(mesh.faces) == 0:
            raise ValueError(f"{path.name} has no triangles")
        watertight = bool(mesh.is_watertight)
        if watertight:
            mesh.fix_normals()
        return cls(mesh.vertices, mesh.faces, path.stem, watertight)

    @classmethod
    def demo(cls) -> MeshModel:
        body = trimesh.creation.cylinder(radius=0.25, height=2.0, sections=48)
        nose = trimesh.creation.cone(radius=0.25, height=0.8, sections=48)
        nose.apply_translation([0, 0, 1.0])
        rocket = trimesh.util.concatenate([body, nose])
        rocket.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 1, 0]))
        return cls(rocket.vertices, rocket.faces, "demo_rocket", True)

    @property
    def bounds(self) -> np.ndarray:
        return np.array([self.vertices.min(axis=0), self.vertices.max(axis=0)])

    @property
    def extent(self) -> np.ndarray:
        low, high = self.bounds
        return high - low

    @property
    def length(self) -> float:
        return float(self.extent.max())

    @property
    def is_watertight(self) -> bool:
        if self._watertight is None:
            self._watertight = bool(trimesh.Trimesh(self.vertices, self.faces, process=False).is_watertight)
        return self._watertight

    def scaled_to(self, length: float) -> MeshModel:
        low, high = self.bounds
        centred = self.vertices - (low + high) / 2.0
        return MeshModel(centred * (length / self.length), self.faces, self.name, self._watertight)

    def simplified(self, max_faces: int) -> MeshModel:
        if len(self.faces) <= max_faces:
            return self
        mesh = trimesh.Trimesh(self.vertices, self.faces, process=False)
        reduced = mesh.simplify_quadric_decimation(face_count=int(max_faces))
        return MeshModel(reduced.vertices, reduced.faces, self.name, self._watertight)
