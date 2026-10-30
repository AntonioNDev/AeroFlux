from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

EXTENSIONS = (".stl", ".obj", ".ply", ".off")

# Threshold used to identify triangles whose are is
# smaller compared with the overall scale of the mesh.
DEGENERATE_AREA_REL_TOL = 1e-12 


class MeshModel:
    def __init__(self, vertices, faces, name: str = "model", watertight: bool | None = None):
        self.vertices = np.asarray(vertices)
        self.faces = np.asarray(faces)

        self.name = name
        self._watertight = watertight

        structural_errors = self.validate_structure()

        if structural_errors:
            raise ValueError("; ".join(structural_errors))

        self.vertices = self.vertices.astype(float, copy=False)
        self.faces = self.faces.astype(int, copy=False)

        tri = self.vertices[self.faces]

        edge_ab = tri[:, 1] - tri[:, 0]
        edge_ac = tri[:, 2] - tri[:, 0]

        cross = np.cross(edge_ab, edge_ac)
        norm = np.linalg.norm(cross, axis=1)

        self.face_areas = norm / 2.0
        self.face_normals = (cross / np.maximum(norm, np.finfo(float).tiny)[:, None])
        self.face_centroids = tri.mean(axis=1)

        geometry_errors = self.validate_geometry()

        if geometry_errors:
            raise ValueError("; ".join(geometry_errors))

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
        return cls(rocket.vertices, rocket.faces, "demo_rocket")

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
        if not np.isfinite(length) or length <= 0:
            raise ValueError("Target length must be finite and positive.")

        if self.length <= 0:
            raise ValueError("Cannot scale a mesh with zero length.")
        
        low, high = self.bounds
        centred = self.vertices - (low + high) / 2.0
        return MeshModel(centred * (length / self.length), self.faces, self.name, self._watertight)

    def simplified(self, max_faces: int) -> MeshModel:
        if len(self.faces) <= max_faces:
            return self
        mesh = trimesh.Trimesh(self.vertices, self.faces, process=False)
        reduced = mesh.simplify_quadric_decimation(face_count=int(max_faces))
        return MeshModel(reduced.vertices, reduced.faces, self.name, self._watertight)

    def validate_structure(self) -> list[str]:
        """
        Validate the raw mesh arrays before performing geometry
        calculations.

        Returns:
            List of validation errors.
            An empty list means the structure is valid.
        """

        errors = list()

        if self.vertices.ndim != 2:
            errors.append(
                f"Vertices must be a 2D array with shape (N, 3), " f"got {self.vertices.shape}.")

        elif self.vertices.shape[1] != 3:
            errors.append(f"Vertices must have shape (N, 3), " f"got {self.vertices.shape}.")

        elif self.vertices.shape[0] == 0:
            errors.append("Mesh must contain at least one vertex.")

        else:
            if not np.issubdtype(self.vertices.dtype, np.number,):
                errors.append("Vertex coordinates must contain numeric values.")

            elif not np.all(np.isfinite(self.vertices)):
                errors.append("Mesh contains NaN or infinite vertex coordinates.")

        if self.faces.ndim != 2:
            errors.append(
                f"Faces must be a 2D array with shape (N, 3), "
                f"got {self.faces.shape}."
            )

        elif self.faces.shape[1] != 3:
            errors.append(
                f"Faces must contain exactly three vertex indices, "
                f"got shape {self.faces.shape}."
            )

        elif self.faces.shape[0] == 0:
            errors.append(
                "Mesh must contain at least one triangular face."
            )

        elif not np.issubdtype(
            self.faces.dtype,
            np.integer,
        ):
            errors.append(
                "Face indices must be integers."
            )
        elif self.vertices.ndim == 2 and self.vertices.shape[0] > 0:

            if np.any(self.faces < 0):
                errors.append(
                    "Face indices cannot be negative."
                )

            if np.any(self.faces >= len(self.vertices)):
                errors.append(
                    "Mesh contains face indices that reference "
                    "non-existent vertices."
                )

        return errors

    def validate_geometry(self) -> list[str]:
        """
        Validate geometric properties after the mesh structure has
        been confirmed to be valid.

        Returns:
            List of geometry validation errors.
        """

        errors: list[str] = []

        bbox_min = np.min(self.vertices, axis=0)
        bbox_max = np.max(self.vertices, axis=0)

        # Bounding-box diagonal:
        # L = ||max - min||

        characteristic_length = np.linalg.norm(
            bbox_max - bbox_min
        )

        if characteristic_length <= 0:
            errors.append(
                "Mesh has zero spatial extent."
            )

            return errors

        # Area tolerance:
        # epsilon_A = epsilon_relative * L^2

        area_tolerance = (
            DEGENERATE_AREA_REL_TOL
            * characteristic_length**2
        )

        degenerate_faces = np.where(
            self.face_areas <= area_tolerance
        )[0]

        if len(degenerate_faces) > 0:
            errors.append(
                f"Mesh contains {len(degenerate_faces)} "
                f"degenerate triangle(s). "
                f"Face indices: {degenerate_faces.tolist()}"
            )

        return errors

    def validate(self) -> list[str]:
        """
        Run all validation checks that are safe for an initialized
        MeshModel.

        Returns:
            List of validation errors.
            An empty list means no errors were detected.
        """

        errors = self.validate_structure()

        if errors:
            return errors

        errors.extend(self.validate_geometry())

        return errors

    def analyze_edges(self) -> dict:
        """
            Analyze mesh edge connectivity.

            Returns:
                Edge statistics and detected boundary/non-manifold edges.
        """
        
        edges = np.vstack([
            self.faces[:, [0,1]],
            self.faces[:, [1,2]],
            self.faces[:, [2,0]]
        ])

        edges = np.sort(edges, axis=1)

        unique_edges, counts = np.unique(edges, axis=0, return_counts=True)

        boundary_edges = unique_edges[counts == 1]
        interior_edges = unique_edges[counts == 2]
        non_manifold_edges = unique_edges[counts > 2]

        return {
            "total_edges": len(unique_edges),
            "unique_edges": unique_edges,
            "edge_counts": counts,
            "boundary_edges": boundary_edges,
            "interior_edges": interior_edges,
            "non_manifold_edges": non_manifold_edges,
        }

    def connected_components(self) -> list[list[int]]:
        """
            Find connected groups of triangles using shared edges with DFS
            (Depth-First-Search).

            Returns:
                Lists containing face indices for each component.
        """
        edge_to_faces = {}

        for face_id, (a, b, c) in enumerate(self.faces):
            for u, v in ((a, b), (b, c), (c, a)):
                edge = tuple(sorted((int(u), int(v))))
                edge_to_faces.setdefault(edge, []).append(face_id)

        visited = set()
        components = []

        for start_face in range(len(self.faces)):
            if start_face in visited:
                continue

            stack = [start_face]
            component = []

            while stack:
                current = stack.pop()

                if current in visited:
                    continue

                visited.add(current)
                component.append(current)

                a, b, c = self.faces[current]

                for u, v in ((a, b), (b, c), (c, a)):
                    edge = tuple(sorted((int(u), int(v))))

                    neighbors = edge_to_faces.pop(edge, [])

                    for neighbor in neighbors:
                        if neighbor not in visited:
                            stack.append(neighbor)

            components.append(component)

        return components

    def analyze_winding(self) -> dict:
        """
            Detect shared edges with inconsistent face orientation.
        """

        directed_edges = np.stack([
                self.faces[:, [0,1]],
                self.faces[:, [1,2]],
                self.faces[:, [2,0]]
            ],
            axis=1
        )

        edge_groups = {}
        for face_edges in directed_edges:
            for u, v in face_edges:
                directed = (int(u), int(v))
                undirected = tuple(sorted(directed))

                edge_groups.setdefault(undirected, []).append(directed)

        inconsistent_edges = []
        for edge, directions in edge_groups.items():
            if len(directions) != 2:
                continue

            first, second = directions

            if first == second:
                inconsistent_edges.append(edge)

        return {
            "inconsistent_edges": inconsistent_edges,
            "count": len(inconsistent_edges)
        }

    def analyze_quality(self) -> dict:
        """
        Calculate triangle quality metrics.

        Returns:
            Minimum angles, edge aspect ratios,
            and statistics for mesh diagnostics.
        """

        tri = self.vertices[self.faces]

        # edges lengths
        A = tri[:, 0, :]
        B = tri[:, 1, :]
        C = tri[:, 2, :]

        # lengths of edges (a,b,c)
        a = np.linalg.norm(B - A, axis=1)
        b = np.linalg.norm(C - B, axis=1)
        c = np.linalg.norm(A - C, axis=1)

        edges = np.stack([a, b, c], axis=1)

        shortest_edges = np.min(edges, axis=1)
        longest_edges = np.max(edges, axis=1)

        aspect_ratios = np.divide( # We use np.inf to prevent zero-length edges, and devision with Zero
            longest_edges,
            shortest_edges,
            out=np.full_like(longest_edges, np.inf),
            where=shortest_edges > 0
        )

        cos_A = (a**2 + c**2 - b**2) / (2 * a * c)
        cos_B = (a**2 + b**2 - c**2) / (2 * a * b)
        cos_C = (b**2 + c**2 - a**2) / (2 * b * c)

        cos_A = np.clip(cos_A, -1.0, 1.0)
        cos_B = np.clip(cos_B, -1.0, 1.0)
        cos_C = np.clip(cos_C, -1.0, 1.0)

        angle_A = np.degrees(np.arccos(cos_A))
        angle_B = np.degrees(np.arccos(cos_B))
        angle_C = np.degrees(np.arccos(cos_C))

        angles = np.stack(
            [angle_A, angle_B, angle_C],
            axis=1
        )

        minimum_angles = np.min(angles, axis=1)

        MIN_ANGLE_THRESHOLD = 10.0

        poor_quality_faces = np.where(
            minimum_angles < MIN_ANGLE_THRESHOLD
        )[0]

        return {
            "aspect_ratios": aspect_ratios,
            "minimum_angles": minimum_angles,
            "min_angle": float(np.min(minimum_angles)),
            "max_aspect_ratio": float(np.max(aspect_ratios)),
            "mean_aspect_ratio": float(np.mean(aspect_ratios)),
            "poor_quality_faces": poor_quality_faces,
            "poor_quality_count": len(poor_quality_faces),
        }