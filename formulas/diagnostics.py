from dataclasses import dataclass
from formulas.mesh import MeshModel


@dataclass
class MeshDiagnostics:
    vertices: int
    faces: int
    connected_components: int
    boundary_edges: int
    non_manifold_edges: int
    inconsistent_edges: int
    minimum_angle: float
    maximum_aspect_ratio: float
    poor_quality_faces: int
    warnings: list[str]

def analyze_mesh(model: MeshModel) -> MeshDiagnostics:
    """
    Generates a diagnostic report for the imported 3D Mesh.

    Return:
        warnings: list of warnings
    """

    edges = model.analyze_edges()
    components = model.connected_components()
    winding = model.analyze_winding()
    quality = model.analyze_quality()

    warnings = []

    if len(edges["boundary_edges"]) > 0:
        warnings.append(f"Mesh contains {len(edges['boundary_edges'])} boundary edges.")

    if len(edges["non_manifold_edges"]) > 0:
        warnings.append(
            f"Mesh contains {len(edges['non_manifold_edges'])} "
            "non-manifold edges."
        )

    if winding["count"] > 0:
        warnings.append(
            f"Mesh contains {winding['count']} "
            "inconsistent winding edges."
        )

    if quality["poor_quality_count"] > 0:
        warnings.append(
            f"Mesh contains {quality['poor_quality_count']} "
            "triangles with minimum angles below 10 degrees."
        )

    return MeshDiagnostics(
        vertices=len(model.vertices),
        faces=len(model.faces),
        connected_components=len(components),
        boundary_edges=len(edges["boundary_edges"]),
        non_manifold_edges=len(edges["non_manifold_edges"]),
        inconsistent_edges=winding["count"],
        minimum_angle=quality["min_angle"],
        maximum_aspect_ratio=quality["max_aspect_ratio"],
        poor_quality_faces=quality["poor_quality_count"],
        warnings=warnings,
    )