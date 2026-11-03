from dataclasses import dataclass
from .mesh import MeshModel

import numpy as np

@dataclass(frozen=True, slots=True)
class PressureForces:
    """Pressure forces on a triangulated surface."""

    face_forces_n: np.ndarray
    total_force_n: np.ndarray

@dataclass(frozen=True, slots=True)
class AerodynamicLoads:
    drag_n: float
    lift_n: float
    side_force_n: float
    cd: float
    cl: float
    cy: float

def integrate_pressure_forces(model: MeshModel, pressure_pa: np.ndarray, ambient_pressure_pa: float) -> PressureForces:
    """
        Pressure forces.

        Must:
            Pressure is specified once per triangle,
            Face normals point outward,
            Pressure acts uniformly over each triangle,
            Pressure represents loading on the modeled surface
    """
    pressure = np.asarray(pressure_pa, dtype=float)

    if pressure.shape != (len(model.faces),):
        raise ValueError("Pressure array must contain one value per face.")

    if not np.all(np.isfinite(pressure)):
        raise ValueError("Pressure values must be finite.")

    if not np.isfinite(ambient_pressure_pa):
        raise ValueError("Ambient pressure must be finite.")

    # delta_pressure - pressure difference relative to the surrounding air
    delta_pressure = pressure - ambient_pressure_pa

    # Pressure force on each triangle
    face_forces = (-delta_pressure[:, None] * model.face_areas[:, None] * model.face_normals)

    total_force = np.sum(face_forces, axis=0)

    return PressureForces(face_forces_n=face_forces, total_force_n=total_force)

def calculate_aerodynamic_loads(total_force_n: np.ndarray, flow_direction: np.ndarray, density_kg_m3: float, speed_mps:float,
                                reference_area_m2: float, reference_up: np.ndarray | None = None) -> AerodynamicLoads:
    
    """Project total force onto aerodynamic axes."""

    force = np.asarray(total_force_n, dtype=float)
    flow = np.asarray(flow_direction, dtype=float)

    if reference_up is None:
        reference_up = np.array([0.0, 0.0, 1.0])

    up = np.asarray(reference_up, dtype=float)

    if force.shape != (3,) or flow.shape != (3,) or up.shape != (3,):
        raise ValueError("Force and direction vectors must have shape (3,).")

    if not (
        np.all(np.isfinite(force))
        and np.all(np.isfinite(flow))
        and np.all(np.isfinite(up))
    ):
        raise ValueError("Force and direction vectors must be finite.")

    if not all(
        np.isfinite(value)
        for value in (density_kg_m3, speed_mps, reference_area_m2)
    ):
        raise ValueError("Aerodynamic parameters must be finite.")

    if density_kg_m3 <= 0 or speed_mps <= 0 or reference_area_m2 <= 0:
        raise ValueError(
            "Density, speed, and reference area must be positive."
        )

    flow_norm = np.linalg.norm(flow)

    if flow_norm == 0:
        raise ValueError("Flow direction cannot be zero.")

    drag_axis = flow / flow_norm

    # Remove any component of 'up' parallel to the flow.
    lift_axis = up - np.dot(up, drag_axis) * drag_axis
    lift_norm = np.linalg.norm(lift_axis)

    if lift_norm < 1e-10:
        raise ValueError(
            "Reference-up direction cannot be parallel to airflow."
        )

    lift_axis /= lift_norm

    # Right-handed aerodynamic coordinate system.
    side_axis = np.cross(lift_axis, drag_axis)

    drag = float(np.dot(force, drag_axis))
    lift = float(np.dot(force, lift_axis))
    side = float(np.dot(force, side_axis))

    dynamic_pressure = 0.5 * density_kg_m3 * speed_mps**2
    denominator = dynamic_pressure * reference_area_m2

    return AerodynamicLoads(
        drag_n=drag,
        lift_n=lift,
        side_force_n=side,
        cd=drag / denominator,
        cl=lift / denominator,
        cy=side / denominator,
    )