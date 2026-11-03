
import numpy as np


def dynamic_pressure(
    density_kg_m3: float,
    speed_mps: float,
) -> float:
    """Calculate dynamic pressure in pascals."""

    if not np.isfinite(density_kg_m3) or not np.isfinite(speed_mps):
        raise ValueError("Density and speed must be finite.")

    if density_kg_m3 <= 0 or speed_mps < 0:
        raise ValueError(
            "Density must be positive and speed cannot be negative."
        )

    return 0.5 * density_kg_m3 * speed_mps**2


def pressure_from_cp(
    cp: np.ndarray,
    ambient_pressure_pa: float,
    density_kg_m3: float,
    speed_mps: float,
) -> np.ndarray:
    """Convert surface pressure coefficients to absolute pressure."""

    coefficients = np.asarray(cp, dtype=float)

    if coefficients.ndim != 1 or not np.all(np.isfinite(coefficients)):
        raise ValueError("Cp must be a finite one-dimensional array.")

    if not np.isfinite(ambient_pressure_pa) or ambient_pressure_pa <= 0:
        raise ValueError("Ambient pressure must be finite and positive.")

    q = dynamic_pressure(density_kg_m3, speed_mps)

    pressure = ambient_pressure_pa + coefficients * q

    if not np.all(np.isfinite(pressure)) or np.any(pressure <= 0):
        raise ValueError("Calculated absolute pressure must be finite and positive.")

    return pressure

def cp_from_pressure(
    pressure_pa: np.ndarray,
    ambient_pressure_pa: float,
    density_kg_m3: float,
    speed_mps: float,
) -> np.ndarray:
    """Convert absolute surface pressure to pressure coefficients."""

    pressure = np.asarray(pressure_pa, dtype=float)

    if pressure.ndim != 1 or not np.all(np.isfinite(pressure)):
        raise ValueError("Pressure must be a finite one-dimensional array.")

    if not np.isfinite(ambient_pressure_pa) or ambient_pressure_pa <= 0:
        raise ValueError("Ambient pressure must be finite and positive.")

    if np.any(pressure <= 0):
        raise ValueError("Absolute pressure must be positive.")

    q = dynamic_pressure(density_kg_m3, speed_mps)

    if q == 0:
        raise ValueError("Cp is undefined when flow speed is zero.")

    return (pressure - ambient_pressure_pa) / q
