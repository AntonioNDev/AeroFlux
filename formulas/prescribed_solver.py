from __future__ import annotations

import numpy as np

from .atmosphere import calculate_atmosphere
from .forces import (integrate_pressure_forces, calculate_aerodynamic_loads)
from.mesh import MeshModel
from .pipeline import Progress, Solver, TestResult, TestSettings
from .pressure import pressure_from_cp


class PrescribedCpSolver(Solver):
    """
        Verification solver using supplied pressure coefficients.
    """
    name = "prescribed-cp-V1"
    placeholder = True

    def __init__(self, cp: np.ndarray, reference_area_m2: float):
        self.cp = np.asarray(cp, dtype=float).copy()
        self.reference_area_m2 = float(reference_area_m2)

    def run(self, model: MeshModel, settings: TestSettings, progress: Progress) -> TestResult:
        progress(0.10, "Validating pressure coefficients")

        face_count = len(model.faces)
        if self.cp.shape != (face_count,):
            raise ValueError("Cp array must contain one value per prepared mesh face.")

        if not np.all(np.isfinite(self.cp)):
            raise ValueError("Cp values must be finite.")

        if not np.isfinite(self.reference_area_m2):
            raise ValueError("Reference area must be finite.")

        if self.reference_area_m2 <= 0:
            raise ValueError("Reference area must be positive.")

        if settings.speed <= 0:
            raise ValueError("The Solver requires positive airflow speed.")

        progress(0.30, "Calculating atmospheric conditions")

        atmosphere = calculate_atmosphere(settings.altitude, settings.speed)
        pressure = pressure_from_cp(cp=self.cp, ambient_pressure_pa=atmosphere.pressure_pa, density_kg_m3=atmosphere.density_kg_m3, speed_mps=settings.speed)

        progress(0.55, "Integrating surface pressure forces")

        forces = integrate_pressure_forces(model=model, pressure_pa=pressure, ambient_pressure_pa=atmosphere.pressure_pa)

        progress(0.80, "Calculating aerodynamic loads")

        reference_up = np.array([0.0, 0.0, 1.0])

        if abs(np.dot(settings.wind, reference_up)) > 0.95:
            reference_up = np.array([0.0, 1.0, 0.0])

        loads = calculate_aerodynamic_loads(total_force_n=forces.total_force_n, flow_direction=settings.wind, density_kg_m3=atmosphere.density_kg_m3, speed_mps=settings.speed, reference_area_m2=self.reference_area_m2, reference_up=reference_up)

        summary = {
            "drag_n": loads.drag_n,
            "lift_n": loads.lift_n,
            "side_force_n": loads.side_force_n,
            "cd": loads.cd,
            "cl": loads.cl,
            "cy": loads.cy,
            "max_pressure_pa": float(np.max(pressure)),
            "reference_area_m2": self.reference_area_m2,
        }

        progress(1.0, "Aerodynamic verification complete")

        return TestResult(
            model=model,
            settings=settings,
            pressure=pressure,
            cp=self.cp.copy(),
            # No thermal model yet, test value
            wall_temperature=np.full(
                face_count,
                atmosphere.temperature_k,
            ),
            # No calculated flow field or streamlines yet.
            streamlines=np.empty((0, 0, 3), dtype=float),
            summary=summary,
            advice=[
                "VERIFICATION MODE: Cp values were supplied, not predicted.",
                "Pressure forces are calculated from the supplied Cp field.",
                "Viscous drag is not included.",
                "Wall temperature is not calculated.",
                "Airflow streamlines are not calculated.",
            ],
        )
