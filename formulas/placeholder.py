"""Stand-in solver so the UI has something to show. The numbers mean nothing.
Delete it once your own Solver subclass is wired up in main.py."""

from math import exp, sqrt

import numpy as np

from .mesh import MeshModel
from .pipeline import Progress, Solver, TestResult, TestSettings

STEPS = 160


class PlaceholderSolver(Solver):
    name = "Placeholder"
    placeholder = True

    def run(self, model: MeshModel, s: TestSettings, progress: Progress) -> TestResult:
        progress(0.15, "Atmosphere")
        temp = 288.15 - 0.0065 * min(s.altitude, 11000.0)
        rho = 1.225 * exp(-s.altitude / 8500.0)
        pressure0 = rho * 287.05 * temp
        mach = s.speed / sqrt(1.4 * 287.05 * temp)
        q = 0.5 * rho * s.speed**2

        progress(0.35, "Surface pressure")
        wind = s.wind
        facing = np.clip(-model.face_normals @ wind, 0.0, None)
        cp = 2.0 * facing**2
        pressure = pressure0 + q * cp

        progress(0.55, "Surface temperature")
        t_total = temp * (1.0 + 0.2 * mach**2)
        wall = temp + 0.85 * s.thermal_weight * (t_total - temp) * (0.25 + 0.75 * facing)

        force = (-(q * cp * model.face_areas)[:, None] * model.face_normals).sum(axis=0)
        up = np.array([0.0, 0.0, 1.0])
        lift_dir = up - (up @ wind) * wind
        lift_dir /= np.linalg.norm(lift_dir)
        drag, lift = float(force @ wind), float(force @ lift_dir)
        ref = float((model.face_areas * facing).sum()) or 1.0

        progress(0.7, "Streamlines")
        lines = self._streamlines(model, s)

        summary = {
            "mach": mach,
            "drag_n": drag,
            "lift_n": lift,
            "cd": drag / (q * ref),
            "cl": lift / (q * ref),
            "max_pressure_pa": float(pressure.max()),
            "max_temp_k": float(wall.max()),
        }
        return TestResult(model, s, pressure, cp, wall, lines, summary)

    @staticmethod
    def _streamlines(model: MeshModel, s: TestSettings) -> np.ndarray:
        wind = s.wind
        radius = 0.5 * model.length * 0.9
        centre = model.bounds.mean(axis=0)

        helper = np.array([0.0, 0.0, 1.0]) if abs(wind[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(wind, helper)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(wind, e1)

        span = np.linspace(-1.6, 1.6, s.line_count) * radius
        grid = np.stack(np.meshgrid(span, span), axis=-1).reshape(-1, 2)
        x = centre + grid[:, :1] * e1 + grid[:, 1:] * e2 - 2.4 * radius * wind

        rng = np.random.default_rng(7)
        dt = 5.4 * radius / STEPS
        path = np.empty((len(x), STEPS + 1, 3))
        path[:, 0] = x
        for k in range(STEPS):
            x = x + dt * _sphere_velocity(x - centre, wind, radius)
            x = x + 0.012 * s.turbulence * radius * rng.standard_normal(x.shape)
            path[:, k + 1] = x
        return path


def _sphere_velocity(r: np.ndarray, wind: np.ndarray, radius: float) -> np.ndarray:
    dist = np.maximum(np.linalg.norm(r, axis=1), 1.02 * radius)[:, None]
    along = (r @ wind)[:, None]
    return wind * (1.0 + radius**3 / (2.0 * dist**3)) - 1.5 * radius**3 * along * r / dist**5
