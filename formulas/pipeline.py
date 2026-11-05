from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from .atmosphere import AtmosphereState, calculate_atmosphere
from .thermal import calculate_thermal_references
from math import cos, radians, sin
from time import perf_counter
from typing import Callable

import numpy as np

from .mesh import MeshModel

Progress = Callable[[float, str], None]

@dataclass(frozen=True)
class TestSettings:
    speed: float = 120.0          # m/s
    altitude: float = 0.0         # m
    angle_of_attack: float = 0.0  # deg
    sideslip: float = 0.0         # deg
    turbulence: float = 1.0       # weight, 1 = nominal
    thermal_weight: float = 1.0   # weight, 1 = nominal
    model_length: float = 4.0     # m, longest side of the model
    mesh_detail: int = 1200       # face budget for the tested mesh
    line_count: int = 7           # streamline seeds per side, lines = n * n

    @property
    def wind(self) -> np.ndarray:
        a, b = radians(self.angle_of_attack), radians(self.sideslip)
        return np.array([cos(a) * cos(b), sin(b), sin(a) * cos(b)])

    def as_dict(self) -> dict:
        return asdict(self)

@dataclass
class TestResult:
    model: MeshModel
    settings: TestSettings
    pressure: np.ndarray           # per face, Pa
    cp: np.ndarray                 # per face
    wall_temperature: np.ndarray   # per face, K
    streamlines: np.ndarray        # (lines, steps, 3)
    summary: dict[str, float] = field(default_factory=dict)
    advice: list[str] = field(default_factory=list)
    solver: str = ""
    elapsed: float = 0.0

    atmosphere: AtmosphereState | None = None # atmospheric conditions used for this simulation

class Solver(ABC):
    """Subclass this in formulas/ and hand it to App in main.py."""

    name = "solver"
    placeholder = False

    @abstractmethod
    def run(self, model: MeshModel, settings: TestSettings, progress: Progress) -> TestResult:
        ...

def run_test(solver: Solver, model: MeshModel, settings: TestSettings, progress: Progress, advisor=None) -> TestResult:
    # we import it logally to avoid circular dependency
    from .readiness import validate_settings
    
    readiness = validate_settings(settings)

    if not readiness.can_run:
        raise ValueError("Invalid simulation configuration:\n"
                        + "\n".join(readiness.errors))

    if not isinstance(model, MeshModel):
        raise ValueError("A valid MeshModel must be loaded.")

    atmosphere = calculate_atmosphere(altitude_m=settings.altitude, speed_mps=settings.speed)
    
    progress(0.03, "Preparing mesh")
    prepared = model.scaled_to(settings.model_length).simplified(settings.mesh_detail)

    started = perf_counter()

    result = solver.run(prepared, settings, progress)
    result.atmosphere = atmosphere

    result.elapsed = perf_counter() - started
    result.solver = solver.name

    thermal_references = calculate_thermal_references(atmosphere)

    result.summary.update(thermal_references)
    result.summary["elapsed_s"] = result.elapsed
    result.summary["mach"] = atmosphere.mach
    result.summary["air_density_kg_m3"] = atmosphere.density_kg_m3
    result.summary["ambient_pressure_pa"] = atmosphere.pressure_pa
    result.summary["ambient_temperature_k"] = atmosphere.temperature_k
    result.summary["speed_of_sound_mps"] = atmosphere.speed_of_sound_mps

    if advisor is not None:
        progress(0.93, "Asking advisor")
        result.advice = list(advisor.advise(result))
    return result
