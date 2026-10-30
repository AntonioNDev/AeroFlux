from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
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


class Solver(ABC):
    """Subclass this in formulas/ and hand it to App in main.py."""

    name = "solver"
    placeholder = False

    @abstractmethod
    def run(self, model: MeshModel, settings: TestSettings, progress: Progress) -> TestResult:
        ...


def run_test(solver: Solver, model: MeshModel, settings: TestSettings,
             progress: Progress, advisor=None) -> TestResult:
    progress(0.03, "Preparing mesh")
    prepared = model.scaled_to(settings.model_length).simplified(settings.mesh_detail)

    started = perf_counter()
    result = solver.run(prepared, settings, progress)
    result.elapsed = perf_counter() - started
    result.solver = solver.name
    result.summary["elapsed_s"] = result.elapsed

    if advisor is not None:
        progress(0.93, "Asking advisor")
        result.advice = list(advisor.advise(result))
    return result
