import math

from dataclasses import dataclass, field
from numbers import Integral
from .pipeline import TestSettings
from .mesh import MeshModel
from .diagnostics import analyze_mesh, MeshDiagnostics


@dataclass(slots=True)
class ReadinessReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def can_run(self) -> bool:
        return not self.errors

def validate_number(value: float, name: str, minimum: float | None = None, maximum: float | None = None) -> list[str]:
    """
    Validate a numeric simulation parameter.
    """

    errors = []

    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return [f"{name} must be a number."]

    if not math.isfinite(value):
        return [f"{name} must be finite."]

    if minimum is not None and value < minimum:
        errors.append(
            f"{name} must be >= {minimum}."
        )

    if maximum is not None and value > maximum:
        errors.append(
            f"{name} must be <= {maximum}."
        )

    return errors

def validate_settings(settings: TestSettings) -> ReadinessReport:
    """
        Validate the numerical and structural configuration
        before AeroFlux starts a simulation.
    """

    report = ReadinessReport()

    numeric_rules = [
        ("speed", "Airspeed", 0.0),
        ("altitude", "Altitude", None),
        ("angle_of_attack", "Angle of attack", None),
        ("sideslip", "Sideslip", None),
        ("turbulence", "Turbulence", 0.0),
        ("thermal_weight", "Thermal weight", 0.0),
    ]

    for attribute, label, minimum in numeric_rules:
        value = getattr(settings, attribute)

        report.errors.extend(
            validate_number(value, label, minimum=minimum)
        )

    # Model length must be strictly positive.
    length_errors = validate_number(
        settings.model_length,
        "Model length"
    )

    report.errors.extend(length_errors)

    if not length_errors and float(settings.model_length) <= 0:
        report.errors.append(
            "Model length must be greater than zero."
        )

    # These parameters are counts, not continuous numbers.
    for attribute, label in [
        ("mesh_detail", "Mesh detail"),
        ("line_count", "Line count"),
    ]:
        value = getattr(settings, attribute)

        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or value <= 0
        ):
            report.errors.append(
                f"{label} must be a positive integer."
            )

    return report

def validate_geometry(model: MeshModel, diagnostics: MeshDiagnostics | None = None,) -> ReadinessReport:
    """
        Evaluate mesh readiness for simulation.
    """

    report = ReadinessReport()

    if not isinstance(model, MeshModel):
        report.errors.append("A valid MeshModel must be loaded.")
        return report

    if diagnostics is None:
        diagnostics = analyze_mesh(model)

    report.warnings.extend(diagnostics.warnings)

    if diagnostics.connected_components > 1:
        report.warnings.append(f"Mesh contains {diagnostics.connected_components} "
            "disconnected components.")

    return report

def check_readiness(model: MeshModel, settings: TestSettings, diagnostics: MeshDiagnostics | None = None) -> ReadinessReport:
    settings_report = validate_settings(settings)
    geometry_report = validate_geometry(model, diagnostics)

    return ReadinessReport(
        errors=settings_report.errors + geometry_report.errors,
        warnings=settings_report.warnings + geometry_report.warnings,
    )




