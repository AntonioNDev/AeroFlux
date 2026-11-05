from __future__ import annotations
from math import isfinite

from .atmosphere import HEAT_CAPACITY_RATIO, AtmosphereState


def stagnation_temperature(ambient_temperature_k: float, mach: float, gamma: float = HEAT_CAPACITY_RATIO) -> float:
    """Calculates ideal-gas stagnation temp in kelvin."""
    if not isfinite(ambient_temperature_k) or ambient_temperature_k <= 0:
        raise ValueError("Ambient temperature must be finite and positive.")

    if not isfinite(mach) or mach < 0:
        raise ValueError("Mach number must be finite and non-negative")

    if not isfinite(gamma) or gamma <= 1:
        raise ValueError("Heat capacity ratio must be finite and greater than 1.")

    return ambient_temperature_k * (1.0 + 0.5 * (gamma - 1.0) * mach**2)

def recovery_factor(prandtl: float, boundary_layer: str = "turbulent") -> float:
    """Estimate the recovery factor for an idealized boundary layer."""
    if not isfinite(prandtl) or prandtl <= 0:
        raise ValueError("Prandtl number must be finite and positive.")
    if boundary_layer == "laminar":
        return prandtl**0.5
    if boundary_layer == "turbulent":
        return prandtl ** (1.0/3.0)

    raise ValueError("Boundary layer must be 'laminar' or 'turbulent'.")

def adiabatic_wall_temperature(ambient_temperature_k: float, mach: float, recovery: float, gamma: float = HEAT_CAPACITY_RATIO) -> float:
    """Calculate idealized adiabatic wall recovery temperature."""

    if not isfinite(ambient_temperature_k) or ambient_temperature_k <= 0:
        raise ValueError("Ambient temperature must be finite and positive.")

    if not isfinite(mach) or mach < 0:
        raise ValueError("Mach number must be finite and non-negative.")

    if not isfinite(recovery) or recovery < 0 or recovery > 1:
        raise ValueError("Recovery factor must be between 0 and 1.")

    if not isfinite(gamma) or gamma <= 1:
        raise ValueError("Heat capacity ratio must be greater than 1.")

    return ambient_temperature_k * (1.0 + recovery * 0.5 * (gamma - 1.0) * mach**2)

def convective_heat_flux(heat_transfer_coefficient: float, fluid_temperature_k: float, surface_temperature_k: float) -> float:
    """Calculate convective heat flux into a surface in W/m^2"""
    if (not isfinite(heat_transfer_coefficient) or heat_transfer_coefficient < 0):
        raise ValueError("Heat transfer coefficient must be finite and non-negative")
    if (not isfinite(fluid_temperature_k) or fluid_temperature_k <= 0):
        raise ValueError("Fluid temperature must be finite and positive.")
    if (not isfinite(surface_temperature_k) or surface_temperature_k <= 0):
        raise ValueError("Surface temperature must be finite and positive.")

    return heat_transfer_coefficient * (fluid_temperature_k - surface_temperature_k)

def convective_heat_transfer_rate(heat_flux_w_m2: float, surface_area_m2: float) -> float:
    """Calculate total convective heat transfer rate into a surface in W."""
    if not isfinite(heat_flux_w_m2):
        raise ValueError("Heat flux must be finite.")

    if not isfinite(surface_area_m2) or surface_area_m2 < 0:
        raise ValueError(
            "Surface area must be finite and non-negative."
        )

    return heat_flux_w_m2 * surface_area_m2

def calculate_thermal_references(atmosphare: AtmosphereState, prandtl: float = 0.71) -> dict[str, float]:
    """Calculate idealized free-stream thermal reference values."""
    turbulent_recovery = recovery_factor(prandtl, "turbulent")

    return {"stagnation_temperature_k": stagnation_temperature(atmosphare.temperature_k, atmosphare.mach),
            "adiabatic_wall_reference_k": adiabatic_wall_temperature(atmosphare.temperature_k, atmosphare.mach, turbulent_recovery),
            "recovery_factor": turbulent_recovery
            }

