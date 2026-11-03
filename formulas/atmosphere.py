from dataclasses import dataclass
from math import exp, isfinite, sqrt


# Constants for the ISA (international-standard-atmoshpere).
SEA_LEVEL_TEMP_K = 288.15
SEA_LEVEL_PRESSURE_PA = 101325.0
GRAVITY_MPS2 = 9.80665
AIR_GAS_CONSTANT = 287.05287
HEAT_CAPACITY_RATIO = 1.4
LAPSE_RATE_K_PER_M = 0.0065

EARTH_RADIUS_M = 6_356_766.0
TROPOPAUSE_M = 11_000.0
MAX_GEOPOTENTIAL_ALTITUDE_M = 20_000.0


@dataclass(frozen=True, slots=True)
class AtmosphereState:
    altitude_m: float
    temperature_k: float
    pressure_pa: float
    density_kg_m3: float
    speed_of_sound_mps: float
    mach: float

def calculate_atmosphere(altitude_m: float, speed_mps: float) -> AtmosphereState:
    """Calculate  ISA properties from geometric altitude and relative airspeed"""
    
    if not isfinite(altitude_m) or not isfinite(speed_mps):
        raise ValueError("Altitude and speed must be finite.")

    if altitude_m < 0 or speed_mps < 0:
        raise ValueError("Altitude and speed cannot be negative.")

    # h - Geopotentional altitude
    h = EARTH_RADIUS_M * altitude_m / (EARTH_RADIUS_M + altitude_m)

    if h > MAX_GEOPOTENTIAL_ALTITUDE_M:
        raise ValueError(
            "Altitude exceeds the supported ISA range (0–20 km)."
        )
    
    if h <= TROPOPAUSE_M:
        # Troposphere
        temperature = (SEA_LEVEL_TEMP_K - LAPSE_RATE_K_PER_M * h)
        pressure = SEA_LEVEL_PRESSURE_PA * (temperature / SEA_LEVEL_TEMP_K) ** (GRAVITY_MPS2 / (AIR_GAS_CONSTANT * LAPSE_RATE_K_PER_M))

    else:
        # Lower stratosphere
        temperature = (SEA_LEVEL_TEMP_K - LAPSE_RATE_K_PER_M * TROPOPAUSE_M)
        pressure_at_tropopause = SEA_LEVEL_PRESSURE_PA * (temperature / SEA_LEVEL_TEMP_K) ** (GRAVITY_MPS2 / (AIR_GAS_CONSTANT * LAPSE_RATE_K_PER_M))
        pressure = pressure_at_tropopause * exp(-GRAVITY_MPS2 * (h - TROPOPAUSE_M) / (AIR_GAS_CONSTANT * temperature))

    density = pressure / (AIR_GAS_CONSTANT * temperature)
    speed_of_sound = sqrt(HEAT_CAPACITY_RATIO * AIR_GAS_CONSTANT * temperature)
    mach = speed_mps / speed_of_sound

    return AtmosphereState(
        altitude_m=altitude_m,
        temperature_k=temperature,
        pressure_pa=pressure,
        density_kg_m3=density,
        speed_of_sound_mps=speed_of_sound,
        mach=mach
    )