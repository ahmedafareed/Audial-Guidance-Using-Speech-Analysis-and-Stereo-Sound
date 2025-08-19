"""Coordinate conversion utilities."""

import numpy as np


def cartesian_to_spherical(x: float, y: float, z: float):
    """Convert Cartesian coordinates to spherical coordinates.

    Coordinate system: x is forward, y is right, z is up.
    Azimuth: angle from x-axis towards y-axis in x-y plane (-180° to +180°)
    Elevation: angle from x-y plane towards z-axis (-90° to +90°)

    The function warns and returns (None, elevation, distance) when the azimuth
    lies outside the front hemisphere (-90..+90) to match the notebook logic.
    """
    print(f"[Coordinate Conversion] Input Cartesian coordinates: x={x}, y={y}, z={z}")

    # Calculate distance
    distance = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Avoid division by zero
    if distance == 0:
        raise ValueError("Input coordinates are all zero; distance is zero.")

    # Azimuth calculation from x and y
    azimuth = np.degrees(np.arctan2(y, x))

    # Only include azimuths from -90° to +90° (front hemisphere)
    if not (-90 <= azimuth <= 90):
        print(f"[Coordinate Conversion] Warning: Azimuth {azimuth}° is outside front hemisphere (|azimuth| > 90°). Excluding object.")
        azimuth = None

    # Elevation calculation from z and distance
    elevation = np.degrees(np.arcsin(z / distance))

    print(f"[Coordinate Conversion] Computed spherical coordinates: azimuth={azimuth}, elevation={elevation}, distance={distance}")
    return azimuth, elevation, distance
