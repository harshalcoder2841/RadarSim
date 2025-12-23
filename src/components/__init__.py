"""
Components Package

High-level radar component models (antenna, receiver, transmitter).
"""

from .antenna import AntennaParameters, PhasedArrayAntenna

__all__ = [
    "PhasedArrayAntenna",
    "AntennaParameters",
]
