"""The exact quantum-inspired arm: the Szegedy walk on the MuLP Google operator."""

from quantum.inspired.google_matrix import GoogleMatrixBuilder, ImplicitGoogleMatrix
from quantum.inspired.mulp import QuantumWalkRanker
from quantum.inspired.szegedy_walk import SzegedyWalk

__all__ = ["GoogleMatrixBuilder", "ImplicitGoogleMatrix", "QuantumWalkRanker", "SzegedyWalk"]
