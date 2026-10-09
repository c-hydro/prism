"""In-memory rainfall interpolation with fixed or dynamic GRISO correlation."""
from .griso_interpolator import GrisoConfig, GrisoInterpolator

__all__ = ["GrisoConfig", "GrisoInterpolator"]
