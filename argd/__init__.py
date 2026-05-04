# argd/__init__.py
from .core.builder import MVHSBuilder, TrainingHarness
from .phases.phase1_mvhs import ARGD_Core
from .core.adaptive_substrate import AdaptiveGraphSubstrate

__version__ = "0.1.0"
__all__ = ["MVHSBuilder", "TrainingHarness", "ARGD_Core", "AdaptiveGraphSubstrate"]
