# argd/data
try:
    from .data_generators import (
        RhythmicDataGenerator,
        GoldenRatioMusicGenerator,
        DataBatchLoader
    )
except ImportError:
    # scipy.signal incompatible with some Python versions
    RhythmicDataGenerator = None
    GoldenRatioMusicGenerator = None
    DataBatchLoader = None

__all__ = [
    'RhythmicDataGenerator',
    'GoldenRatioMusicGenerator',
    'DataBatchLoader'
]
