from data_process._0_0_0_root_assign import (
    MapAnythingConfig,
    SampleConfig,
    SpatiaConfig,
)
from data_process.sample_builder import build_training_sample
from data_process.types import SampleIndices, VideoGeometry

# Backward compatibility aliases
EpisodeConfig = SampleConfig
EpisodeIndices = SampleIndices
build_spatia_episode = build_training_sample

__all__ = [
    # New names
    "SampleConfig",
    "MapAnythingConfig",
    "SpatiaConfig",
    "SampleIndices",
    "VideoGeometry",
    "build_training_sample",
    # Backward compatibility
    "EpisodeConfig",
    "EpisodeIndices",
    "build_spatia_episode",
]
