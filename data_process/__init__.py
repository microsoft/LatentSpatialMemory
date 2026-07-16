from data_process.data_config import (
    CONFIG,
    DataConfig,
    SampleConfig,
    get_episode_config,
    get_sample_config,
)
from data_process.sample_builder import build_training_sample
from data_process.types import SampleIndices, VideoGeometry

# Backward compatibility aliases
EpisodeConfig = SampleConfig
EpisodeIndices = SampleIndices

__all__ = [
    # New names
    "CONFIG",
    "DataConfig",
    "SampleConfig",
    "get_sample_config",
    "get_episode_config",
    "SampleIndices",
    "VideoGeometry",
    "build_training_sample",
    # Backward compatibility
    "EpisodeConfig",
    "EpisodeIndices",
]
