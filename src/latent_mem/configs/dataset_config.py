from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataConfig:
    data_path: Path = Path("data")
    fps: int = 16

    # Task specific
    drop_text_prompt: float = 0.2

    # Sampling strategy
    random_sample_ref: bool = False
    random_sample_preceding: bool = False
    max_reference_frames: int | None = None
    max_preceding_frames: int | None = None
