from dataclasses import dataclass, field
from typing import List


@dataclass
class BackboneConfig:
    wan_model_path: str
    # Optional VACE override/resume checkpoint. Base VACE init still comes
    # from the matching Wan backbone blocks even when this is empty.
    vace_model_path: str

    # Wan Backbone architecture
    dim: int
    eps: float
    ffn_dim: int
    freq_dim: int
    in_dim: int
    model_type: str
    num_heads: int
    num_layers: int
    out_dim: int
    text_len: int

    # VACE specific
    vace_layers: List[int]
    vace_in_dim: int

    # VAE
    vae_checkpoint: str
    vae_stride: List[int]
    image_or_video_shape: List[int]

    patch_size: List[int] = field(default_factory=lambda: [1, 2, 2])
    window_size: List[int] = field(default_factory=lambda: [-1, -1])
    qk_norm: bool = True
    cross_attn_norm: bool = True

    # Shared defaults

    gradient_checkpointing: bool = (
        True  # 训练默认为 True，推理时可在 config 中覆盖为 False
    )

    timestep_shift: float = 5.0
    use_lora: bool = True
    load_ema: bool = True


@dataclass
class Wan2_1_14BConfig(BackboneConfig):
    wan_model_path: str = "hf_cache/models--Wan-AI--Wan2.1-I2V-14B-480P/snapshots/6b73f84e66371cdfe870c72acd6826e1d61cf279"
    vace_model_path: str = "hf_cache/models--Wan-AI--Wan2.1-VACE-14B/snapshots/539c162b1387eac9dc4c20bd3f74671309e76a4c"

    # Wan backbone architecture
    dim: int = 5120
    eps: float = 1e-06
    ffn_dim: int = 13824
    freq_dim: int = 256
    in_dim: int = 36
    model_type: str = "i2v"
    num_heads: int = 40
    num_layers: int = 40
    out_dim: int = 16
    text_len: int = 512

    # VAE
    vae_checkpoint: str = "Wan2.1_VAE.pth"
    vae_stride: List[int] = field(default_factory=lambda: [4, 8, 8])
    image_or_video_shape: List[int] = field(
        default_factory=lambda: [1, 9, 16, 60, 104]
    )  # [batch, frames, channels, height, width]

    # VACE specific
    vace_layers: List[int] = field(
        default_factory=lambda: [0, 5, 10, 15, 20, 25, 30, 35]
    )
    vace_in_dim: int = 16


@dataclass
class Wan2_2_5BConfig(BackboneConfig):
    wan_model_path: str = "data/Wan-AI/Wan2.2-TI2V-5B"
    vace_model_path: str = ""

    # Wan backbone architecture
    dim: int = 3072
    eps: float = 1e-06
    ffn_dim: int = 14336
    freq_dim: int = 256
    in_dim: int = 48
    model_type: str = "ti2v"
    num_heads: int = 24
    num_layers: int = 30
    out_dim: int = 48
    text_len: int = 512
    patch_size: List[int] = field(default_factory=lambda: [1, 2, 2])
    window_size: List[int] = field(default_factory=lambda: [-1, -1])
    qk_norm: bool = True
    cross_attn_norm: bool = True

    # VAE
    vae_checkpoint: str = "Wan2.2_VAE.pth"
    vae_stride: List[int] = field(default_factory=lambda: [4, 16, 16])
    image_or_video_shape: List[int] = field(default_factory=lambda: [1, 9, 48, 44, 80])

    # VACE specific
    vace_layers: List[int] = field(
        default_factory=lambda: [0, 4, 8, 12, 16, 20, 24, 28]
    )
    vace_in_dim: int = 49  # 48 or 49(inject hole mask)
