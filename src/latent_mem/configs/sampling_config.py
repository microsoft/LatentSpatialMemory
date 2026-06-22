from dataclasses import dataclass


@dataclass
class SamplingConfig:
    seed: int = 42
    guidance_scale: float = 7.0
    timestep_shift: float = 5.0
    negative_prompt: str = (
        "过曝，静态，细节模糊不清，字幕，静止，最差质量，低质量，JPEG压缩残留，"
        "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
        "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，"
        "背景人很多，倒着走"
    )

    # Inference specifics
    vis_debug: bool = False
    infer_steps: int = 20
