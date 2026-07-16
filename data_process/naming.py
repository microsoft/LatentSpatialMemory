from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SampleNaming:
    """Naming scheme for training sample output files."""

    # RGB videos
    preceding_rgb: str
    target_rgb: str
    reference_rgb: str
    # Camera parameters (txt files)
    preceding_poses_c2w: str
    target_poses_c2w: str
    preceding_intrinsics: str
    target_intrinsics: str
    # Scene projection videos
    preceding_scene_proj: str
    target_scene_proj: str
    preceding_scene_proj_rgb: str
    target_scene_proj_rgb: str


# Backward compatibility alias
EpisodeNaming = SampleNaming


_NAMING_SCHEMES = {
    "legacy": SampleNaming(
        preceding_rgb="P_rgb",
        target_rgb="T_rgb",
        reference_rgb="R_rgb",
        preceding_poses_c2w="P_poses_c2w",
        target_poses_c2w="T_poses_c2w",
        preceding_intrinsics="P_intrinsics",
        target_intrinsics="T_intrinsics",
        preceding_scene_proj="proj_P",
        target_scene_proj="proj_T",
        preceding_scene_proj_rgb="proj_P_rgb",
        target_scene_proj_rgb="proj_T_rgb",
    ),
    "paper": SampleNaming(
        preceding_rgb="P_rgb",
        target_rgb="T_rgb",
        reference_rgb="R_rgb",
        preceding_poses_c2w="P_poses_c2w",
        target_poses_c2w="T_poses_c2w",
        preceding_intrinsics="P_intrinsics",
        target_intrinsics="T_intrinsics",
        preceding_scene_proj="SP",
        target_scene_proj="ST",
        preceding_scene_proj_rgb="SP_rgb",
        target_scene_proj_rgb="ST_rgb",
    ),
    "figure": SampleNaming(
        preceding_rgb="preceding_rgb",
        target_rgb="target_rgb",
        reference_rgb="reference_rgb",
        preceding_poses_c2w="preceding_poses_c2w",
        target_poses_c2w="target_poses_c2w",
        preceding_intrinsics="preceding_intrinsics",
        target_intrinsics="target_intrinsics",
        preceding_scene_proj="preceding_scene_proj",
        target_scene_proj="target_scene_proj",
        preceding_scene_proj_rgb="preceding_scene_proj_rgb",
        target_scene_proj_rgb="target_scene_proj_rgb",
    ),
}


def get_sample_naming(style: str) -> SampleNaming:
    """Get naming scheme for training sample output files."""
    if style not in _NAMING_SCHEMES:
        raise ValueError(f"Unknown naming style: {style}")
    return _NAMING_SCHEMES[style]


# Backward compatibility alias
get_episode_naming = get_sample_naming
