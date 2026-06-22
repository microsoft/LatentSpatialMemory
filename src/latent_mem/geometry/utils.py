"""
SCENE PROJECTION LOADING FUNCTIONS
"""

from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
import torch
from tqdm import tqdm

from latent_mem.data_process.point_cloud import build_scene_point_cloud
from latent_mem.data_process.projection import render_projection
from latent_mem.data_process.qwen3vl_prompts import Qwen3VLEntityExtractor
from latent_mem.data_process.sam3_segmenter import Sam3VideoSegmenter
from latent_mem.data_process.video_io import load_video_frames

from .scene_context import SceneProjectionData


def generate_new_scene(
    data_dir: str,
    target_frame_indices: list,
    output_size: tuple,
    vae,
    device,
    voxel_size: float,
    qwen_model_path: str,
    sam3_model_path: str,
    custom_poses_c2w: np.ndarray,
    custom_intrinsics: np.ndarray,
) -> SceneProjectionData:
    """
    Generate new scene projection with full pipeline (Qwen + SAM3).

    Returns:
        SceneProjectionData: Scene projection and geometry data
    """
    print("Generating scene projection with Qwen + SAM3 pipeline...")

    result = generate_scene_projection_from_geometry(
        data_dir=data_dir,
        target_frames=target_frame_indices,
        output_size=output_size,
        vae=vae,
        device=device,
        voxel_size=voxel_size,
        qwen_model_path=qwen_model_path,
        sam3_model_path=sam3_model_path,
        qwen_device="cuda",
        custom_poses_c2w=custom_poses_c2w,
        custom_intrinsics=custom_intrinsics,
    )

    (
        scene_proj,
        points_world,
        colors,
        dynamic_mask_frame0,
        poses_c2w,
        intrinsics,
        processed_size,
        anchor_depth_frame0,
        anchor_frame0,
    ) = result
    print(f"Generated scene projection shape: {scene_proj.shape}")

    # 保存初始点云副本，用于最后生成对比视频
    initial_points_world = points_world.copy()
    initial_colors = colors.copy()

    return SceneProjectionData(
        scene_proj=scene_proj,
        points_world=points_world,
        colors=colors,
        dynamic_mask_frame0=dynamic_mask_frame0,
        poses_c2w=poses_c2w,
        intrinsics=intrinsics,
        processed_size=processed_size,
        anchor_depth_frame0=anchor_depth_frame0,
        anchor_frame0=anchor_frame0,
        initial_points_world=initial_points_world,
        initial_colors=initial_colors,
    )


def load_scene_projection(
    path: str | Path,
    device,
    custom_poses_c2w: np.ndarray = None,
    custom_intrinsics: np.ndarray = None,
) -> SceneProjectionData:
    """
    Load scene projection from path, automatically detecting whether it's a pre-computed
    scene directory or a standalone latent file.

    Priority:
    1. If path is a directory with pre-computed scene files (geometry.npz, clip.mp4, etc.),
       load full scene data with geometry
    2. Otherwise, treat as a latent file and load only scene_proj

    Args:
        path: Path to either a data directory or a latent file
        device: Target device
        custom_poses_c2w: Optional custom camera poses (only used for pre-computed scenes)
        custom_intrinsics: Optional custom intrinsics (only used for pre-computed scenes)

    Returns:
        SceneProjectionData: Scene projection and optional geometry data
    """
    path = Path(path)

    # Check if this is a pre-computed scene directory
    if path.is_dir():
        scene_proj_path = path / "train_target_scene_proj_rgb.pt"
        geometry_path = path / "geometry.npz"
        clip_path = path / "clip.mp4"

        # If all required files exist, use pre-computed scene logic
        if scene_proj_path.exists() and geometry_path.exists() and clip_path.exists():
            print(f"Loading pre-computed scene from directory: {path}")
            scene_proj = load_scene_projection_latent(path, device, torch.bfloat16)

            # Load geometry for later iterations
            geometry = np.load(geometry_path, allow_pickle=True)
            poses_c2w = (
                custom_poses_c2w
                if custom_poses_c2w is not None
                else geometry["poses_c2w"]
            )
            intrinsics = (
                custom_intrinsics
                if custom_intrinsics is not None
                else geometry["intrinsics"]
            )
            processed_size = tuple(geometry["processed_size"].astype(int))
            anchor_depth_frame0 = geometry["depths"][0]  # 只取第0帧

            # Load first frame from video
            anchor_frame0 = load_video_frames(
                clip_path, target_size=(processed_size[1], processed_size[0])
            )[0]

            return SceneProjectionData(
                scene_proj=scene_proj,
                poses_c2w=poses_c2w,
                intrinsics=intrinsics,
                processed_size=processed_size,
                anchor_depth_frame0=anchor_depth_frame0,
                anchor_frame0=anchor_frame0,
            )
        else:
            raise FileNotFoundError(
                f"Directory {path} does not contain required pre-computed scene files "
                f"(train_target_scene_proj_rgb.pt, geometry.npz, clip.mp4)"
            )

    # Otherwise, treat as a latent file
    print(f"Loading scene projection from latent file: {path}")
    scene_proj_data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(scene_proj_data, dict):
        scene_proj = scene_proj_data.get("latent", scene_proj_data.get("scene_proj"))
    else:
        scene_proj = scene_proj_data
    scene_proj = scene_proj.to(device=device, dtype=torch.bfloat16)
    # Dataset latents are stored as [T, C, H, W] for current Wan2.1/Wan2.2 setups.
    if scene_proj.dim() == 4 and scene_proj.shape[1] in (16, 48):
        scene_proj = scene_proj.permute(1, 0, 2, 3)
    print(f"Loaded scene projection shape: {scene_proj.shape}")

    return SceneProjectionData(scene_proj=scene_proj)


def load_scene_projection_latent(data_dir: Path, device, dtype) -> torch.Tensor:
    """Load pre-computed scene projection latent from .pt file."""
    scene_proj_path = data_dir / "train_target_scene_proj_rgb.pt"
    assert scene_proj_path.exists()

    scene_proj_data = torch.load(
        scene_proj_path, map_location="cpu", weights_only=False
    )
    if isinstance(scene_proj_data, dict):
        scene_proj = scene_proj_data.get("latent", scene_proj_data.get("scene_proj"))
    else:
        scene_proj = scene_proj_data

    scene_proj = scene_proj.to(device=device, dtype=dtype)

    # scene_proj shape from dataset: [T, C, H, W]
    # VACE expects: [C, T, H, W]
    if scene_proj.dim() == 4 and scene_proj.shape[1] in (16, 48):
        scene_proj = scene_proj.permute(1, 0, 2, 3)  # -> [C, T, H, W]

    return scene_proj


def generate_scene_projection_from_geometry(
    data_dir: Path,
    target_frames: list,
    output_size: tuple,
    vae,
    device,
    voxel_size: float = 0.02,
    # ========== Qwen + SAM3 配置 ==========
    qwen_model_path: str = None,
    sam3_model_path: str = None,
    qwen_device: str = "cuda",
    # ========== 相机位姿接口 ==========
    custom_poses_c2w: np.ndarray = None,
    custom_intrinsics: np.ndarray = None,
) -> tuple:
    """
    Generate scene projection on-the-fly with full pipeline:
    1. Use Qwen to detect dynamic objects (people, cars, etc.)
    2. Use SAM3 to segment dynamic objects
    3. Build point cloud excluding dynamic objects
    4. Project to target camera poses

    Args:
        data_dir: Directory containing geometry.npz and clip.mp4
        target_frames: List of frame indices to generate projection for
        output_size: (H, W) output size for projection
        vae: VAE wrapper for encoding
        device: Target device
        dtype: Target dtype
        voxel_size: Point cloud voxel size
        qwen_model_path: Path to Qwen3-VL model
        sam3_model_path: Path to SAM3 model
        qwen_device: Device for Qwen model
        sam3_device: Device for SAM3 model
        custom_poses_c2w: Custom camera poses [T, 4, 4], overrides geometry.npz
        custom_intrinsics: Custom intrinsics [T, 3, 3] or [3, 3], overrides geometry.npz

    Returns:
        tuple: (scene_proj_latent, points_world, colors, dynamic_mask_frame0,
                poses_c2w, intrinsics, processed_size, depth_frame0, rgb_frame0)
               Note: Only frame 0 data is returned since point cloud is built from frame 0 only.
    """

    geometry_path = data_dir / "geometry.npz"
    clip_path = data_dir / "clip.mp4"

    assert geometry_path.exists()
    assert clip_path.exists()

    # ========== 加载几何信息 ==========
    geometry = np.load(geometry_path, allow_pickle=True)
    depths = geometry["depths"]
    processed_size = geometry["processed_size"]  # [H, W]
    proc_H, proc_W = int(processed_size[0]), int(processed_size[1])

    # ========== 相机位姿 ==========
    if custom_poses_c2w is not None:
        poses_c2w = custom_poses_c2w
        print(f"Using custom camera poses: {poses_c2w.shape}")
    else:
        poses_c2w = geometry["poses_c2w"]
        print(f"Using poses from geometry.npz: {poses_c2w.shape}")

    if custom_intrinsics is not None:
        if custom_intrinsics.ndim == 2:
            intrinsics = np.tile(custom_intrinsics[None], (len(poses_c2w), 1, 1))
        else:
            intrinsics = custom_intrinsics
        print(f"Using custom intrinsics: {intrinsics.shape}")
    else:
        intrinsics = geometry["intrinsics"]
        print(f"Using intrinsics from geometry.npz: {intrinsics.shape}")

    # ========== Step 1: Qwen 检测动态物体 ==========
    # 优先使用 input_cropped.png（更清晰），否则用 clip.mp4

    if qwen_model_path is None:
        qwen_model_path = "/data/models/Qwen2.5-VL-3B-Instruct"

    qwen_extractor = Qwen3VLEntityExtractor(
        model_path=qwen_model_path,
        device=qwen_device,
    )

    first_frame_path = data_dir / "input_cropped.png"
    qwen_input = first_frame_path if first_frame_path.exists() else clip_path
    dynamic_prompts, raw_output = qwen_extractor.extract(qwen_input)
    print(f"Detected dynamic objects: {dynamic_prompts}")

    del qwen_extractor

    # ========== Step 2: SAM3 分割动态物体和天空（只需要第0帧） ==========

    if sam3_model_path is None:
        sam3_model_path = "/data/models/sam2.1_hiera_large.pt"

    # 合并需要分割的 prompts：动态物体 + 天空（永远排除天空）
    all_prompts_init = list(dynamic_prompts) if dynamic_prompts else []
    all_prompts_init.append("sky")  # 永远排除天空
    print(f"    Prompts for initial frame: {all_prompts_init}")

    sam3_segmenter = Sam3VideoSegmenter(
        checkpoint_path=sam3_model_path,
        mask_dilate=0,
    )

    # 只分割第0帧（构建点云只需要第0帧的 mask）
    dynamic_mask_frame0 = sam3_segmenter.segment(
        video_path=clip_path,
        prompts=all_prompts_init,
        frame_index=0,
        expected_frames=1,
    )[0]  # [H, W]
    print(f"    Generated exclusion mask for frame 0: {dynamic_mask_frame0.shape}")
    print(
        f"    Excluded pixels: {dynamic_mask_frame0.sum()} ({dynamic_mask_frame0.sum() / dynamic_mask_frame0.size * 100:.1f}%)"
    )

    del sam3_segmenter
    torch.cuda.empty_cache()

    # ========== Step 3: 构建点云并投影 ==========

    # 只加载第0帧用于构建点云
    first_frame = load_video_frames(clip_path, target_size=(proc_W, proc_H))[
        0
    ]  # [H, W, 3]

    # Resize dynamic mask if needed
    if dynamic_mask_frame0.shape != (proc_H, proc_W):
        dynamic_mask_frame0 = cv2.resize(
            dynamic_mask_frame0.astype(np.uint8),
            (proc_W, proc_H),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    # 构建点云（只用第0帧）
    points_world, colors = build_scene_point_cloud(
        depth=depths[0],
        K=intrinsics[0],
        c2w=poses_c2w[0],
        rgb=first_frame,
        dynamic_mask=dynamic_mask_frame0,
        voxel_size=voxel_size,
    )
    print(f"    Built point cloud: {len(points_world)} points")

    # ========== Render projections ==========
    scene_proj_latent = generate_scene_projection_from_pointcloud(
        points_world=points_world,
        colors=colors,
        target_frames=target_frames,
        poses_c2w=poses_c2w,
        intrinsics=intrinsics,
        output_size=output_size,
        processed_size=(proc_H, proc_W),
        vae=vae,
        device=device,
    )

    print(f"    Scene projection latent shape: {scene_proj_latent.shape}")

    # 返回值：只返回第0帧的数据（后续 anchor 初始化只需要第0帧）
    return (
        scene_proj_latent,
        points_world,
        colors,
        dynamic_mask_frame0,  # 只返回第0帧的 mask
        poses_c2w,
        intrinsics,
        (proc_H, proc_W),
        depths[0],  # 只返回第0帧的深度
        first_frame,  # 只返回第0帧的 RGB
    )


def generate_scene_projection_from_pointcloud(
    points_world: np.ndarray,
    colors: np.ndarray,
    target_frames: list,
    poses_c2w: np.ndarray,
    intrinsics: np.ndarray,
    output_size: tuple,
    processed_size: tuple,
    vae,
    device,
) -> torch.Tensor:
    """
    Generate scene projection latent from existing point cloud.

    Args:
        points_world: World-space point cloud [N, 3]
        colors: Point colors [N, 3]
        target_frames: List of frame indices to render
        poses_c2w: Camera poses [total_frames, 4, 4]
        intrinsics: Camera intrinsics [total_frames, 3, 3]
        output_size: (H, W) output pixel size
        processed_size: (H, W) size that intrinsics correspond to
        vae: VAE wrapper for encoding
        device: Target device

    Returns:
        scene_proj: [C, T, h, w] latent tensor
    """

    H, W = output_size
    proc_H, proc_W = processed_size

    # ========== Render projections ==========
    projections = []
    for frame_idx in tqdm(target_frames, desc="Rendering projections"):
        # Scale intrinsics for output size
        K_scaled = scale_intrinsics(intrinsics[frame_idx], (proc_H, proc_W), (H, W))

        proj = render_projection(
            points_world=points_world,
            K=K_scaled,
            c2w=poses_c2w[frame_idx],
            image_size=(H, W),
            channels=["rgb"],
            colors=colors,
            fill_holes_kernel=0,  # 永远不填充
        )
        projections.append(proj)

    # ========== Stack and encode ==========
    projections = np.stack(projections, axis=0)  # [T, H, W, C]
    projections = projections.transpose(0, 3, 1, 2)  # [T, C, H, W]

    # Normalize to [-1, 1]
    proj_tensor = torch.from_numpy(projections).float() / 127.5 - 1.0
    proj_tensor = proj_tensor.to(device=device, dtype=torch.bfloat16)

    # Encode with VAE
    with torch.no_grad():
        proj_tensor = proj_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W]
        scene_proj_latent = vae.encode_to_latent(proj_tensor)  # [1, T, C_latent, h, w]
        scene_proj_latent = scene_proj_latent.squeeze(0).permute(
            1, 0, 2, 3
        )  # [C, T, h, w]

    return scene_proj_latent


def scale_intrinsics(
    K: np.ndarray, from_size: Tuple[int, int], to_size: Tuple[int, int]
) -> np.ndarray:
    """
    将内参从一个尺寸缩放到另一个尺寸

    Args:
        K: 内参矩阵 (3, 3)
        from_size: 原始尺寸 (H, W)
        to_size: 目标尺寸 (H, W)

    Returns:
        缩放后的内参矩阵 (3, 3)
    """
    from_h, from_w = from_size
    to_h, to_w = to_size

    scale_x = to_w / from_w
    scale_y = to_h / from_h

    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x  # fx
    K_scaled[1, 1] *= scale_y  # fy
    K_scaled[0, 2] *= scale_x  # cx
    K_scaled[1, 2] *= scale_y  # cy

    return K_scaled


def save_point_cloud_ply(
    save_path: Union[str, Path],
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
    """
    保存点云为 PLY 文件

    Args:
        save_path: 保存路径
        points: 点坐标 (N, 3)
        colors: 点颜色 (N, 3) uint8 RGB
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure colors are uint8
    if colors.dtype != np.uint8:
        colors = (colors * 255).clip(0, 255).astype(np.uint8)

    num_points = len(points)

    with open(save_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for i in range(num_points):
            f.write(
                f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
                f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n"
            )

    print(f"Saved point cloud: {save_path} ({num_points} points)")


def scale_intrinsics_batch(
    intrinsics: np.ndarray, from_size: Tuple[int, int], to_size: Tuple[int, int]
) -> np.ndarray:
    """
    批量缩放内参

    Args:
        intrinsics: 内参矩阵 (N, 3, 3) 或 (3, 3)
        from_size: 原始尺寸 (H, W)
        to_size: 目标尺寸 (H, W)

    Returns:
        缩放后的内参矩阵
    """
    if intrinsics.ndim == 2:
        return scale_intrinsics(intrinsics, from_size, to_size)

    return np.stack(
        [scale_intrinsics(K, from_size, to_size) for K in intrinsics], axis=0
    )
