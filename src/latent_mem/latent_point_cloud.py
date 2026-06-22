"""
Core data structure for latent-mem.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class LatentPointCloud:
    """
    A latent point cloud where every point carries a C-dimensional VAE latent feature vector.

    Attributes:
        points_world: (N, 3) world-frame 3D coordinates.
        features: (N, C) latent features per point.
        valid_mask: (N,) boolean mask indicating valid points.
    """

    def __init__(
        self,
        points_world: torch.Tensor,
        features: torch.Tensor,
        valid_mask: torch.Tensor,
        intrinsics_latent: torch.Tensor,
        latent_hw: tuple[int, int],
    ) -> None:
        """
        Args:
            points_world: (N, 3) 3D point positions in world frame.
            features: (N, C) VAE latent feature per point.
            valid_mask: (N,) boolean, True for geometrically valid points.
            intrinsics_latent: (3, 3) camera intrinsics rescaled to latent res.
            latent_hw: (h, w) spatial resolution of the latent grid.
        """
        self.points_world = points_world
        self.features = features
        self.valid_mask = valid_mask
        self.intrinsics_latent = intrinsics_latent
        self.latent_hw = latent_hw

    @classmethod
    def from_image_path(
        cls,
        image_path: str,
        vae,
        intrinsics: Optional[np.ndarray | torch.Tensor] = None,
        cam2world: Optional[np.ndarray | torch.Tensor] = None,
        mask: Optional[np.ndarray | torch.Tensor] = None,
        device: str | torch.device = "cuda",
        mapanything_model=None,
    ) -> LatentPointCloud:
        """
        Build a latent point cloud from a single RGB image.

        Pipeline (mirrors the warp_latent notebook):
        1. Run MapAnything to obtain dense depth, intrinsics, and cam2world pose.
        2. Encode the image with the Wan VAE to get (C, 1, h, w) latents.
        3. Downsample the depth map to the latent spatial resolution (h, w).
        4. Rescale intrinsics accordingly.
        5. Back-project every latent pixel to 3D using depth + intrinsics + pose.

        Args:
            image_path: Path to the RGB image.
            vae: Pre-loaded WanVAEWrapper on *device*.
            intrinsics: Optional camera intrinsics for the input image.
            cam2world: Optional camera-to-world pose for the input image.
            mask: Optional binary mask at the original image resolution. Masked
                pixels are excluded from latent point cloud construction.
            device: Target torch device.
            mapanything_model: Optional pre-loaded MapAnything model to reuse.

        Returns:
            A new LatentPointCloud instance.
        """
        from mapanything.utils.image import load_images, preprocess_inputs

        if mapanything_model is None:
            from mapanything.models import MapAnything

            mapanything_model = MapAnything.from_pretrained("facebook/map-anything").to(
                device
            )

        # -- 1. MapAnything inference --
        if intrinsics is None and cam2world is None:
            # load_images accepts a file path and matches the original behavior.
            views = load_images([image_path])
        else:

            def _to_float32_numpy(
                array: np.ndarray | torch.Tensor,
            ) -> np.ndarray:
                if isinstance(array, torch.Tensor):
                    array = array.detach().cpu().numpy()
                return np.asarray(array, dtype=np.float32)

            image = np.asarray(Image.open(image_path).convert("RGB"))
            view = {"img": image}
            if intrinsics is not None:
                view["intrinsics"] = _to_float32_numpy(intrinsics)
            if cam2world is not None:
                view["camera_poses"] = _to_float32_numpy(cam2world)
            views = preprocess_inputs([view])

        predictions = mapanything_model.infer(
            views,
            memory_efficient_inference=True,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
        )

        pred = predictions[0]
        image = np.array(Image.open(image_path).convert("RGB"), copy=True)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().to(device)
        image_tensor = image_tensor / 255.0
        video_tensor = image_tensor.mul(2.0).sub(1.0).unsqueeze(1)

        with torch.no_grad():
            latent = vae.encode([video_tensor])[0].float().to(device)

        return cls.from_geometry(
            depth=pred["depth_z"][0, :, :, 0],
            intrinsics=pred["intrinsics"][0],
            cam2world=pred["camera_poses"][0],
            latent=latent,
            mask=mask,
            device=device,
        )

    @classmethod
    def from_geometry(
        cls,
        depth: np.ndarray | torch.Tensor,
        intrinsics: np.ndarray | torch.Tensor,
        cam2world: np.ndarray | torch.Tensor,
        latent: torch.Tensor,
        mask: Optional[np.ndarray | torch.Tensor] = None,
        device: str | torch.device = "cuda",
    ) -> LatentPointCloud:
        """
        Build a latent point cloud from geometry aligned to the same camera view.

        Args:
            depth: Depth map aligned with the camera intrinsics.
            intrinsics: Camera intrinsics for the depth map resolution.
            cam2world: Camera-to-world transform for the frame.
            latent: Pre-computed VAE latent with shape (C, 1, h, w).
            mask: Optional binary mask at the depth-map resolution to exclude.
            device: Target torch device.
        """
        assert latent.ndim == 4, (
            f"latent must have shape (C, 1, h, w), got {tuple(latent.shape)}"
        )

        _, _, h_lat, w_lat = latent.shape
        latent = latent.float().to(device)
        depth_tensor = torch.as_tensor(depth, dtype=torch.float32, device=device)
        intrinsics_tensor = torch.as_tensor(
            intrinsics,
            dtype=torch.float32,
            device=device,
        )
        cam2world_tensor = torch.as_tensor(
            cam2world,
            dtype=torch.float32,
            device=device,
        )

        assert depth_tensor.ndim == 2, (
            f"depth must have shape (H, W), got {tuple(depth_tensor.shape)}"
        )
        assert tuple(intrinsics_tensor.shape) == (3, 3), (
            f"intrinsics must have shape (3, 3), got {tuple(intrinsics_tensor.shape)}"
        )
        assert tuple(cam2world_tensor.shape) == (4, 4), (
            f"cam2world must have shape (4, 4), got {tuple(cam2world_tensor.shape)}"
        )

        depth_h, depth_w = depth_tensor.shape

        mask_latent = None
        if mask is not None:
            mask_tensor = torch.as_tensor(mask, dtype=torch.float32, device=device)
            assert tuple(mask_tensor.shape) == (depth_h, depth_w), (
                f"mask must have shape {(depth_h, depth_w)}, got {tuple(mask_tensor.shape)}"
            )
            mask_latent = (
                F.interpolate(
                    mask_tensor.unsqueeze(0).unsqueeze(0),
                    size=(h_lat, w_lat),
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
                > 0
            )

        depth_latent = (
            F.interpolate(
                depth_tensor.unsqueeze(0).unsqueeze(0),
                size=(h_lat, w_lat),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )

        scale_w = w_lat / depth_w
        scale_h = h_lat / depth_h
        intrinsics_latent = intrinsics_tensor.clone()
        intrinsics_latent[0, 0] *= scale_w
        intrinsics_latent[0, 2] *= scale_w
        intrinsics_latent[1, 1] *= scale_h
        intrinsics_latent[1, 2] *= scale_h

        v_coords, u_coords = torch.meshgrid(
            torch.arange(h_lat, device=device),
            torch.arange(w_lat, device=device),
            indexing="ij",
        )
        fx, fy = intrinsics_latent[0, 0], intrinsics_latent[1, 1]
        cx, cy = intrinsics_latent[0, 2], intrinsics_latent[1, 2]

        x_cam = (u_coords - cx) * depth_latent / fx
        y_cam = (v_coords - cy) * depth_latent / fy
        z_cam = depth_latent
        points_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)

        rotation = cam2world_tensor[:3, :3]
        translation = cam2world_tensor[:3, 3]
        points_world = (points_cam @ rotation.T) + translation

        point_valid_mask = depth_latent > 0
        if mask_latent is not None:
            point_valid_mask = point_valid_mask & (~mask_latent)

        features = latent[:, 0].permute(1, 2, 0).reshape(-1, latent.shape[0])
        return cls(
            points_world=points_world.reshape(-1, 3),
            features=features,
            valid_mask=point_valid_mask.reshape(-1),
            intrinsics_latent=intrinsics_latent,
            latent_hw=(h_lat, w_lat),
        )

    @classmethod
    def from_video_geometry(
        cls,
        geometry: object,
        frame_idx: int,
        latent: torch.Tensor,
        mask: Optional[np.ndarray | torch.Tensor] = None,
        device: str | torch.device = "cuda",
    ) -> LatentPointCloud:
        """Build a latent point cloud from one frame in a geometry container."""
        return cls.from_geometry(
            depth=geometry.depths[frame_idx],
            intrinsics=geometry.intrinsics[frame_idx],
            cam2world=geometry.poses_c2w[frame_idx],
            latent=latent,
            mask=mask,
            device=device,
        )

    def update(
        self,
        depths: np.ndarray | torch.Tensor,
        intrinsics: np.ndarray | torch.Tensor,
        cam2worlds: np.ndarray | torch.Tensor,
        latents: torch.Tensor,
        masks: Optional[np.ndarray | torch.Tensor] = None,
    ) -> None:
        """Append latent RGBD views to this point cloud."""
        if latents.ndim == 3:
            latents = latents.unsqueeze(0)
        assert latents.ndim == 4, (
            f"latents must have shape (C, h, w) or (T, C, h, w), got {tuple(latents.shape)}"
        )

        depths = torch.as_tensor(
            depths,
            dtype=torch.float32,
            device=self.points_world.device,
        )
        intrinsics = torch.as_tensor(
            intrinsics,
            dtype=torch.float32,
            device=self.points_world.device,
        )
        cam2worlds = torch.as_tensor(
            cam2worlds,
            dtype=torch.float32,
            device=self.points_world.device,
        )
        if depths.ndim == 2:
            depths = depths.unsqueeze(0)
        if intrinsics.ndim == 2:
            intrinsics = intrinsics.unsqueeze(0)
        if cam2worlds.ndim == 2:
            cam2worlds = cam2worlds.unsqueeze(0)
        if masks is not None:
            masks = torch.as_tensor(
                masks,
                dtype=torch.float32,
                device=self.points_world.device,
            )
            if masks.ndim == 2:
                masks = masks.unsqueeze(0)

        num_frames = latents.shape[0]
        assert depths.shape[0] == num_frames, (
            f"depths has {depths.shape[0]} frames, got {num_frames} latents"
        )
        assert intrinsics.shape[0] == num_frames, (
            f"intrinsics has {intrinsics.shape[0]} frames, got {num_frames} latents"
        )
        assert cam2worlds.shape[0] == num_frames, (
            f"cam2worlds has {cam2worlds.shape[0]} frames, got {num_frames} latents"
        )
        if masks is not None:
            assert masks.shape[0] == num_frames, (
                f"masks has {masks.shape[0]} frames, got {num_frames} latents"
            )

        points_to_add = []
        features_to_add = []
        for frame_idx in range(num_frames):
            frame_lpc = LatentPointCloud.from_geometry(
                depth=depths[frame_idx],
                intrinsics=intrinsics[frame_idx],
                cam2world=cam2worlds[frame_idx],
                latent=latents[frame_idx].unsqueeze(1),
                mask=None if masks is None else masks[frame_idx],
                device=self.points_world.device,
            )
            frame_valid = frame_lpc.valid_mask.bool()
            if frame_valid.any():
                points_to_add.append(frame_lpc.points_world[frame_valid])
                features_to_add.append(frame_lpc.features[frame_valid])

        if not points_to_add:
            return

        new_points = torch.cat(points_to_add, dim=0).to(
            device=self.points_world.device,
            dtype=self.points_world.dtype,
        )
        new_features = torch.cat(features_to_add, dim=0).to(
            device=self.features.device,
            dtype=self.features.dtype,
        )

        # Multi-view updates append unstructured points beyond the initial latent grid.
        self.points_world = torch.cat([self.points_world, new_points], dim=0)
        self.features = torch.cat([self.features, new_features], dim=0)
        self.valid_mask = torch.cat(
            [
                self.valid_mask.to(device=self.points_world.device, dtype=torch.bool),
                torch.ones(
                    new_points.shape[0],
                    device=self.points_world.device,
                    dtype=torch.bool,
                ),
            ],
            dim=0,
        )

    def save(self, path: str | Path) -> None:
        """
        Save the latent point cloud to local disk.
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "points_world": self.points_world,
                "features": self.features,
                "valid_mask": self.valid_mask,
                "intrinsics_latent": self.intrinsics_latent,
                "latent_hw": self.latent_hw,
            },
            save_path,
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        map_location: str | torch.device | None = None,
    ) -> LatentPointCloud:
        """
        Load a latent point cloud from local disk.

        Returns:
            A restored LatentPointCloud instance.
        """
        data = torch.load(path, map_location=map_location, weights_only=False)
        return cls(
            points_world=data["points_world"],
            features=data["features"],
            valid_mask=data["valid_mask"],
            intrinsics_latent=data["intrinsics_latent"],
            latent_hw=tuple(data["latent_hw"]),
        )

    def project(
        self,
        cam2world: torch.Tensor | np.ndarray,
        intrinsics: Optional[torch.Tensor | np.ndarray] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Project the latent point cloud into 2D latent space for a given camera.

        Uses z-buffer forward splatting (nearest-pixel, closest-depth wins),
        matching the warp_latent notebook implementation.

        Args:
            cam2world: (4, 4) camera-to-world pose of the target view.
            intrinsics: (3, 3) intrinsics for the target view.  If None, reuses
                the intrinsics stored at construction time.
            target_hw: (h, w) output spatial size. Defaults to the original
                latent resolution.

        Returns:
            warped_latent: (C, h, w) projected latent features.
            warped_mask: (h, w) boolean mask of pixels that received a splat.
        """
        device = self.points_world.device
        cam2world = torch.as_tensor(cam2world, dtype=torch.float32, device=device)
        K = torch.as_tensor(
            intrinsics if intrinsics is not None else self.intrinsics_latent,
            dtype=torch.float32,
            device=device,
        )
        assert tuple(cam2world.shape) == (4, 4), (
            f"cam2world must have shape (4, 4), got {tuple(cam2world.shape)}"
        )
        assert tuple(K.shape) == (3, 3), (
            f"intrinsics must have shape (3, 3), got {tuple(K.shape)}"
        )
        h, w = self.latent_hw

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # World -> new camera frame
        world2cam = torch.inverse(cam2world)
        pts_cam = (self.points_world @ world2cam[:3, :3].T) + world2cam[:3, 3]  # (N, 3)

        # Perspective projection
        z = pts_cam[:, 2]
        u = (pts_cam[:, 0] * fx / z) + cx
        v = (pts_cam[:, 1] * fy / z) + cy

        u_int = torch.round(u).long()
        v_int = torch.round(v).long()

        in_bounds = (u_int >= 0) & (u_int < w) & (v_int >= 0) & (v_int < h)
        proj_valid = self.valid_mask & in_bounds & (z > 0)

        C = self.features.shape[1]
        warped_latent = torch.zeros(C, h, w, device=device)
        zbuffer = torch.full((h, w), float("inf"), device=device)

        u_valid = u_int[proj_valid]
        v_valid = v_int[proj_valid]
        z_valid = z[proj_valid]
        feat_valid = self.features[proj_valid]  # (M, C)

        # Sort by depth descending so that closer points overwrite farther ones
        # when we scatter sequentially. This avoids the per-pixel loop.
        order = torch.argsort(z_valid, descending=True)
        u_valid = u_valid[order]
        v_valid = v_valid[order]
        z_valid = z_valid[order]
        feat_valid = feat_valid[order]

        # Flat index scatter (last write wins = closest after desc sort)
        flat_idx = v_valid * w + u_valid  # (M,)
        zbuffer_flat = zbuffer.view(-1)
        warped_flat = warped_latent.view(C, -1)  # (C, h*w)

        zbuffer_flat[flat_idx] = z_valid
        warped_flat[:, flat_idx] = feat_valid.T  # (C, M)

        warped_latent = warped_flat.view(C, h, w)
        zbuffer = zbuffer_flat.view(h, w)
        warped_mask = zbuffer < float("inf")

        return warped_latent, warped_mask
