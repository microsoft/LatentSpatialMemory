"""
This script performs iterative multi-round inference for Spatia model:
- Supports generating videos longer than 33 frames through multiple iterations
- Each iteration generates up to 33 frames (or 4N+1 frames for the last iteration)
- After each iteration, updates the scene point cloud using MapAnything
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from einops import rearrange
from PIL import Image
from torchvision.io import write_video
from tqdm import tqdm

from latent_mem.configs.inference_config import SpatiaPipelineConfig
from latent_mem.geometry import SceneProjectionData
from latent_mem.geometry.utils import (
    generate_new_scene,
    generate_scene_projection_from_pointcloud,
    load_scene_projection,
    save_point_cloud_ply,
    scale_intrinsics_batch,
)
from latent_mem.inference.utils import (
    compute_iteration_plan,
    get_generator_model_type,
)
from latent_mem.wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from latent_mem.wrapper.wan.bidirectonal_vace import BidirectionalWanWrapperVACE


class SpatiaPipeline:
    def __init__(
        self,
        config: SpatiaPipelineConfig,
        output_frames: int,
        generator: BidirectionalWanWrapperVACE,
        vae,
        text_encoder,
        clip_encoder,
    ):
        """
        output_frames: Total frames in output.
        """
        self.config = config
        self.iteration_plan = compute_iteration_plan(output_frames)
        self.generator = generator
        self.vae = vae
        self.text_encoder = text_encoder
        self.clip_encoder = clip_encoder

    @torch.inference_mode()
    def generate(
        self,
        first_frame,
        text_prompt,
        h_pixel,
        w_pixel,
        data_dir=None,
        t0: int = 0,
        output_dir=None,
        video_dir=None,
        pointcloud_dir=None,
        device: str = "cuda",
        infer_steps: int = 50,
        no_cfg: bool = False,
        fps: int = 16,
        custom_poses_c2w=None,
        custom_intrinsics=None,
    ):
        scene_data = self._build_scene_projection(
            data_dir=Path(data_dir) if data_dir is not None else None,
            t0=t0,
            h_pixel=h_pixel,
            w_pixel=w_pixel,
            vae=self.vae,
            device=device,
            custom_poses_c2w=custom_poses_c2w,
            custom_intrinsics=custom_intrinsics,
        )
        state = self._init_iteration_state(scene_data, h_pixel, w_pixel, t0)
        runtime_args = self._make_runtime_args(infer_steps=infer_steps, no_cfg=no_cfg)
        runtime_config = self._make_runtime_config()

        current_first_frame = first_frame

        for iter_idx, (output_start, output_end, model_frames) in enumerate(
            self.iteration_plan
        ):
            target_frame_indices = self._build_target_frame_indices(
                iter_idx=iter_idx,
                output_start=output_start,
                model_frames=model_frames,
                t0=t0,
            )
            scene_proj = self._prepare_iteration_scene_projection(
                iter_idx=iter_idx,
                target_frame_indices=target_frame_indices,
                state=state,
                scene_data=scene_data,
                h_pixel=h_pixel,
                w_pixel=w_pixel,
                device=device,
            )
            preceding_frames, preceding_scene_proj = self._prepare_preceding_context(
                iter_idx=iter_idx,
                model_frames=model_frames,
                target_frame_indices=target_frame_indices,
                state=state,
                scene_data=scene_data,
                h_pixel=h_pixel,
                w_pixel=w_pixel,
                device=device,
            )
            reference_frames = self._prepare_reference_frames(
                iter_idx=iter_idx,
                target_frame_indices=target_frame_indices,
                state=state,
                h_pixel=h_pixel,
                w_pixel=w_pixel,
            )
            generated_video = run_generation_single_iter(
                generator=self.generator,
                vae=self.vae,
                text_encoder=self.text_encoder,
                clip_encoder=self.clip_encoder,
                first_frame=current_first_frame,
                scene_proj=scene_proj,
                prompt=text_prompt,
                num_frames=model_frames,
                config=runtime_config,
                args=runtime_args,
                preceding_frames=preceding_frames,
                preceding_scene_proj=preceding_scene_proj,
                reference_frames=reference_frames,
            )
            self._store_generated_frames(
                iter_idx=iter_idx, generated_video=generated_video, state=state
            )
            self._update_point_cloud_state(
                generated_video=generated_video,
                target_frame_indices=target_frame_indices,
                iter_idx=iter_idx,
                output_dir=output_dir,
                pointcloud_dir=pointcloud_dir,
                state=state,
                scene_data=scene_data,
                h_pixel=h_pixel,
                w_pixel=w_pixel,
            )
            self._update_frame_visibility(
                iter_idx=iter_idx,
                output_start=output_start,
                target_frame_indices=target_frame_indices,
                state=state,
                h_pixel=h_pixel,
                w_pixel=w_pixel,
            )
            self._save_iteration_video(
                iter_idx=iter_idx,
                output_start=output_start,
                output_end=output_end,
                generated_video=generated_video,
                video_dir=video_dir,
                fps=fps,
            )
            self._render_iteration_scene_projection(
                iter_idx=iter_idx,
                target_frame_indices=target_frame_indices,
                state=state,
                scene_data=scene_data,
                h_pixel=h_pixel,
                w_pixel=w_pixel,
                device=device,
            )
            if iter_idx < len(self.iteration_plan) - 1:
                current_first_frame = Image.fromarray(generated_video[-1].numpy())

        return self._finalize_outputs(
            state=state,
            scene_data=scene_data,
            t0=t0,
            h_pixel=h_pixel,
            w_pixel=w_pixel,
            video_dir=video_dir,
            pointcloud_dir=pointcloud_dir,
            output_dir=output_dir,
            fps=fps,
            device=device,
        )

    def _make_runtime_args(self, infer_steps: int, no_cfg: bool):
        return type(
            "RuntimeArgs",
            (),
            {
                "infer_steps": infer_steps,
                "no_cfg": no_cfg,
            },
        )()

    def _make_runtime_config(self):
        runtime_config = type("RuntimeConfig", (), {})()
        runtime_config.image_or_video_shape = self.config.image_or_video_shape
        runtime_config.vae_stride = self.config.vae_stride
        runtime_config.timestep_shift = self.config.timestep_shift
        runtime_config.guidance_scale = self.config.guidance_scale
        runtime_config.negative_prompt = self.config.negative_prompt
        runtime_config.num_train_timestep = self.config.num_train_timestep
        return runtime_config

    def _init_iteration_state(
        self,
        scene_data: SceneProjectionData,
        h_pixel: int,
        w_pixel: int,
        t0: int,
    ):
        init_frame = cv2.resize(
            scene_data.anchor_frame0, (w_pixel, h_pixel), interpolation=cv2.INTER_LINEAR
        )
        init_intrinsics = (
            scene_data.intrinsics[0:1].copy()
            if scene_data.intrinsics.ndim == 3
            else scene_data.intrinsics[None].copy()
        )
        init_depth = cv2.resize(
            scene_data.anchor_depth_frame0,
            (w_pixel, h_pixel),
            interpolation=cv2.INTER_NEAREST,
        )
        return {
            "scene_proj": scene_data.scene_proj,
            "points_world": scene_data.points_world,
            "colors": scene_data.colors,
            "poses_c2w": scene_data.poses_c2w,
            "intrinsics": scene_data.intrinsics,
            "all_generated_frames": [],
            "all_scene_proj_frames": [],
            "frame_visible_points": {},
            "accumulated_anchor_frames": init_frame[None],
            "accumulated_anchor_poses": scene_data.poses_c2w[0:1],
            "accumulated_anchor_intrinsics": scale_intrinsics_batch(
                init_intrinsics,
                (scene_data.processed_size[0], scene_data.processed_size[1]),
                (h_pixel, w_pixel),
            ),
            "accumulated_anchor_depths": init_depth[None],
            "accumulated_anchor_global_indices": [t0],
        }

    def _build_target_frame_indices(
        self, iter_idx: int, output_start: int, model_frames: int, t0: int
    ):
        if iter_idx == 0:
            return list(range(t0, t0 + model_frames))
        pose_start = t0 + output_start - 1
        return list(range(pose_start, pose_start + model_frames))

    def _prepare_iteration_scene_projection(
        self,
        iter_idx: int,
        target_frame_indices,
        state: dict,
        scene_data: SceneProjectionData,
        h_pixel: int,
        w_pixel: int,
        device: str,
    ):
        if iter_idx > 0 and state["points_world"] is not None:
            state["scene_proj"] = generate_scene_projection_from_pointcloud(
                points_world=state["points_world"],
                colors=state["colors"],
                target_frames=target_frame_indices,
                poses_c2w=state["poses_c2w"],
                intrinsics=state["intrinsics"],
                output_size=(h_pixel, w_pixel),
                processed_size=scene_data.processed_size,
                vae=self.vae,
                device=device,
            )
        return state["scene_proj"]

    def _prepare_preceding_context(
        self,
        iter_idx: int,
        model_frames: int,
        target_frame_indices,
        state: dict,
        scene_data: SceneProjectionData,
        h_pixel: int,
        w_pixel: int,
        device: str,
    ):
        preceding_frames = None
        preceding_scene_proj = None
        if iter_idx <= 0 or len(state["all_generated_frames"]) == 0:
            return preceding_frames, preceding_scene_proj

        prev_iter_start, prev_iter_end, _ = self.iteration_plan[iter_idx - 1]
        prev_iter_frame_count = prev_iter_end - prev_iter_start
        max_preceding = min(prev_iter_frame_count, model_frames - 1)
        preceding_start_idx = len(state["all_generated_frames"]) - max_preceding
        preceding_frames = np.stack(
            state["all_generated_frames"][preceding_start_idx:], axis=0
        )
        if state["points_world"] is None:
            return preceding_frames, preceding_scene_proj

        preceding_pose_start = target_frame_indices[0] - len(preceding_frames)
        if preceding_pose_start < 0:
            return preceding_frames, preceding_scene_proj
        preceding_pose_indices = list(
            range(preceding_pose_start, target_frame_indices[0])
        )
        preceding_scene_proj = generate_scene_projection_from_pointcloud(
            points_world=state["points_world"],
            colors=state["colors"],
            target_frames=preceding_pose_indices,
            poses_c2w=state["poses_c2w"],
            intrinsics=state["intrinsics"],
            output_size=(h_pixel, w_pixel),
            processed_size=scene_data.processed_size,
            vae=self.vae,
            device=device,
        )
        return preceding_frames, preceding_scene_proj

    def _prepare_reference_frames(
        self,
        iter_idx: int,
        target_frame_indices,
        state: dict,
        h_pixel: int,
        w_pixel: int,
    ):
        if (
            iter_idx <= 0
            or state["points_world"] is None
            or len(state["frame_visible_points"]) == 0
        ):
            return None
        reference_frames, _ = retrieve_reference_frames(
            target_frame_indices=target_frame_indices,
            all_generated_frames=state["all_generated_frames"],
            frame_visible_points=state["frame_visible_points"],
            points_world=state["points_world"],
            poses_c2w=state["poses_c2w"],
            intrinsics=state["intrinsics"],
            image_size=(h_pixel, w_pixel),
            max_reference_frames=4,
            iou_threshold=0.05,
            voxel_size_iou=self.config.voxel_size * 2,
        )
        return reference_frames

    def _store_generated_frames(
        self, iter_idx: int, generated_video: torch.Tensor, state: dict
    ):
        if iter_idx == 0:
            frames_to_store = [f.numpy() for f in generated_video]
        else:
            frames_to_store = [f.numpy() for f in generated_video[1:]]
        state["all_generated_frames"].extend(frames_to_store)

    def _update_point_cloud_state(
        self,
        generated_video: torch.Tensor,
        target_frame_indices,
        iter_idx: int,
        output_dir,
        pointcloud_dir,
        state: dict,
        scene_data: SceneProjectionData,
        h_pixel: int,
        w_pixel: int,
    ):
        if (
            state["accumulated_anchor_frames"] is None
            or state["accumulated_anchor_depths"] is None
        ):
            return
        this_iter_frames = np.array([f.numpy() for f in generated_video])
        new_frames_resized = np.array(
            [
                cv2.resize(
                    f,
                    (scene_data.processed_size[1], scene_data.processed_size[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                for f in this_iter_frames
            ]
        )
        current_anchor_frames = np.array(
            [
                cv2.resize(
                    f,
                    (scene_data.processed_size[1], scene_data.processed_size[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                for f in state["accumulated_anchor_frames"]
            ]
        )
        current_anchor_intrinsics = scale_intrinsics_batch(
            state["accumulated_anchor_intrinsics"],
            (h_pixel, w_pixel),
            (scene_data.processed_size[0], scene_data.processed_size[1]),
        )
        current_anchor_depths = np.array(
            [
                cv2.resize(
                    d,
                    (scene_data.processed_size[1], scene_data.processed_size[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                for d in state["accumulated_anchor_depths"]
            ]
        )
        try:
            (
                points_world,
                colors,
                _,
                all_poses,
                all_intrinsics,
                new_depths,
                new_poses,
                new_intrinsics,
            ) = update_point_cloud_with_new_frames(
                anchor_frames=current_anchor_frames,
                anchor_poses_c2w=state["accumulated_anchor_poses"],
                anchor_intrinsics=current_anchor_intrinsics,
                anchor_depths=current_anchor_depths,
                new_frames=new_frames_resized,
                voxel_size=self.config.voxel_size,
                device="cuda",
                qwen_model_path=self.config.qwen_model_path,
                sam3_model_path=self.config.sam3_model_path,
                output_dir=output_dir,
                iteration_idx=iter_idx + 1,
                exclude_sky=True,
            )
        except Exception as e:
            print(
                f"Warning: point cloud update failed at iteration {iter_idx + 1}: {e}"
            )
            return

        state["points_world"] = points_world
        state["colors"] = colors
        num_anchor_in_update = len(current_anchor_frames)
        num_new_in_update = len(new_frames_resized)
        # FIXME Closed-loop/custom camera trajectories are overwritten here by
        # MapAnything-refined poses. Add a pose-lock path before relying on
        # Spatia closed-loop behavior across multi-iteration inference.
        for local_idx, global_idx in enumerate(
            state["accumulated_anchor_global_indices"]
        ):
            if global_idx < len(state["poses_c2w"]):
                state["poses_c2w"][global_idx] = all_poses[local_idx]
                if state["intrinsics"].ndim == 3:
                    state["intrinsics"][global_idx] = all_intrinsics[local_idx]
        for local_idx in range(num_new_in_update):
            all_idx = num_anchor_in_update + local_idx
            global_idx = target_frame_indices[local_idx]
            if global_idx < len(state["poses_c2w"]):
                state["poses_c2w"][global_idx] = all_poses[all_idx]
                if state["intrinsics"].ndim == 3:
                    state["intrinsics"][global_idx] = all_intrinsics[all_idx]
        if (
            new_depths is not None
            and new_poses is not None
            and new_intrinsics is not None
        ):
            new_intrinsics_scaled = scale_intrinsics_batch(
                new_intrinsics,
                (scene_data.processed_size[0], scene_data.processed_size[1]),
                (h_pixel, w_pixel),
            )
            state["accumulated_anchor_frames"] = np.concatenate(
                [state["accumulated_anchor_frames"], this_iter_frames], axis=0
            )
            state["accumulated_anchor_poses"] = np.concatenate(
                [state["accumulated_anchor_poses"], new_poses], axis=0
            )
            state["accumulated_anchor_intrinsics"] = np.concatenate(
                [state["accumulated_anchor_intrinsics"], new_intrinsics_scaled], axis=0
            )
            new_depths_scaled = np.array(
                [
                    cv2.resize(d, (w_pixel, h_pixel), interpolation=cv2.INTER_NEAREST)
                    for d in new_depths
                ]
            )
            state["accumulated_anchor_depths"] = np.concatenate(
                [state["accumulated_anchor_depths"], new_depths_scaled], axis=0
            )
            for local_idx in range(num_new_in_update):
                state["accumulated_anchor_global_indices"].append(
                    target_frame_indices[local_idx]
                )
        if (
            pointcloud_dir is not None
            and state["points_world"] is not None
            and state["colors"] is not None
        ):
            save_point_cloud_ply(
                pointcloud_dir / f"iteration_{iter_idx + 1:02d}_pointcloud.ply",
                state["points_world"],
                state["colors"],
            )

    def _update_frame_visibility(
        self,
        iter_idx: int,
        output_start: int,
        target_frame_indices,
        state: dict,
        h_pixel: int,
        w_pixel: int,
    ):
        if state["points_world"] is None or state["poses_c2w"] is None:
            return
        intr = (
            state["intrinsics"]
            if state["intrinsics"].ndim == 3
            else np.tile(state["intrinsics"][None], (len(state["poses_c2w"]), 1, 1))
        )
        frames_to_process = []
        for local_idx, frame_idx in enumerate(target_frame_indices):
            if frame_idx >= len(state["poses_c2w"]):
                continue
            if iter_idx == 0:
                global_frame_idx = local_idx
            else:
                if local_idx == 0:
                    continue
                global_frame_idx = output_start + local_idx - 1
            frames_to_process.append((frame_idx, global_frame_idx))
        for frame_idx, global_frame_idx in tqdm(
            frames_to_process, desc="  Computing frame visibility", leave=False
        ):
            visible_pts, _ = get_visible_points_for_frame(
                state["points_world"],
                state["colors"],
                state["poses_c2w"][frame_idx],
                intr[frame_idx],
                (h_pixel, w_pixel),
            )
            if len(visible_pts) > 0:
                state["frame_visible_points"][global_frame_idx] = visible_pts

    def _save_iteration_video(
        self,
        iter_idx: int,
        output_start: int,
        output_end: int,
        generated_video: torch.Tensor,
        video_dir,
        fps: int,
    ):
        if video_dir is None:
            return
        video_frames_to_save = generated_video if iter_idx == 0 else generated_video[1:]
        write_video(
            str(
                video_dir
                / f"iteration_{iter_idx + 1:02d}_frames{output_start}-{output_end - 1}.mp4"
            ),
            video_frames_to_save,
            fps=fps,
        )

    def _render_iteration_scene_projection(
        self,
        iter_idx: int,
        target_frame_indices,
        state: dict,
        scene_data: SceneProjectionData,
        h_pixel: int,
        w_pixel: int,
        device: str,
    ):
        if state["points_world"] is None or state["colors"] is None:
            return
        proj_frame_indices = (
            target_frame_indices if iter_idx == 0 else target_frame_indices[1:]
        )
        if len(proj_frame_indices) == 0:
            return
        iter_scene_proj = generate_scene_projection_from_pointcloud(
            points_world=state["points_world"],
            colors=state["colors"],
            target_frames=proj_frame_indices,
            poses_c2w=state["poses_c2w"],
            intrinsics=state["intrinsics"],
            output_size=(h_pixel, w_pixel),
            processed_size=scene_data.processed_size,
            vae=self.vae,
            device=device,
        )
        iter_scene_proj_decode = iter_scene_proj.permute(1, 0, 2, 3).unsqueeze(0)
        iter_scene_proj_video = self.vae.decode_to_pixel(iter_scene_proj_decode)
        iter_scene_proj_video = (iter_scene_proj_video + 1.0) / 2.0
        iter_scene_proj_video = iter_scene_proj_video.clamp(0, 1)
        iter_scene_proj_np = (iter_scene_proj_video[0] * 255).byte().cpu()
        iter_scene_proj_np = rearrange(iter_scene_proj_np, "t c h w -> t h w c").numpy()
        state["all_scene_proj_frames"].extend([f for f in iter_scene_proj_np])

    def _build_scene_projection_video(
        self,
        state: dict,
        scene_data: SceneProjectionData,
        t0: int,
        num_frames: int,
        h_pixel: int,
        w_pixel: int,
        device: str,
    ):
        if len(state["all_scene_proj_frames"]) > 0:
            return torch.from_numpy(np.stack(state["all_scene_proj_frames"], axis=0))
        if scene_data.initial_points_world is None or state["poses_c2w"] is None:
            return None
        all_target_indices = list(range(t0, t0 + num_frames))
        full_scene_proj = generate_scene_projection_from_pointcloud(
            points_world=scene_data.initial_points_world,
            colors=scene_data.initial_colors,
            target_frames=all_target_indices,
            poses_c2w=state["poses_c2w"],
            intrinsics=state["intrinsics"],
            output_size=(h_pixel, w_pixel),
            processed_size=scene_data.processed_size,
            vae=self.vae,
            device=device,
        )
        scene_proj_for_decode = full_scene_proj.permute(1, 0, 2, 3).unsqueeze(0)
        scene_proj_video = self.vae.decode_to_pixel(scene_proj_for_decode)
        scene_proj_video = (scene_proj_video + 1.0) / 2.0
        scene_proj_video = scene_proj_video.clamp(0, 1)
        scene_proj_video_save = (scene_proj_video[0] * 255).byte().cpu()
        return rearrange(scene_proj_video_save, "t c h w -> t h w c")

    def _build_initial_projection_video(
        self,
        scene_data: SceneProjectionData,
        state: dict,
        t0: int,
        num_frames: int,
        h_pixel: int,
        w_pixel: int,
        device: str,
    ):
        if scene_data.initial_points_world is None or state["poses_c2w"] is None:
            return None
        all_target_indices = list(range(t0, t0 + num_frames))
        initial_scene_proj = generate_scene_projection_from_pointcloud(
            points_world=scene_data.initial_points_world,
            colors=scene_data.initial_colors,
            target_frames=all_target_indices,
            poses_c2w=state["poses_c2w"],
            intrinsics=state["intrinsics"],
            output_size=(h_pixel, w_pixel),
            processed_size=scene_data.processed_size,
            vae=self.vae,
            device=device,
        )
        initial_proj_for_decode = initial_scene_proj.permute(1, 0, 2, 3).unsqueeze(0)
        initial_proj_video = self.vae.decode_to_pixel(initial_proj_for_decode)
        initial_proj_video = (initial_proj_video + 1.0) / 2.0
        initial_proj_video = initial_proj_video.clamp(0, 1)
        initial_proj_video_save = (initial_proj_video[0] * 255).byte().cpu()
        return rearrange(initial_proj_video_save, "t c h w -> t h w c")

    def _match_num_frames(self, frames: torch.Tensor, target_count: int):
        if frames is None:
            return None
        if frames.shape[0] > target_count:
            return frames[:target_count]
        if frames.shape[0] < target_count:
            pad = target_count - frames.shape[0]
            return torch.cat([frames, frames[-1:].repeat(pad, 1, 1, 1)], dim=0)
        return frames

    def _finalize_outputs(
        self,
        state: dict,
        scene_data: SceneProjectionData,
        t0: int,
        h_pixel: int,
        w_pixel: int,
        video_dir,
        pointcloud_dir,
        output_dir,
        fps: int,
        device: str,
    ):
        if len(state["all_generated_frames"]) == 0:
            return {}
        generated_video_final = torch.from_numpy(
            np.stack(state["all_generated_frames"], axis=0)
        )
        num_gen_frames = generated_video_final.shape[0]
        scene_proj_video_save = self._build_scene_projection_video(
            state=state,
            scene_data=scene_data,
            t0=t0,
            num_frames=num_gen_frames,
            h_pixel=h_pixel,
            w_pixel=w_pixel,
            device=device,
        )
        scene_proj_video_save = self._match_num_frames(
            scene_proj_video_save, num_gen_frames
        )
        if scene_proj_video_save is None:
            scene_proj_video_save = torch.zeros_like(generated_video_final)

        initial_proj_video_save = self._build_initial_projection_video(
            scene_data=scene_data,
            state=state,
            t0=t0,
            num_frames=num_gen_frames,
            h_pixel=h_pixel,
            w_pixel=w_pixel,
            device=device,
        )
        initial_proj_video_save = self._match_num_frames(
            initial_proj_video_save, num_gen_frames
        )
        if initial_proj_video_save is None:
            initial_proj_video_save = scene_proj_video_save

        comparison_video = torch.cat(
            [generated_video_final, scene_proj_video_save], dim=2
        )
        if (
            pointcloud_dir is not None
            and state["points_world"] is not None
            and state["colors"] is not None
        ):
            save_point_cloud_ply(
                pointcloud_dir / "final_pointcloud.ply",
                state["points_world"],
                state["colors"],
            )
        if video_dir is not None:
            write_video(
                str(video_dir / "generated_full.mp4"), generated_video_final, fps=fps
            )
            write_video(
                str(video_dir / "comparison_full.mp4"), comparison_video, fps=fps
            )
            write_video(
                str(video_dir / "scene_projection_full.mp4"),
                scene_proj_video_save,
                fps=fps,
            )
            write_video(
                str(video_dir / "initial_projection_full.mp4"),
                initial_proj_video_save,
                fps=fps,
            )
        if output_dir is not None:
            first_frame_output = output_dir / "first_frame.png"
            Image.fromarray(generated_video_final[0].numpy()).save(first_frame_output)

        return {
            "generated_video": generated_video_final,
            "scene_projection_video": scene_proj_video_save,
            "initial_projection_video": initial_proj_video_save,
            "comparison_video": comparison_video,
            "points_world": state["points_world"],
            "colors": state["colors"],
            "poses_c2w": state["poses_c2w"],
            "intrinsics": state["intrinsics"],
        }

    def _build_scene_projection(
        self,
        data_dir: Path,
        t0: int,
        h_pixel: int,
        w_pixel: int,
        vae,
        device,
        custom_poses_c2w=None,
        custom_intrinsics=None,
    ) -> SceneProjectionData:
        """
        Load or generate scene projection based on available inputs.

        Args:
            args: Command line arguments containing scene projection settings
            data_dir: Directory containing input data
            t0: Starting frame index
            h_pixel: Output height in pixels
            w_pixel: Output width in pixels
            h: Latent space height
            w: Latent space width
            vae: VAE model for encoding
            device: Device for computation
            config: Model configuration
            custom_poses_c2w: Custom camera poses (optional)
            custom_intrinsics: Custom camera intrinsics (optional)

        Returns:
            scene_data: Dataclass containing scene projection and all related data
        """

        if self.config.scene_data_dir is None:
            if data_dir is None:
                raise ValueError(
                    "data_dir is required when config.scene_data_dir is None"
                )
            # Generate new scene with full pipeline
            first_iter_frames = self.iteration_plan[0][2]
            target_frame_indices = list(range(t0, t0 + first_iter_frames))
            scene_data = generate_new_scene(
                data_dir=data_dir,
                target_frame_indices=target_frame_indices,
                output_size=(h_pixel, w_pixel),
                vae=vae,
                device=device,
                voxel_size=self.config.voxel_size,
                qwen_model_path=self.config.qwen_model_path,
                sam3_model_path=self.config.sam3_model_path,
                custom_poses_c2w=custom_poses_c2w,
                custom_intrinsics=custom_intrinsics,
            )
        else:
            scene_data = load_scene_projection(
                path=self.config.scene_data_dir,
                device=device,
                custom_poses_c2w=custom_poses_c2w,
                custom_intrinsics=custom_intrinsics,
            )

        return scene_data


# ============================================================================
# MAPANYTHING INTEGRATION
# ============================================================================
def run_mapanything_reconstruction(
    frames: np.ndarray,
    poses_c2w: np.ndarray,
    intrinsics: np.ndarray,
    device: str = "cuda",
) -> dict:
    """
    Run MapAnything to reconstruct 3D point cloud from frames with known poses and intrinsics.

    This function uses MapAnything's Multi-Modal Inference mode, passing:
    - Images (RGB frames)
    - Camera intrinsics
    - Camera poses (cam2world, OpenCV convention)

    This ensures the reconstructed point cloud is in the same world coordinate system
    as the original geometry.

    Args:
        frames: RGB frames [T, H, W, 3] with values in [0, 255]
        poses_c2w: Camera-to-world poses [T, 4, 4] in OpenCV convention
        intrinsics: Camera intrinsics [T, 3, 3] or [3, 3]
        device: Device for MapAnything inference

    Returns:
        dict: {
            'pts3d': World-space 3D points [T, H, W, 3],
            'depths': Z-depth maps [T, H, W],
            'intrinsics': (Possibly refined) intrinsics [T, 3, 3],
            'poses_c2w': (Possibly refined) poses [T, 4, 4]
        }
    """
    # ========== Add MapAnything to path ==========
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mapanything_path = os.path.join(project_root, "misc", "map-anything")
    if mapanything_path not in sys.path:
        sys.path.insert(0, mapanything_path)

    from mapanything.models import MapAnything
    from mapanything.utils.image import preprocess_inputs

    print(f"  [MapAnything] Running reconstruction on {len(frames)} frames...")

    # ========== Initialize MapAnything model ==========
    model = MapAnything.from_pretrained("facebook/map-anything").to(device)
    model.eval()

    # ========== Prepare input views ==========
    # Expand intrinsics if single matrix provided
    if intrinsics.ndim == 2:
        intrinsics = np.tile(intrinsics[None], (len(frames), 1, 1))

    # Ensure float32 dtype for intrinsics and poses (MapAnything expects float32)
    intrinsics = intrinsics.astype(np.float32)
    poses_c2w = poses_c2w.astype(np.float32)

    views = []
    for i in range(len(frames)):
        view = {
            "img": frames[i],  # [H, W, 3] uint8
            "intrinsics": intrinsics[i],  # [3, 3] float32
            "camera_poses": poses_c2w[i],  # [4, 4] float32, cam2world OpenCV convention
            "is_metric_scale": torch.tensor([True]),
        }
        views.append(view)

    # ========== Preprocess inputs ==========
    processed_views = preprocess_inputs(views)

    # ========== Run inference ==========
    with torch.no_grad():
        predictions = model.infer(
            processed_views,
            memory_efficient_inference=True,  # Use memory efficient mode for longer videos
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=False,
        )

    # ========== Extract results ==========
    pts3d_list = []
    depths_list = []
    intrinsics_list = []
    poses_list = []

    for pred in predictions:
        pts3d_list.append(pred["pts3d"].cpu().numpy())  # [1, H, W, 3]
        depths_list.append(pred["depth_z"].cpu().numpy())  # [1, H, W, 1]
        intrinsics_list.append(pred["intrinsics"].cpu().numpy())  # [1, 3, 3]
        poses_list.append(pred["camera_poses"].cpu().numpy())  # [1, 4, 4]

    # ========== Stack results ==========
    pts3d = np.concatenate(pts3d_list, axis=0)  # [T, H, W, 3]
    depths = np.concatenate(depths_list, axis=0).squeeze(-1)  # [T, H, W]
    intrinsics_out = np.concatenate(intrinsics_list, axis=0)  # [T, 3, 3]
    poses_out = np.concatenate(poses_list, axis=0)  # [T, 4, 4]

    # ========== Clean up ==========
    del model
    torch.cuda.empty_cache()

    print(
        f"  [MapAnything] Reconstruction complete: {pts3d.shape[1]}x{pts3d.shape[2]} resolution"
    )

    return {
        "pts3d": pts3d,
        "depths": depths,
        "intrinsics": intrinsics_out,
        "poses_c2w": poses_out,
    }


def update_point_cloud_with_new_frames(
    anchor_frames: np.ndarray,
    anchor_poses_c2w: np.ndarray,
    anchor_intrinsics: np.ndarray,
    anchor_depths: np.ndarray,
    *,
    new_frames: np.ndarray,
    qwen_model_path: str,
    sam3_model_path: str,
    voxel_size: float = 0.02,
    device: str = "cuda",
    output_dir: str = None,
    iteration_idx: int = 0,
    exclude_sky: bool = True,
) -> tuple:
    """
    Update scene point cloud using MapAnything's multi-modal inference.

    This uses MapAnything's ability to process mixed inputs:
    - Anchor frames: Have known pose, intrinsics, and depth (from initial estimation or previous iterations)
    - New frames: Only have images, MapAnything will estimate their geometry

    MapAnything will align all frames to a consistent world coordinate system using
    the anchor frames as reference.

    **重要**:
    - Anchor frames 的点会直接使用（它们之前已经做过动态物体过滤）
    - New frames 会经过 Qwen + SAM3 检测和分割动态物体，只有背景点才会被加入点云

    Args:
        anchor_frames: RGB frames with known geometry [T_anchor, H, W, 3] uint8
        anchor_poses_c2w: Camera poses for anchor frames [T_anchor, 4, 4]
        anchor_intrinsics: Camera intrinsics for anchor frames [T_anchor, 3, 3] or [3, 3]
        anchor_depths: Z-depth maps for anchor frames [T_anchor, H, W]
        new_frames: New RGB frames (only images) [T_new, H, W, 3] uint8
        voxel_size: Voxel size for point cloud downsampling
        device: Device for MapAnything
        qwen_model_path: Path to Qwen model for dynamic object detection
        sam3_model_path: Path to SAM3 model for segmentation
        output_dir: Directory to save dynamic masks (optional)
        iteration_idx: Current iteration index for naming saved files
        exclude_sky: Whether to exclude sky regions from point cloud (default: True)

    Returns:
        tuple: (points_world, colors, all_depths, all_poses, all_intrinsics, new_depths, new_poses, new_intrinsics)
            - points_world: Updated point cloud [N, 3] (anchor points + new background points)
            - colors: Point colors [N, 3]
            - all_depths: Estimated depths for ALL frames [T_anchor + T_new, H, W] (in new coordinate system)
            - all_poses: Estimated poses for ALL frames [T_anchor + T_new, 4, 4] (in new coordinate system)
            - all_intrinsics: Estimated intrinsics for ALL frames [T_anchor + T_new, 3, 3]
            - new_depths: Estimated depths for new frames only [T_new, H, W]
            - new_poses: Estimated poses for new frames only [T_new, 4, 4]
            - new_intrinsics: Estimated intrinsics for new frames only [T_new, 3, 3]
    """
    import tempfile

    # ========== Add paths ==========
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mapanything_path = os.path.join(project_root, "misc", "map-anything")
    if mapanything_path not in sys.path:
        sys.path.insert(0, mapanything_path)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from mapanything.models import MapAnything
    from mapanything.utils.image import preprocess_inputs

    from latent_mem.data_process.qwen3vl_prompts import Qwen3VLEntityExtractor
    from latent_mem.data_process.sam3_segmenter import Sam3VideoSegmenter

    from .utils import save_video as save_video_util

    num_anchor = len(anchor_frames)
    num_new = len(new_frames)
    H, W = new_frames[0].shape[:2]
    print(
        f"  [PointCloud Update] Processing {num_anchor} anchor frames + {num_new} new frames..."
    )

    # ========== Step 1: Detect dynamic objects in new frames with Qwen ==========
    print("[PointCloud Update] Step 1/4: Detecting dynamic objects with Qwen...")

    # 保存新帧为临时视频供 Qwen 分析
    temp_video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    save_video_util(new_frames, temp_video_path, fps=16.0)

    try:
        qwen_extractor = Qwen3VLEntityExtractor(
            model_path=qwen_model_path,
            device=device,
        )
        dynamic_prompts, _ = qwen_extractor.extract(temp_video_path)
        print(f"    Detected dynamic objects: {dynamic_prompts}")
        del qwen_extractor
        torch.cuda.empty_cache()
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

    # ========== Step 2: Segment dynamic objects and sky with SAM3 ==========
    print(
        "[PointCloud Update] Step 2/4: Segmenting dynamic objects and sky with SAM3..."
    )

    # 合并需要分割的 prompts：动态物体 + 天空（如果需要排除）
    all_prompts = list(dynamic_prompts) if dynamic_prompts else []
    if exclude_sky:
        all_prompts.append("sky")
        print("Adding 'sky' to segmentation prompts for exclusion")

    if all_prompts:
        # 保存新帧为临时视频供 SAM3 分割
        temp_video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        save_video_util(new_frames, temp_video_path, fps=16.0)

        try:
            sam3_segmenter = Sam3VideoSegmenter(
                checkpoint_path=sam3_model_path,
                mask_dilate=5,  # 稍微膨胀 mask，避免边缘残留
            )
            new_frame_dynamic_masks = sam3_segmenter.segment(
                video_path=temp_video_path,
                prompts=all_prompts,  # 包含动态物体 + 天空
                frame_index=0,
                expected_frames=num_new,
            )  # [T_new, H, W]
            print(f"    Generated exclusion masks: {new_frame_dynamic_masks.shape}")
            print(f"    Prompts used: {all_prompts}")
            del sam3_segmenter
            torch.cuda.empty_cache()
        finally:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)

        # 保存 mask（如果指定了输出目录）
        if output_dir is not None:
            mask_dir = os.path.join(output_dir, "masks")
            os.makedirs(mask_dir, exist_ok=True)

            # 保存为 .npy 文件
            mask_path = os.path.join(
                mask_dir, f"iteration_{iteration_idx:02d}_exclusion_masks.npy"
            )
            np.save(mask_path, new_frame_dynamic_masks)
            print(f"    Saved exclusion masks to: {mask_path}")

            # 保存可视化视频
            mask_vis_frames = []
            for i in range(num_new):
                frame = new_frames[i].copy()
                mask = new_frame_dynamic_masks[i]
                # 用红色半透明覆盖被排除的区域
                overlay = frame.copy()
                overlay[mask] = [255, 0, 0]  # Red
                vis = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)
                mask_vis_frames.append(vis)
            mask_vis_frames = np.array(mask_vis_frames)
            mask_vis_path = os.path.join(
                mask_dir, f"iteration_{iteration_idx:02d}_exclusion_masks.mp4"
            )
            save_video_util(mask_vis_frames, mask_vis_path, fps=16.0)
            print(f"    Saved mask visualization to: {mask_vis_path}")

            # 保存 prompts 信息
            prompts_path = os.path.join(
                mask_dir, f"iteration_{iteration_idx:02d}_prompts.txt"
            )
            with open(prompts_path, "w") as f:
                f.write(f"Dynamic objects: {dynamic_prompts}\n")
                f.write(f"Exclude sky: {exclude_sky}\n")
                f.write(f"All prompts: {all_prompts}\n")
    else:
        print("    No objects to segment, using empty masks")
        new_frame_dynamic_masks = np.zeros((num_new, H, W), dtype=bool)

    # ========== Step 3: Run MapAnything to estimate geometry ==========
    print("[PointCloud Update] Step 3/4: Running MapAnything...")

    model = MapAnything.from_pretrained("facebook/map-anything").to(device)
    model.eval()

    # Prepare anchor intrinsics
    if anchor_intrinsics.ndim == 2:
        anchor_intrinsics = np.tile(anchor_intrinsics[None], (num_anchor, 1, 1))

    # Ensure float32 dtype
    anchor_intrinsics = anchor_intrinsics.astype(np.float32)
    anchor_poses_c2w = anchor_poses_c2w.astype(np.float32)
    anchor_depths = anchor_depths.astype(np.float32)

    # Build views list
    views = []

    # Anchor frames: provide full geometry (img + intrinsics + pose + depth)
    for i in range(num_anchor):
        view = {
            "img": anchor_frames[i],  # [H, W, 3] uint8
            "intrinsics": anchor_intrinsics[i],  # [3, 3] float32
            "camera_poses": anchor_poses_c2w[
                i
            ],  # [4, 4] float32, cam2world OpenCV convention
            "depth_z": anchor_depths[i],  # [H, W] float32
            "is_metric_scale": torch.tensor([True]),
        }
        views.append(view)

    # New frames: only provide images
    for i in range(num_new):
        view = {
            "img": new_frames[i],  # [H, W, 3] uint8
        }
        views.append(view)

    # Preprocess inputs
    processed_views = preprocess_inputs(views)

    # Run inference
    with torch.no_grad():
        predictions = model.infer(
            processed_views,
            memory_efficient_inference=True,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
        )

    del model
    torch.cuda.empty_cache()

    # ========== Step 4: Extract points from all frames ==========
    # - Anchor frames: 直接使用（它们之前已经做过动态物体过滤）
    # - New frames: 需要用 dynamic mask 过滤，只保留背景点
    #
    # 同时收集所有帧的 poses/intrinsics（MapAnything 重估后的新坐标系）
    print("[PointCloud Update] Step 4/4: Extracting points from all frames...")

    all_points_list = []
    all_colors_list = []

    # Store estimated geometry for ALL frames (in new coordinate system)
    all_depths_list = []
    all_poses_list = []
    all_intrinsics_list = []

    # Store estimated geometry for new frames only (for accumulating anchors)
    new_depths_list = []
    new_poses_list = []
    new_intrinsics_list = []

    for i, pred in enumerate(predictions):
        pts3d = pred["pts3d"].cpu().numpy().squeeze(0)  # [H, W, 3]
        depth_valid = np.isfinite(pts3d[..., 2]) & (pts3d[..., 2] > 0)

        # 收集所有帧的几何信息（MapAnything 重估后的新坐标系）
        all_depths_list.append(
            pred["depth_z"].cpu().numpy().squeeze(0).squeeze(-1)
        )  # [H, W]
        all_poses_list.append(pred["camera_poses"].cpu().numpy().squeeze(0))  # [4, 4]
        all_intrinsics_list.append(
            pred["intrinsics"].cpu().numpy().squeeze(0)
        )  # [3, 3]

        if i < num_anchor:
            # Anchor frames: 直接使用所有有效点（它们之前已经过滤过动态物体）
            frame_colors = anchor_frames[i]  # [H, W, 3]
            all_points_list.append(pts3d[depth_valid])
            all_colors_list.append(frame_colors[depth_valid])
            print(f"    Anchor frame {i}: {depth_valid.sum()} points added")
        else:
            # New frames: 需要用 dynamic mask 过滤
            new_idx = i - num_anchor
            frame_colors = new_frames[new_idx]  # [H, W, 3]
            dynamic_mask = new_frame_dynamic_masks[new_idx]  # [H, W]

            # Resize dynamic mask if needed
            if dynamic_mask.shape != (pts3d.shape[0], pts3d.shape[1]):
                dynamic_mask = cv2.resize(
                    dynamic_mask.astype(np.uint8),
                    (pts3d.shape[1], pts3d.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            # Create validity mask: valid depth AND NOT dynamic object (background only)
            background_mask = depth_valid & (~dynamic_mask)

            # Extract background points and colors
            all_points_list.append(pts3d[background_mask])
            all_colors_list.append(frame_colors[background_mask])

            # Store geometry for new frames only
            new_depths_list.append(
                pred["depth_z"].cpu().numpy().squeeze(0).squeeze(-1)
            )  # [H, W]
            new_poses_list.append(
                pred["camera_poses"].cpu().numpy().squeeze(0)
            )  # [4, 4]
            new_intrinsics_list.append(
                pred["intrinsics"].cpu().numpy().squeeze(0)
            )  # [3, 3]

            masked_pixels = dynamic_mask.sum()
            bg_pixels = background_mask.sum()
            print(
                f"    New frame {new_idx}: {masked_pixels} dynamic pixels masked, {bg_pixels} background points added"
            )

    # Concatenate all points
    if all_points_list:
        all_points = np.concatenate(all_points_list, axis=0)
        all_colors = np.concatenate(all_colors_list, axis=0)
    else:
        all_points = np.zeros((0, 3), dtype=np.float32)
        all_colors = np.zeros((0, 3), dtype=np.uint8)

    print(
        f"  [PointCloud Update] Total points before voxel downsample: {len(all_points)}"
    )

    # Voxel downsample to remove duplicates
    if voxel_size > 0 and len(all_points) > 0:
        vox = np.floor(all_points / voxel_size).astype(np.int32)
        _, unique_idx = np.unique(vox, axis=0, return_index=True)
        all_points = all_points[unique_idx].astype(np.float32)
        all_colors = all_colors[unique_idx]

    print(f"  [PointCloud Update] After voxel downsample: {len(all_points)} points")

    # Stack ALL frames geometry (in new coordinate system)
    all_depths = np.stack(all_depths_list, axis=0)  # [T_anchor + T_new, H, W]
    all_poses = np.stack(all_poses_list, axis=0)  # [T_anchor + T_new, 4, 4]
    all_intrinsics = np.stack(all_intrinsics_list, axis=0)  # [T_anchor + T_new, 3, 3]

    # Stack new frame geometry only
    new_depths = (
        np.stack(new_depths_list, axis=0) if new_depths_list else None
    )  # [T_new, H, W]
    new_poses = (
        np.stack(new_poses_list, axis=0) if new_poses_list else None
    )  # [T_new, 4, 4]
    new_intrinsics = (
        np.stack(new_intrinsics_list, axis=0) if new_intrinsics_list else None
    )  # [T_new, 3, 3]

    print(
        f"  [PointCloud Update] Returned {len(all_poses)} poses in new coordinate system"
    )

    return (
        all_points,
        all_colors,
        all_depths,
        all_poses,
        all_intrinsics,
        new_depths,
        new_poses,
        new_intrinsics,
    )


# ============================================================================
# REFERENCE FRAME RETRIEVAL (Algorithm 1 from Spatia paper)
# ============================================================================


def compute_3d_iou(
    points_a: np.ndarray,
    points_b: np.ndarray,
    voxel_size: float = 0.1,
) -> float:
    """
    Compute 3D IoU (Intersection over Union) between two point clouds.

    This implements the SPATIALOVERLAP function from Algorithm 1 in Spatia paper.
    The IoU is computed by voxelizing both point clouds and computing the
    intersection/union of occupied voxels.

    Args:
        points_a: First point cloud [N, 3]
        points_b: Second point cloud [M, 3]
        voxel_size: Voxel size for discretization

    Returns:
        iou: 3D IoU value between 0 and 1
    """
    if len(points_a) == 0 or len(points_b) == 0:
        return 0.0

    # Voxelize both point clouds
    vox_a = np.floor(points_a / voxel_size).astype(np.int32)
    vox_b = np.floor(points_b / voxel_size).astype(np.int32)

    # Convert to set of tuples for fast intersection/union
    set_a = set(map(tuple, vox_a))
    set_b = set(map(tuple, vox_b))

    # Compute IoU
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    if union == 0:
        return 0.0

    return intersection / union


def get_visible_points_for_frame(
    points_world: np.ndarray,
    colors: np.ndarray,
    pose_c2w: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple,
    depth_threshold: float = 100.0,
) -> tuple:
    """
    Get the subset of points visible from a specific camera viewpoint.

    Args:
        points_world: World-space point cloud [N, 3]
        colors: Point colors [N, 3]
        pose_c2w: Camera-to-world pose [4, 4]
        intrinsics: Camera intrinsics [3, 3]
        image_size: (H, W) image dimensions
        depth_threshold: Maximum depth to consider

    Returns:
        visible_points: Points visible from this viewpoint [M, 3]
        visible_colors: Colors of visible points [M, 3]
    """
    H, W = image_size

    # Transform to camera coordinates
    pose_w2c = np.linalg.inv(pose_c2w)
    R = pose_w2c[:3, :3]
    t = pose_w2c[:3, 3]

    points_cam = (R @ points_world.T).T + t  # [N, 3]

    # Filter points behind camera
    valid_depth = points_cam[:, 2] > 0.01
    valid_depth &= points_cam[:, 2] < depth_threshold

    # Project to image plane
    K = intrinsics
    points_proj = (K @ points_cam.T).T  # [N, 3]
    points_proj = points_proj[:, :2] / (points_proj[:, 2:3] + 1e-8)

    # Check if within image bounds
    valid_x = (points_proj[:, 0] >= 0) & (points_proj[:, 0] < W)
    valid_y = (points_proj[:, 1] >= 0) & (points_proj[:, 1] < H)

    valid = valid_depth & valid_x & valid_y

    return points_world[valid], colors[valid] if colors is not None else None


def retrieve_reference_frames(
    target_frame_indices: list,
    all_generated_frames: list,
    frame_visible_points: dict,
    points_world: np.ndarray,
    poses_c2w: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple,
    max_reference_frames: int = 4,
    iou_threshold: float = 0.1,
    voxel_size_iou: float = 0.1,
) -> tuple:
    """
    Retrieve reference frames based on 3D IoU spatial overlap.

    Implements Algorithm 1 from Spatia paper:
    For each target frame, find historically generated frames that have
    high spatial overlap (3D IoU) with the target viewpoint.

    Args:
        target_frame_indices: Frame indices for target generation
        all_generated_frames: List of all previously generated frames [N, H, W, 3] uint8
        frame_visible_points: Dict mapping frame_idx -> visible points for that frame
        points_world: Current scene point cloud [N, 3]
        poses_c2w: Camera poses for all frames [T, 4, 4]
        intrinsics: Camera intrinsics [T, 3, 3] or [3, 3]
        image_size: (H, W) image dimensions
        max_reference_frames: Maximum number of reference frames to retrieve
        iou_threshold: Minimum 3D IoU to consider a frame as reference
        voxel_size_iou: Voxel size for IoU computation

    Returns:
        reference_frames: Selected reference frames [R, H, W, 3] uint8
        reference_indices: Indices of selected reference frames
    """
    if len(all_generated_frames) == 0 or points_world is None:
        return None, []

    # Expand intrinsics if single matrix
    if intrinsics.ndim == 2:
        intrinsics = np.tile(intrinsics[None], (len(poses_c2w), 1, 1))

    # Uniformly sample historical frames if too many (limit to 33 for efficiency)
    max_hist_frames_for_iou = 33
    hist_items = list(frame_visible_points.items())
    if len(hist_items) > max_hist_frames_for_iou:
        # Uniformly sample indices
        step = len(hist_items) / max_hist_frames_for_iou
        sampled_indices = [int(i * step) for i in range(max_hist_frames_for_iou)]
        hist_items = [hist_items[i] for i in sampled_indices]
        print(
            f"    Uniformly sampled {len(hist_items)} historical frames for IoU computation"
        )

    # Compute visible points for target frames
    target_visible_points = []
    for frame_idx in tqdm(
        target_frame_indices, desc="    Computing target visibility", leave=False
    ):
        if frame_idx < len(poses_c2w):
            visible_pts, _ = get_visible_points_for_frame(
                points_world,
                None,
                poses_c2w[frame_idx],
                intrinsics[frame_idx],
                image_size,
            )
            target_visible_points.append(visible_pts)
        else:
            target_visible_points.append(np.zeros((0, 3)))

    # Combine all target visible points
    if len(target_visible_points) > 0:
        target_points_combined = np.concatenate(
            [p for p in target_visible_points if len(p) > 0], axis=0
        )
    else:
        return None, []

    if len(target_points_combined) == 0:
        return None, []

    # Compute IoU with each historical frame (sampled)
    iou_scores = []
    for hist_idx, hist_points in tqdm(
        hist_items, desc="    Computing 3D IoU", leave=False
    ):
        if hist_points is not None and len(hist_points) > 0:
            iou = compute_3d_iou(target_points_combined, hist_points, voxel_size_iou)
            iou_scores.append((hist_idx, iou))

    # Sort by IoU and select top frames
    iou_scores.sort(key=lambda x: x[1], reverse=True)

    # Filter by threshold and select top-k
    selected_indices = []
    for hist_idx, iou in iou_scores:
        if iou >= iou_threshold and len(selected_indices) < max_reference_frames:
            selected_indices.append(hist_idx)

    if len(selected_indices) == 0:
        return None, []

    # Gather reference frames
    reference_frames = []
    for idx in selected_indices:
        if idx < len(all_generated_frames):
            reference_frames.append(all_generated_frames[idx])

    if len(reference_frames) == 0:
        return None, []

    reference_frames = np.stack(reference_frames, axis=0)
    print(
        f"    Retrieved {len(reference_frames)} reference frames with IoU >= {iou_threshold}"
    )

    return reference_frames, selected_indices


# ============================================================================
# SINGLE ITERATION INFERENCE
# ============================================================================


def run_generation_single_iter(
    generator,
    vae,
    text_encoder,
    clip_encoder,
    first_frame: Image.Image,
    scene_proj: torch.Tensor,
    prompt: str,
    num_frames: int,
    config,
    args,
    preceding_frames: torch.Tensor = None,
    preceding_scene_proj: torch.Tensor = None,
    reference_frames: torch.Tensor = None,
) -> torch.Tensor:
    """
    Run a single iteration of video generation with Spatia conditioning.

    According to Spatia paper (Figure 3), the model uses:
    - Frame order: [T, P, R] where T=target(noisy), P=preceding(clean), R=reference(clean)
    - VACE ControlNet: processes [T_scene, P_scene] scene projections
    - Main model: processes [T_latent, P_latent, R_latent] concatenated latents

    Args:
        generator: The video generator model
        vae: VAE wrapper
        text_encoder: Text encoder
        clip_encoder: Optional CLIP encoder for I2V backbones
        first_frame: First frame PIL image (condition for I2V)
        scene_proj: Target scene projection latent [C, T, h, w]
        prompt: Text prompt
        num_frames: Number of frames to generate this iteration
        config: Model config
        device: Device
        args: Command line arguments
        preceding_frames: Preceding frames tensor [P, H, W, 3] uint8 (from previous iteration)
        preceding_scene_proj: Preceding scene projection latent [C, P, h, w]
        reference_frames: Reference frames tensor [R, H, W, 3] uint8 (retrieved by 3D IoU)

    Returns:
        generated_video: Generated video tensor [T, H, W, 3] uint8
    """
    with torch.inference_mode():
        h, w = config.image_or_video_shape[-2:]
        generator_param = next(generator.parameters())
        runtime_device = generator_param.device
        runtime_dtype = generator_param.dtype
        generator_model_type = get_generator_model_type(generator)
        use_image_conditioning = generator_model_type == "i2v"
        use_ti2v_first_frame_conditioning = generator_model_type == "ti2v"

        # ========== Calculate latent frames ==========
        num_latent_frames = (num_frames - 1) // config.vae_stride[0] + 1
        num_t = num_latent_frames  # Target frames (noisy)

        # ========== Prepare preceding latents (P) ==========
        num_p = 0
        preceding_latent = None
        if preceding_frames is not None and len(preceding_frames) > 0:
            # Convert preceding frames to latent
            # preceding_frames: [P_pixel, H, W, 3] uint8 -> [P_pixel, 3, H, W] float [-1, 1]
            p_tensor = (
                torch.from_numpy(preceding_frames).float().permute(0, 3, 1, 2) / 127.5
                - 1.0
            )
            p_tensor = p_tensor.to(device=runtime_device, dtype=runtime_dtype)
            # Encode to latent: [1, 3, P_pixel, H, W] -> [1, P_latent, C, h, w]
            p_tensor = p_tensor.permute(1, 0, 2, 3).unsqueeze(0)
            preceding_latent = vae.encode_to_latent(p_tensor).to(
                device=runtime_device, dtype=runtime_dtype
            )
            num_p = preceding_latent.shape[1]
            print(
                f"Preceding frames: {len(preceding_frames)} pixel -> {num_p} latent frames"
            )

        # ========== Prepare reference latents (R) ==========
        num_r = 0
        reference_latent = None
        if reference_frames is not None and len(reference_frames) > 0:
            # Convert reference frames to latent
            # reference_frames: [R_pixel, H, W, 3] uint8 -> [R_pixel, 3, H, W] float [-1, 1]
            r_tensor = (
                torch.from_numpy(reference_frames).float().permute(0, 3, 1, 2) / 127.5
                - 1.0
            )
            r_tensor = r_tensor.to(device=runtime_device, dtype=runtime_dtype)
            # Encode to latent: [1, 3, R_pixel, H, W] -> [1, R_latent, C, h, w]
            r_tensor = r_tensor.permute(1, 0, 2, 3).unsqueeze(0)
            reference_latent = vae.encode_to_latent(r_tensor).to(
                device=runtime_device, dtype=runtime_dtype
            )
            num_r = reference_latent.shape[1]
            print(
                f"Reference frames: {len(reference_frames)} pixel -> {num_r} latent frames"
            )

        # ========== Prepare text conditioning ==========
        text_conditional_dict = text_encoder(text_prompts=[prompt])
        context = text_conditional_dict["prompt_embeds"].to(
            device=runtime_device, dtype=runtime_dtype
        )

        if not args.no_cfg:
            negative_prompt = getattr(config, "negative_prompt", "")
            unconditional_dict = text_encoder(text_prompts=[negative_prompt])
            uncon_context = unconditional_dict["prompt_embeds"].to(
                device=runtime_device, dtype=runtime_dtype
            )
        else:
            uncon_context = None

        clip_fea = None
        y = None
        if use_image_conditioning:
            assert clip_encoder is not None
            img_tensor = (
                TF.to_tensor(first_frame)
                .sub_(0.5)
                .div_(0.5)
                .to(device=runtime_device, dtype=runtime_dtype)
            )
            clip_fea = clip_encoder(img_tensor).to(
                device=runtime_device, dtype=runtime_dtype
            )

            # Encode first frame with VAE (for I2V conditioning)
            # IMPORTANT: y must have total frames (T+P+R) to match model input
            total_latent_frames = num_t + num_p + num_r
            total_pixel_frames = (total_latent_frames - 1) * config.vae_stride[0] + 1
            y = vae.run_vae_encoder(
                img_tensor,
                new_target_video_length=total_pixel_frames,
            )
            y = y.unsqueeze(0).to(device=runtime_device, dtype=runtime_dtype)

        first_frame_latent = None
        if use_ti2v_first_frame_conditioning:
            img_tensor = (
                TF.to_tensor(first_frame)
                .sub_(0.5)
                .div_(0.5)
                .to(device=runtime_device, dtype=runtime_dtype)
            )
            first_frame_video = img_tensor.unsqueeze(0).unsqueeze(2)
            first_frame_latent = vae.encode_to_latent(first_frame_video).to(
                device=runtime_device, dtype=runtime_dtype
            )

        # ========== Prepare VACE context: [T_scene, P_scene] ==========
        # scene_proj: [C, T, h, w] - target scene projection
        # Adjust target scene projection frame count
        if scene_proj.shape[1] > num_latent_frames:
            target_scene_proj = scene_proj[:, :num_latent_frames, :, :]
        elif scene_proj.shape[1] < num_latent_frames:
            pad_frames = num_latent_frames - scene_proj.shape[1]
            last_frame = scene_proj[:, -1:, :, :].repeat(1, pad_frames, 1, 1)
            target_scene_proj = torch.cat([scene_proj, last_frame], dim=1)
        else:
            target_scene_proj = scene_proj

        # Concatenate target and preceding scene projections for VACE
        if preceding_scene_proj is not None and num_p > 0:
            # preceding_scene_proj: [C, P, h, w]
            # Adjust preceding scene projection frame count to match num_p latent frames
            if preceding_scene_proj.shape[1] > num_p:
                preceding_scene_proj_adj = preceding_scene_proj[:, :num_p, :, :]
            elif preceding_scene_proj.shape[1] < num_p:
                pad_frames = num_p - preceding_scene_proj.shape[1]
                last_frame = preceding_scene_proj[:, -1:, :, :].repeat(
                    1, pad_frames, 1, 1
                )
                preceding_scene_proj_adj = torch.cat(
                    [preceding_scene_proj, last_frame], dim=1
                )
            else:
                preceding_scene_proj_adj = preceding_scene_proj

            # VACE context: [T_scene, P_scene] concatenated along time dimension
            vace_scene_proj = torch.cat(
                [target_scene_proj, preceding_scene_proj_adj], dim=1
            )
        else:
            vace_scene_proj = target_scene_proj

        vace_context = [vace_scene_proj.to(device=runtime_device, dtype=runtime_dtype)]

        # ========== Initialize noise for target frames ==========
        noise = torch.randn(
            1,
            num_latent_frames,
            config.image_or_video_shape[2],
            h,
            w,
            device=runtime_device,
            dtype=runtime_dtype,
        )

        # ========== Setup scheduler ==========
        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=config.num_train_timestep,
            shift=1,
            use_dynamic_shifting=False,
        )
        sample_scheduler.set_timesteps(
            args.infer_steps,
            device=runtime_device,
            shift=config.timestep_shift,
        )

        latents = noise
        if first_frame_latent is not None:
            latents[:, :1].copy_(first_frame_latent)

        guidance_scale = config.guidance_scale
        timestep_dtype = sample_scheduler.timesteps.dtype
        timestep_p = (
            torch.zeros([1, num_p], device=runtime_device, dtype=timestep_dtype)
            if num_p > 0
            else None
        )
        timestep_r = (
            torch.zeros([1, num_r], device=runtime_device, dtype=timestep_dtype)
            if num_r > 0
            else None
        )

        # ========== Denoising loop ==========
        denoising_pbar = tqdm(
            enumerate(sample_scheduler.timesteps),
            total=len(sample_scheduler.timesteps),
            desc=f"Denoising (T={num_t}, P={num_p}, R={num_r})",
        )

        for step_idx, t in denoising_pbar:
            # ========== Build per-frame timestep [B, T+P+R] ==========
            # T frames: actual timestep t
            # P frames: timestep 0 (clean)
            # R frames: timestep 0 (clean)
            timestep_value = t.to(device=runtime_device, dtype=timestep_dtype)
            timestep_t = timestep_value * torch.ones(
                [1, num_t], device=runtime_device, dtype=timestep_dtype
            )
            if first_frame_latent is not None:
                timestep_t[:, 0].zero_()

            timestep_parts = [timestep_t]
            if timestep_p is not None:
                timestep_parts.append(timestep_p)
            if timestep_r is not None:
                timestep_parts.append(timestep_r)
            timestep = torch.cat(timestep_parts, dim=1)  # [1, T+P+R]

            # ========== Build input latents [B, T+P+R, C, h, w] ==========
            # [T, P, R] order: target noisy first, then preceding clean, then reference clean
            latent_parts = [latents]
            if preceding_latent is not None and num_p > 0:
                latent_parts.append(preceding_latent)
            if reference_latent is not None and num_r > 0:
                latent_parts.append(reference_latent)

            combined_latents = torch.cat(latent_parts, dim=1)

            generator_kwargs = {
                "noisy_image_or_video": combined_latents,
                "timestep": timestep,
                "context": context,
                "clip_fea": clip_fea,
                "y": y,
                "vace_context": vace_context,
                "vace_context_scale": 1.0,
                "num_t": num_t,
                "num_p": num_p,
                "num_r": num_r,
            }

            flow_pred_cond = generator(**generator_kwargs)

            if not args.no_cfg and uncon_context is not None:
                generator_kwargs["context"] = uncon_context
                flow_pred_uncond = generator(**generator_kwargs)
                flow_pred = flow_pred_uncond + guidance_scale * (
                    flow_pred_cond - flow_pred_uncond
                )
            else:
                flow_pred = flow_pred_cond

            # ========== Extract only T frames flow prediction ==========
            # flow_pred is [1, T+P+R, C, h, w], we only update T frames
            flow_pred_t = flow_pred[:, :num_t]

            latents = sample_scheduler.step(flow_pred_t, t, latents, return_dict=False)[
                0
            ]
            if first_frame_latent is not None:
                latents[:, :1].copy_(first_frame_latent)

            denoising_pbar.set_postfix(
                {
                    "timestep": f"{t:.3f}",
                    "step": f"{step_idx + 1}/{len(sample_scheduler.timesteps)}",
                }
            )

        # ========== Decode latents ==========
        videos = vae.decode_to_pixel(latents)
        videos = (videos + 1.0) / 2.0
        videos = videos.clamp(0, 1)

        # Convert to saveable format
        generated_video_tensor = videos[0]  # [T, 3, H, W]
        generated_video = (generated_video_tensor * 255).byte().cpu()
        generated_video = rearrange(generated_video, "t c h w -> t h w c")

        return generated_video
