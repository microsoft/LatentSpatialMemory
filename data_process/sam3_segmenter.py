"""
SAM3 视频分割模块

使用SAM3 (Segment Anything 3) 对视频进行分割
根据文字prompt (如 "person", "car") 分割对应的前景物体
输出每帧的binary mask
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from PIL import Image
from sam3.model_builder import build_sam3_video_predictor

# 表示"无需分割"的prompt短语
_NULL_PROMPT_PHRASES = (
    "nothing",
    "none",
    "no moving",
    "no motion",
    "no movement",
    "no moving object",
    "no moving objects",
    "no dynamic",
    "only background",
    "background only",
    "static scene",
    "static background",
)


@dataclass
class Sam3VideoSegmenter:
    """
    SAM3视频分割器

    根据文字prompt分割视频中的物体，支持:
    - 单prompt逐个分割后合并
    - 多prompt一次性分割 (multi_prompt=True)
    - 双向传播 (forward/backward/both)
    - mask膨胀 (mask_dilate)
    """

    gpus_to_use: Optional[list[int]] = None
    propagation_direction: str = "both"  # forward/backward/both
    mask_dilate: int = 0  # mask膨胀像素数
    score_threshold_detection: Optional[float] = None
    new_det_thresh: Optional[float] = None
    checkpoint_path: Optional[str] = None
    multi_prompt: bool = False
    resize_input_to: Optional[int] = None

    def __post_init__(self):
        self.predictor = build_sam3_video_predictor(
            gpus_to_use=self.gpus_to_use,
            checkpoint_path=self.checkpoint_path,
        )
        self._apply_model_overrides()

    def _apply_model_overrides(self) -> None:
        model = getattr(self.predictor, "model", None)
        if model is None:
            return
        if self.score_threshold_detection is not None and hasattr(
            model, "score_threshold_detection"
        ):
            model.score_threshold_detection = float(self.score_threshold_detection)
        if self.new_det_thresh is not None and hasattr(model, "new_det_thresh"):
            model.new_det_thresh = float(self.new_det_thresh)

    @staticmethod
    def normalize_prompts(prompts: Iterable[str]) -> list[str]:
        normalized = []
        seen = set()
        for prompt in prompts:
            if prompt is None:
                continue
            text = str(prompt).strip()
            if not text:
                continue
            lowered = text.lower()
            if lowered in {"nothing", "none"} or any(
                p in lowered for p in _NULL_PROMPT_PHRASES
            ):
                continue
            if lowered in seen:
                continue
            seen.add(lowered)
            normalized.append(text)
        return normalized

    def _segment_single_prompt(
        self,
        video_resource: str | list[Image.Image],
        prompt: str,
        frame_index: int,
        start_frame_index: int | None,
        max_frame_num_to_track: int | None,
        expected_frames: int | None,
    ) -> np.ndarray:
        response = self.predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=video_resource,
            )
        )
        session_id = response["session_id"]

        masks_by_frame: dict[int, np.ndarray] = {}
        try:
            self.predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=frame_index,
                    text=prompt,
                    obj_id=1,
                )
            )

            for response in self.predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                    propagation_direction=self.propagation_direction,
                    start_frame_index=start_frame_index,
                    max_frame_num_to_track=max_frame_num_to_track,
                )
            ):
                response["prompt"] = prompt
                frame_idx = int(response["frame_index"])
                outputs = response["outputs"]
                masks = outputs["out_binary_masks"]
                if hasattr(masks, "detach"):
                    masks = masks.detach().cpu().numpy()
                else:
                    masks = np.asarray(masks)
                if masks.size == 0:
                    merged = None
                else:
                    merged = np.any(masks, axis=0)

                if merged is None:
                    if frame_idx not in masks_by_frame:
                        masks_by_frame[frame_idx] = None
                else:
                    if (
                        frame_idx in masks_by_frame
                        and masks_by_frame[frame_idx] is not None
                    ):
                        masks_by_frame[frame_idx] = masks_by_frame[frame_idx] | merged
                    else:
                        masks_by_frame[frame_idx] = merged
        finally:
            self.predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=session_id,
                )
            )

        result = self._finalize_masks(masks_by_frame, expected_frames)
        masks_by_frame.clear()
        return result

    def _segment_multi_prompt(
        self,
        video_resource: str | list[Image.Image],
        prompts: Iterable[str],
        frame_index: int,
        start_frame_index: int | None,
        max_frame_num_to_track: int | None,
        expected_frames: int | None,
    ) -> np.ndarray:
        response = self.predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=video_resource,
            )
        )
        session_id = response["session_id"]

        masks_by_frame: dict[int, np.ndarray] = {}
        try:
            prompt_list = list(prompts)
            for obj_id, prompt in enumerate(prompt_list, start=1):
                self.predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=frame_index,
                        text=prompt,
                        obj_id=obj_id,
                    )
                )

            for response in self.predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                    propagation_direction=self.propagation_direction,
                    start_frame_index=start_frame_index,
                    max_frame_num_to_track=max_frame_num_to_track,
                )
            ):
                response["prompts"] = prompt_list
                frame_idx = int(response["frame_index"])
                outputs = response["outputs"]
                masks = outputs["out_binary_masks"]
                if hasattr(masks, "detach"):
                    masks = masks.detach().cpu().numpy()
                else:
                    masks = np.asarray(masks)
                if masks.size == 0:
                    merged = None
                else:
                    merged = np.any(masks, axis=0)

                if merged is None:
                    if frame_idx not in masks_by_frame:
                        masks_by_frame[frame_idx] = None
                else:
                    if (
                        frame_idx in masks_by_frame
                        and masks_by_frame[frame_idx] is not None
                    ):
                        masks_by_frame[frame_idx] = masks_by_frame[frame_idx] | merged
                    else:
                        masks_by_frame[frame_idx] = merged
        finally:
            self.predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=session_id,
                )
            )

        result = self._finalize_masks(masks_by_frame, expected_frames)
        masks_by_frame.clear()
        return result

    @staticmethod
    def _finalize_masks(
        masks_by_frame: dict[int, np.ndarray],
        expected_frames: int | None,
    ) -> np.ndarray:
        if not masks_by_frame:
            return np.zeros((0, 0, 0), dtype=bool)

        max_idx = max(masks_by_frame.keys())
        num_frames = expected_frames if expected_frames is not None else max_idx + 1

        sample_mask = next((m for m in masks_by_frame.values() if m is not None), None)
        if sample_mask is None:
            return np.zeros((0, 0, 0), dtype=bool)
        H, W = sample_mask.shape

        masks_out = np.zeros((num_frames, H, W), dtype=bool)
        for idx, mask in masks_by_frame.items():
            if mask is None:
                continue
            if idx < num_frames:
                masks_out[idx] = mask

        if not masks_out.any():
            return np.zeros((0, 0, 0), dtype=bool)

        return masks_out

    @staticmethod
    def _resize_masks_to_hw(
        masks: np.ndarray, target_hw: tuple[int, int]
    ) -> np.ndarray:
        import cv2

        H, W = target_hw
        resized = []
        for mask in masks:
            if mask.shape != (H, W):
                mask_u8 = mask.astype(np.uint8) * 255
                mask_u8 = cv2.resize(mask_u8, (W, H), interpolation=cv2.INTER_NEAREST)
                mask = mask_u8 > 0
            resized.append(mask)
        return np.stack(resized, axis=0)

    @staticmethod
    def _letterbox_frames(
        frames: list[np.ndarray],
        target_size: int,
    ) -> tuple[list[np.ndarray], tuple[int, int], tuple[int, int], tuple[int, int]]:
        import cv2

        if not frames:
            return [], (0, 0), (0, 0), (0, 0)
        orig_h, orig_w = frames[0].shape[:2]
        scale = target_size / float(max(orig_h, orig_w))
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        pad_x = max(0, (target_size - new_w) // 2)
        pad_y = max(0, (target_size - new_h) // 2)
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        padded_frames = []
        for frame in frames:
            resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)
            canvas = np.zeros((target_size, target_size, 3), dtype=resized.dtype)
            canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
            padded_frames.append(canvas)
        return padded_frames, (orig_h, orig_w), (new_h, new_w), (pad_y, pad_x)

    def _restore_masks_from_letterbox(
        self,
        masks: np.ndarray,
        orig_hw: tuple[int, int],
        resized_hw: tuple[int, int],
        pad_xy: tuple[int, int],
    ) -> np.ndarray:
        pad_y, pad_x = pad_xy
        resized_h, resized_w = resized_hw
        if masks.size == 0:
            return masks
        cropped = masks[:, pad_y : pad_y + resized_h, pad_x : pad_x + resized_w]
        return self._resize_masks_to_hw(cropped, orig_hw)

    def _release_gpu_memory(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def segment(
        self,
        video_path: str,
        prompts: Iterable[str],
        frame_index: int = 0,
        start_frame_index: int | None = None,
        max_frame_num_to_track: int | None = None,
        expected_frames: int | None = None,
        output_dir: str | Path | None = None,
    ) -> np.ndarray:
        prompts = self.normalize_prompts(prompts)
        if not prompts:
            return np.zeros((0, 0, 0), dtype=bool)

        video_resource: str | list[Image.Image] = video_path
        resize_meta = None
        pil_frames: list[Image.Image] | None = None
        if self.resize_input_to:
            from data_process.video_io import load_video_frames

            frames = load_video_frames(video_path)
            padded_frames, orig_hw, resized_hw, pad_xy = self._letterbox_frames(
                frames, self.resize_input_to
            )
            pil_frames = [Image.fromarray(frame) for frame in padded_frames]
            video_resource = pil_frames
            expected_frames = len(pil_frames)
            resize_meta = (orig_hw, resized_hw, pad_xy)
            del frames, padded_frames

        try:
            if self.multi_prompt and len(prompts) > 1:
                merged_masks = self._segment_multi_prompt(
                    video_resource=video_resource,
                    prompts=prompts,
                    frame_index=frame_index,
                    start_frame_index=start_frame_index,
                    max_frame_num_to_track=max_frame_num_to_track,
                    expected_frames=expected_frames,
                )
            else:
                merged_masks = None
                for prompt in prompts:
                    masks = self._segment_single_prompt(
                        video_resource=video_resource,
                        prompt=prompt,
                        frame_index=frame_index,
                        start_frame_index=start_frame_index,
                        max_frame_num_to_track=max_frame_num_to_track,
                        expected_frames=expected_frames,
                    )
                    if masks.size == 0:
                        continue
                    if merged_masks is None:
                        merged_masks = masks
                    else:
                        merged_masks = merged_masks | masks
        finally:
            if pil_frames is not None:
                for img in pil_frames:
                    img.close()
                pil_frames.clear()
                del pil_frames
            self._release_gpu_memory()

        if merged_masks is None or not merged_masks.any():
            return np.zeros((0, 0, 0), dtype=bool)

        if resize_meta is not None:
            orig_hw, resized_hw, pad_xy = resize_meta
            merged_masks = self._restore_masks_from_letterbox(
                merged_masks, orig_hw, resized_hw, pad_xy
            )

        if self.mask_dilate > 0:
            import cv2

            kernel = np.ones((3, 3), dtype=np.uint8)
            dilated = []
            for frame_mask in merged_masks:
                mask_u8 = frame_mask.astype(np.uint8)
                mask_u8 = cv2.dilate(mask_u8, kernel, iterations=self.mask_dilate)
                dilated.append(mask_u8.astype(bool))
            merged_masks = np.stack(dilated, axis=0)

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            for idx in range(merged_masks.shape[0]):
                mask_u8 = (merged_masks[idx].astype(np.uint8)) * 255
                Image.fromarray(mask_u8).save(output_dir / f"{idx:05d}.png")

        return merged_masks
