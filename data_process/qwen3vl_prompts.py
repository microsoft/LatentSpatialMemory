"""
Qwen3-VL foreground entity extraction for the Spatia pipeline.

This module uses the DashScope API through LiteLLM instead of local
Transformers/vLLM inference.
"""

from __future__ import annotations

import base64
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from litellm import completion

from .qwen_utils import response_to_text

# Phrases that mean "no foreground entities"
_NO_ENTITY_PHRASES = (
    "no moving",
    "no motion",
    "no movement",
    "no moving object",
    "no moving objects",
    "no dynamic",
    "no foreground",
    "nothing moving",
    "nothing moves",
    "nothing is moving",
    "nothing",
    "none",
    "only background",
    "background only",
    "static scene",
    "static background",
)

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}
_STYLE_TEMPLATES = [
    ("Write a concise, literal caption.", 120, "1-2"),
    ("Write a detailed, cinematic caption.", 200, "2-4"),
    ("Write a calm, observational caption.", 180, "2-3"),
    ("Write a thorough, descriptive caption.", 250, "3-5"),
    ("Write a compact, vivid caption.", 140, "1-2"),
    ("Write a rich, immersive caption.", 180, "2-3"),
]
CLIP_ANALYSIS_MAX_NEW_TOKENS = 256


def _looks_like_no_entity(text: str) -> bool:
    if not text:
        return True
    lowered = text.strip().lower()
    return any(phrase in lowered for phrase in _NO_ENTITY_PHRASES)


def _is_null_entity(entity: str) -> bool:
    lowered = entity.strip().lower()
    if not lowered:
        return True
    if lowered in {"none", "nothing", "no", "n/a", "na"}:
        return True
    return any(phrase in lowered for phrase in _NO_ENTITY_PHRASES)


DEFAULT_PROMPT = (
    "List FOREGROUND CATEGORIES to remove, keeping ONLY static background.\n\n"
    "BACKGROUND (keep): walls, floor, ceiling, furniture, doors, windows, "
    "buildings, roads, sidewalks, trees, grass, sky, scenery, fixed structures.\n\n"
    "FOREGROUND (remove): ANY of these, even if stationary:\n"
    "- People (including hands/arms/feet)\n"
    "- Vehicles (car, truck, bus, motorcycle, bicycle, scooter, wheelchair)\n"
    "- Animals\n"
    "- Handheld/portable items (bag, phone, cup, food, umbrella)\n"
    "- Movable objects (stroller, cart, luggage, carried boxes)\n\n"
    "RULES:\n"
    "1. Output CATEGORIES, not individual instances\n"
    "2. Use AT MOST 3 categories total (merge aggressively)\n"
    "3. Pick broad categories that cover the most foreground\n"
    "4. Use generic labels. For any human, ALWAYS write exactly: person\n"
    "5. Keep items short (1-5 words). Simple location hints are ok (e.g., on the table)\n"
    "6. When unsure, LIST IT (over-remove is better than under-remove)\n\n"
    "OUTPUT FORMAT:\n"
    "Nothing\n"
    "OR numbered list (max 3 items):\n"
    "1) person\n"
    "2) car\n"
    "3) cup on the table\n\n"
    "EXAMPLES:\n"
    "Scene: office with desks, a person typing, coffee cup on desk\n"
    "1) person\n"
    "2) cup on the table\n"
    "---\n"
    "Scene: street with buildings, parked cars, two pedestrians walking\n"
    "1) person\n"
    "2) car\n"
    "---\n"
    "Scene: empty room with table and chairs\n"
    "Nothing\n"
    "---\n"
    "Scene: kitchen, person's hands visible preparing food\n"
    "1) person\n"
    "2) cup on the table\n"
)


def parse_entities(text: str) -> list[str]:
    """Parse the model output into entity labels."""
    cleaned = text.strip().strip("[]")
    cleaned = cleaned.replace("。", ".").replace("\n", ".")
    if _looks_like_no_entity(cleaned):
        return []
    parts = [p.strip() for p in cleaned.split(".") if p.strip()]
    entities = []
    for part in parts:
        part = re.sub(r"^[\d\s\-\)\(\.]+", "", part).strip()
        if not part:
            continue
        if _is_null_entity(part):
            continue
        entities.append(part)
    entities = [_canonicalize_entity(ent) for ent in entities]
    seen = set()
    unique = []
    for ent in entities:
        key = ent.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(ent)
    return unique[:3]


_PERSON_TERMS = (
    "person",
    "people",
    "human",
    "man",
    "woman",
    "child",
    "kid",
    "pedestrian",
    "hand",
    "hands",
    "arm",
    "arms",
    "leg",
    "legs",
    "head",
    "face",
    "body",
)


def _canonicalize_entity(entity: str) -> str:
    """Normalize all human-related terms into "person"."""
    lowered = entity.lower()
    if any(term in lowered for term in _PERSON_TERMS):
        return "person"
    return entity


@dataclass
class _PreparedMediaInput:
    image_urls: list[str]


@dataclass(frozen=True)
class CaptionPromptSpec:
    """Caption prompt metadata reused across pipeline entry points."""

    style_prompt: str
    max_words: int
    num_sentences: str
    prompt: str


@dataclass(frozen=True)
class ClipAnalysisResult:
    """Joint clip analysis result returned by one multimodal request."""

    entities_raw: list[str]
    foreground_status: str
    clip_caption: str
    caption_prompt: str
    caption_style_prompt: str
    caption_max_words: int
    caption_num_sentences: str
    raw_response: str


def build_caption_prompt(caption_prompt: str | None = None) -> CaptionPromptSpec:
    """Build the exact caption prompt used by run_video_captioning."""
    if caption_prompt:
        return CaptionPromptSpec(
            style_prompt="custom",
            max_words=250,
            num_sentences="custom",
            prompt=caption_prompt,
        )

    style_prompt, max_words, num_sentences = random.choice(_STYLE_TEMPLATES)
    prompt = (
        f"{style_prompt} Describe what is visible in the scene (objects, environment, people, lighting, atmosphere) "
        "and how the camera moves (e.g., panning left, moving forward, tilting up, rotating, zooming in/out, tracking). "
        "Be specific about the scene details and camera motion direction. "
        "Do NOT mention the camera operator, videographer, photographer, or any person holding/operating the camera. "
        "Use 'person' for any humans visible in the scene. "
        f"Use at most {max_words} words. Output {num_sentences} sentences."
    )
    return CaptionPromptSpec(
        style_prompt=style_prompt,
        max_words=max_words,
        num_sentences=num_sentences,
        prompt=prompt,
    )


def normalize_caption(text: str, max_words: int) -> str:
    """Trim whitespace and clamp the caption length."""
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return cleaned
    words = cleaned.split()
    if len(words) > max_words:
        cleaned = " ".join(words[:max_words])
    return cleaned


def _strip_json_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _build_clip_analysis_prompt(
    entity_prompt: str,
    caption_spec: CaptionPromptSpec,
) -> str:
    return (
        "Analyze the same video clip for two tasks and output STRICT JSON only.\n\n"
        "TASK A - foreground removal entities:\n"
        "Follow all semantic rules from the instruction below, but ignore its original output formatting because the final output must be JSON.\n"
        f"{entity_prompt}\n\n"
        "TASK B - clip caption:\n"
        "Follow the caption instruction below exactly for content, wording constraints, and sentence count.\n"
        f"{caption_spec.prompt}\n\n"
        "Return exactly one JSON object with this schema:\n"
        "{\n"
        '  "entities_raw": ["person", "car"],\n'
        '  "foreground_status": "ok",\n'
        '  "clip_caption": "A person walks through a bright hallway while the camera moves forward."\n'
        "}\n\n"
        "Rules:\n"
        '- "entities_raw" must be an array of strings with at most 3 items.\n'
        "- Use broad removable foreground categories only.\n"
        '- Use "foreground_status": "nothing" when there is no removable foreground entity, and then return an empty list for "entities_raw".\n'
        '- Use "foreground_status": "ok" when at least one removable foreground entity exists.\n'
        '- "clip_caption" must be plain text with no markdown.\n'
        "- Do not output any text before or after the JSON object.\n"
    )


class Qwen3VLClient:
    """Shared multimodal Qwen client used by entity extraction and captioning."""

    def __init__(
        self,
        model_path: str,
        api_base: str | None = None,
        api_key_env: str = "DASHSCOPE_API_KEY",
        max_new_tokens: int = 128,
        video_fps: float = 2.0,
        video_min_frames: int = 4,
        video_max_frames: int = 10,
    ) -> None:
        self.model_path = model_path
        self.api_base = api_base
        self.api_key = os.environ.get(api_key_env)
        self.max_new_tokens = max_new_tokens
        self.video_fps = video_fps
        self.video_min_frames = video_min_frames
        self.video_max_frames = video_max_frames
        self.max_retries = 3

    def _prepare_media_input(self, input_path: str) -> _PreparedMediaInput:
        ext = os.path.splitext(input_path)[1].lower()
        if ext in _VIDEO_EXTENSIONS:
            return self._prepare_video_input(input_path)
        return _PreparedMediaInput(
            image_urls=[self._encode_image_as_data_url(Path(input_path))]
        )

    def _prepare_video_input(self, input_path: str) -> _PreparedMediaInput:
        cap = cv2.VideoCapture(input_path)
        image_urls: list[str] = []
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if frame_count <= 0:
                raise RuntimeError(f"Failed to read video metadata from {input_path}")

            video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_indices = self._sample_video_indices(frame_count, video_fps)

            for frame_index in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    continue
                ok, encoded = cv2.imencode(".png", frame_bgr)
                assert ok, f"Failed to encode frame {frame_index} from {input_path}"
                image_urls.append(
                    self._encode_bytes_as_data_url(
                        encoded.tobytes(),
                        "image/png",
                    )
                )

            if not image_urls:
                raise RuntimeError(f"Failed to decode any frames for {input_path}")

            return _PreparedMediaInput(image_urls=image_urls)
        finally:
            cap.release()

    def _sample_video_indices(
        self,
        frame_count: int,
        video_fps: float,
    ) -> np.ndarray:
        if frame_count <= 0:
            raise ValueError(f"Invalid frame count: {frame_count}")

        if video_fps > 0:
            sampled_count = int(frame_count / video_fps * self.video_fps)
            sampled_count = min(
                min(max(sampled_count, self.video_min_frames), self.video_max_frames),
                frame_count,
            )
        else:
            sampled_count = min(
                max(frame_count, self.video_min_frames),
                self.video_max_frames,
            )
            sampled_count = min(sampled_count, frame_count)

        sampled_count = max(1, sampled_count)
        return np.linspace(0, frame_count - 1, sampled_count).round().astype(np.int32)

    @staticmethod
    def _build_messages(image_urls: Sequence[str], prompt: str) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image_url in image_urls:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                }
            )
        return [{"role": "user", "content": content}]

    @staticmethod
    def _encode_image_as_data_url(image_path: Path) -> str:
        suffix = image_path.suffix.lower()
        mime_type = _IMAGE_MIME_TYPES.get(suffix, "image/png")
        return Qwen3VLClient._encode_bytes_as_data_url(
            image_path.read_bytes(),
            mime_type,
        )

    @staticmethod
    def _encode_bytes_as_data_url(image_bytes: bytes, mime_type: str) -> str:
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def _complete_prepared(
        self,
        prepared: _PreparedMediaInput,
        prompt: str,
        max_new_tokens: int | None = None,
    ) -> str:
        messages = self._build_messages(prepared.image_urls, prompt)
        response = completion(
            model=self.model_path,
            messages=messages,
            api_base=self.api_base,
            api_key=self.api_key,
            max_tokens=max_new_tokens or self.max_new_tokens,
            temperature=0.0,
        )
        return response_to_text(response)

    def complete(
        self,
        input_path: str,
        prompt: str,
    ) -> tuple[str, str | None]:
        prepared = self._prepare_media_input(input_path)
        messages = self._build_messages(prepared.image_urls, prompt)
        response = completion(
            model=self.model_path,
            messages=messages,
            api_base=self.api_base,
            api_key=self.api_key,
            max_tokens=self.max_new_tokens,
            temperature=0.0,
        )
        output_text = response_to_text(response)
        finish_reason = response.choices[0].finish_reason
        return output_text, finish_reason


class Qwen3VLEntityExtractor:
    """Foreground entity extractor backed by LiteLLM."""

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 128,
        video_fps: float = 2.0,
        video_min_frames: int = 4,
        video_max_frames: int = 10,
        api_base: str | None = None,
        api_key_env: str = "DASHSCOPE_API_KEY",
    ) -> None:
        self.client = Qwen3VLClient(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            video_fps=video_fps,
            video_min_frames=video_min_frames,
            video_max_frames=video_max_frames,
            api_base=api_base,
            api_key_env=api_key_env,
        )
        self.max_retries = self.client.max_retries

    def extract(
        self, input_path: str, prompt: str = DEFAULT_PROMPT
    ) -> tuple[list[str], str]:
        """Extract entities from one input video or image."""
        return self.extract_many([input_path], prompt=prompt)[0]

    def extract_many(
        self, input_paths: Sequence[str], prompt: str = DEFAULT_PROMPT
    ) -> list[tuple[list[str], str]]:
        """Extract entities from multiple inputs."""
        prepared_inputs = [
            self.client._prepare_media_input(path) for path in input_paths
        ]
        return [
            self._extract_dashscope(prepared=prepared, prompt=prompt)
            for prepared in prepared_inputs
        ]

    def _extract_dashscope(
        self, prepared: _PreparedMediaInput, prompt: str
    ) -> tuple[list[str], str]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                output_text = self.client._complete_prepared(prepared, prompt=prompt)
                entities = parse_entities(output_text)
                if not entities:
                    return [], "Nothing"
                return entities, output_text
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(1.0 + attempt)

        raise last_error

    def analyze_clip(
        self,
        input_path: str,
        prompt: str = DEFAULT_PROMPT,
        caption_prompt: str | None = None,
    ) -> ClipAnalysisResult:
        """Analyze one clip for both foreground entities and caption."""
        return self.analyze_clip_many(
            [input_path],
            prompt=prompt,
            caption_prompt=caption_prompt,
        )[0]

    def analyze_clip_many(
        self,
        input_paths: Sequence[str],
        prompt: str = DEFAULT_PROMPT,
        caption_prompt: str | None = None,
    ) -> list[ClipAnalysisResult]:
        """Analyze multiple clips with one request per clip."""
        prepared_inputs = [
            self.client._prepare_media_input(path) for path in input_paths
        ]
        return [
            self._analyze_clip_dashscope(
                prepared=prepared,
                entity_prompt=prompt,
                caption_spec=build_caption_prompt(
                    caption_prompt=caption_prompt,
                ),
            )
            for prepared in prepared_inputs
        ]

    def _analyze_clip_dashscope(
        self,
        prepared: _PreparedMediaInput,
        entity_prompt: str,
        caption_spec: CaptionPromptSpec,
    ) -> ClipAnalysisResult:
        prompt = _build_clip_analysis_prompt(entity_prompt, caption_spec)
        last_error: Exception | None = None
        max_new_tokens = max(
            self.client.max_new_tokens,
            CLIP_ANALYSIS_MAX_NEW_TOKENS,
        )
        for attempt in range(self.max_retries):
            try:
                output_text = self.client._complete_prepared(
                    prepared,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                )
                return self._parse_clip_analysis_output(output_text, caption_spec)
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(1.0 + attempt)
        raise last_error

    @staticmethod
    def _parse_clip_analysis_output(
        output_text: str,
        caption_spec: CaptionPromptSpec,
    ) -> ClipAnalysisResult:
        payload = json.loads(_strip_json_code_fence(output_text))
        if not isinstance(payload, dict):
            raise ValueError("Clip analysis response must be a JSON object.")

        raw_entities_value = payload.get("entities_raw")
        if not isinstance(raw_entities_value, list):
            raise ValueError("Clip analysis JSON must contain list field entities_raw.")
        raw_entities: list[str] = []
        seen_entities: set[str] = set()
        for item in raw_entities_value:
            entity = str(item).strip()
            if not entity or _is_null_entity(entity):
                continue
            entity = _canonicalize_entity(entity)
            entity_key = entity.lower()
            if entity_key in seen_entities:
                continue
            seen_entities.add(entity_key)
            raw_entities.append(entity)

        foreground_status = str(payload.get("foreground_status") or "").strip().lower()
        if not foreground_status:
            foreground_status = "nothing" if not raw_entities else "ok"
        if foreground_status not in {"ok", "nothing"}:
            raise ValueError(
                f"Unsupported foreground_status in clip analysis response: {foreground_status}"
            )

        clip_caption = normalize_caption(
            str(payload.get("clip_caption") or ""),
            max_words=caption_spec.max_words,
        )
        if not clip_caption:
            raise ValueError("Clip analysis JSON must contain non-empty clip_caption.")

        return ClipAnalysisResult(
            entities_raw=raw_entities,
            foreground_status=foreground_status,
            clip_caption=clip_caption,
            caption_prompt=caption_spec.prompt,
            caption_style_prompt=caption_spec.style_prompt,
            caption_max_words=caption_spec.max_words,
            caption_num_sentences=caption_spec.num_sentences,
            raw_response=output_text,
        )
