"""
Qwen3-VL 前景物体识别模块

使用Qwen3-VL多模态模型识别视频中的前景/动态物体
输出物体描述列表，如 ["person", "car", "cup on the table"]
这些描述将被用于SAM3分割
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# 表示"没有前景物体"的短语
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
    """解析模型输出，提取物体描述列表"""
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
    # Deduplicate while preserving order
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
    """标准化实体名称: 所有人相关的词统一为"person" """
    lowered = entity.lower()
    if any(term in lowered for term in _PERSON_TERMS):
        return "person"
    return entity


@dataclass
class Qwen3VLEntityExtractor:
    """Qwen3-VL前景物体提取器"""

    model_path: str
    device: str
    max_new_tokens: int = 128
    attn_implementation: str = "flash_attention_2"

    def __post_init__(self):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            attn_implementation=self.attn_implementation,
            device_map={"": self.device},
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def extract(
        self, input_path: str, prompt: str = DEFAULT_PROMPT
    ) -> tuple[list[str], str]:
        """
        从视频或图片中提取前景物体描述，返回(实体列表, 原始输出文本)

        自动检测输入类型：
        - .mp4/.avi/.mov 等视频文件：使用视频模式（需要 >= 2帧）
        - .png/.jpg/.jpeg 等图片文件：使用图片模式
        - 对于单帧视频：先提取第一帧，然后使用图片模式
        """
        import tempfile

        import cv2

        # 判断输入类型
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

        ext = os.path.splitext(input_path)[1].lower()

        if ext in image_extensions:
            # 图片模式
            content_type = "image"
            content_path = input_path
            temp_image_path = None
        elif ext in video_extensions:
            # 检查视频帧数
            cap = cv2.VideoCapture(input_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if frame_count >= 2:
                # 多帧视频：使用视频模式
                cap.release()
                content_type = "video"
                content_path = input_path
                temp_image_path = None
            else:
                # 单帧视频：提取第一帧，使用图片模式
                ret, frame = cap.read()
                cap.release()

                if not ret or frame is None:
                    return [], "Failed to read video frame"

                # 保存临时图片
                temp_image_path = tempfile.NamedTemporaryFile(
                    suffix=".png", delete=False
                ).name
                cv2.imwrite(temp_image_path, frame)
                content_type = "image"
                content_path = temp_image_path
        else:
            # 未知类型，尝试作为图片
            content_type = "image"
            content_path = input_path
            temp_image_path = None

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": content_type, content_type: str(content_path)},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            generated_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            entities = parse_entities(output_text)
            if not entities:
                return [], "Nothing"
            return entities, output_text
        finally:
            # 清理临时文件
            if temp_image_path and os.path.exists(temp_image_path):
                os.remove(temp_image_path)
