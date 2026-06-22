from typing import List
import os
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import hydra
from hydra.core.global_hydra import GlobalHydra
import re

from core.raft import RAFT
from core.utils.utils import load_ckpt
from core.parser import parse_args

# Grounding DINO
import argparse
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# Load spaCy's small English model
# make sure to run: python -m spacy download en_core_web_sm
import spacy
nlp = spacy.load("en_core_web_sm")

from worldscore.benchmark.metrics.base_metrics import BaseMetric


def mask_resize(mask, target_size):
    """
    Resize mask to target size while preserving the binary mask format
    
    Args:
        mask (np.ndarray): Binary mask of shape (H, W)
        target_size (tuple): Target size (H, W)
        
    Returns:
        np.ndarray: Resized binary mask of shape target_size
    """
    
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    if isinstance(target_size, torch.Size):
        target_size = tuple(target_size)
    
    mask = (mask > 0.5).astype(np.uint8)
    
    resized_mask = cv2.resize(
        mask,
        (target_size[1], target_size[0]), 
        interpolation=cv2.INTER_NEAREST
    )
    
    resized_mask = (resized_mask > 0.5).astype(np.uint8)
    
    return resized_mask

def load_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cuda"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def standardize_string(s):
    # Use a regular expression to remove spaces around special symbols
    # Adjust the characters in the square brackets as needed
    s = re.sub(r"\s*([_\-*/+])\s*", r"\1", s)  
    return s

def extract_noun_adjective(phrase):
    doc = nlp(phrase)
    adjectives = []
    nouns = []
    
    for token in doc:
        if token.pos_ in ["ADJ", "JJ", "JJR", "JJS"]:  # Adjectives and related tags
            adjectives.append(token.text.lower())
        elif token.pos_ in ["NOUN", "PROPN", "NN", "NNS", "NNP", "NNPS"]:  # Nouns and related tags
            nouns.append(token.text.lower())
        elif token.pos_ == "VERB" and token.tag_ == "VBN":  # Past participle verbs often used as adjectives
            adjectives.append(token.text.lower())

    return adjectives, nouns

def get_match_point(result, prompt_list):
    matched_prompts = set()
    
    if "##" in result:
        result = result.replace("##", "")
        for prompt in prompt_list:
            if result in prompt:
                matched_prompts.add(prompt)
                return 1, matched_prompts
    else:
        result_adjectives, result_nouns = extract_noun_adjective(result)
        for prompt in prompt_list:
            prompt_adjectives, prompt_nouns = extract_noun_adjective(prompt)
            
            # Count matches for nouns
            noun_matches = sum(1 for noun in result_nouns if noun in prompt_nouns)
            # If no nouns match, skip this prompt
            if noun_matches == 0:
                continue

            matched_prompts.add(prompt)
            return 1, matched_prompts

    return 0, matched_prompts

def load_ann_png(path):
    """Load a PNG file as a mask and its palette."""
    mask = Image.open(path)
    palette = mask.getpalette()
    mask = np.array(mask).astype(np.uint8)
    return mask, palette


def get_per_obj_mask(mask):
    """Split a mask into per-object masks."""
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids > 0].tolist()
    per_obj_mask = {object_id: (mask == object_id) for object_id in object_ids}
    return per_obj_mask


def put_per_obj_mask(per_obj_mask, height, width):
    """Combine per-object masks into a single mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    object_ids = sorted(per_obj_mask)[::-1]
    for object_id in object_ids:
        object_mask = per_obj_mask[object_id]
        object_mask = object_mask.reshape(height, width)
        mask[object_mask] = object_id
    return mask


def load_masks_from_dir(
    input_mask_path, per_obj_png_file, allow_missing=False
):
    """Load masks from a directory as a dict of per-object masks."""
    if not per_obj_png_file:
        if allow_missing and not os.path.exists(input_mask_path):
            return {}, None
        input_mask, input_palette = load_ann_png(input_mask_path)
        per_obj_input_mask = get_per_obj_mask(input_mask)
    else:
        per_obj_input_mask = {}
        input_palette = None
        # each object is a directory in "{object_id:%03d}" format
        for object_id, file_name in enumerate(sorted([f for f in os.listdir(input_mask_path) if f.endswith(".png") or f.endswith(".jpg")])):
            input_mask_path_ = os.path.join(
                input_mask_path, file_name
            )
            if allow_missing and not os.path.exists(input_mask_path_):
                continue
            input_mask, input_palette = load_ann_png(input_mask_path_)
            per_obj_input_mask[object_id] = input_mask > 0

    return per_obj_input_mask, input_palette


def generate_masks(
    per_obj_output_mask,
    height,
    width,
    per_obj_png_file,
):
    """Save masks to a directory as PNG files."""
    if not per_obj_png_file:
        output_mask = put_per_obj_mask(per_obj_output_mask, height, width).astype(np.bool_)
    else:
        output_mask = np.zeros((height, width), dtype=np.uint8)
        for _, object_mask in per_obj_output_mask.items():
            output_mask_tmp = object_mask.reshape(height, width).astype(np.bool_)
            output_mask |= output_mask_tmp
    return output_mask


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def vos_inference(
    predictor,
    video_dir,
    input_mask,
    score_thresh=0.0,
    per_obj_png_file=False,
    save=False,
):
    """Run VOS inference on a single video with the given predictor."""
    # load the video frames and initialize the inference state on this video
    frame_names = [
        os.path.splitext(p)[0]
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    inference_state = predictor.init_state(
        video_path=video_dir, async_loading_frames=False
    )
    height = inference_state["video_height"]
    width = inference_state["video_width"]

    input_frame_inds = [0]

    # add those input masks to SAM 2 inference state before propagation
    object_ids_set = None
    for input_frame_idx in input_frame_inds:
        try:
            per_obj_input_mask, input_palette = load_masks_from_dir(
                input_mask_path=input_mask,
                per_obj_png_file=per_obj_png_file,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Failed to load input mask for frame {input_frame_idx=}. "
                "Please add the `--track_object_appearing_later_in_video` flag "
                "for VOS datasets that don't have all objects to track appearing "
                "in the first frame (such as LVOS or YouTube-VOS)."
            ) from e
        # get the list of object ids to track from the first input frame
        if object_ids_set is None:
            object_ids_set = set(per_obj_input_mask)
        for object_id, object_mask in per_obj_input_mask.items():
            # check and make sure no new object ids appear only in later frames
            if object_id not in object_ids_set:
                raise RuntimeError(
                    f"Got a new {object_id=} appearing only in a "
                    f"later {input_frame_idx=} (but not appearing in the first frame). "
                    "Please add the `--track_object_appearing_later_in_video` flag "
                    "for VOS datasets that don't have all objects to track appearing "
                    "in the first frame (such as LVOS or YouTube-VOS)."
                )
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=input_frame_idx,
                obj_id=object_id,
                mask=object_mask,
            )

    # check and make sure we have at least one object to track
    if object_ids_set is None or len(object_ids_set) == 0:
        raise RuntimeError(
            f"Got no object ids on {input_frame_inds=}. "
            "Please add the `--track_object_appearing_later_in_video` flag "
            "for VOS datasets that don't have all objects to track appearing "
            "in the first frame (such as LVOS or YouTube-VOS)."
        )
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        per_obj_output_mask = {
            out_obj_id: (out_mask_logits[i] > score_thresh).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        video_segments[out_frame_idx] = per_obj_output_mask
    
    def expand_mask(mask, pixels=10):
        """Expand the true part of the mask by n pixels"""
        kernel = np.ones((pixels*2+1, pixels*2+1), np.uint8)
        dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        return dilated_mask.astype(np.bool_)

    masks = []
    # write the output masks as palette PNG files to output_mask_dir
    for out_frame_idx, per_obj_output_mask in video_segments.items():
        output_mask = generate_masks(
            per_obj_output_mask=per_obj_output_mask,
            height=height,
            width=width,
            per_obj_png_file=per_obj_png_file,
        )
        
        output_mask = expand_mask(output_mask)
        masks.append(output_mask)
        
        if save:
            from torchvision.transforms import ToPILImage
            mask = torch.from_numpy(output_mask).unsqueeze(0)
            mask = ToPILImage()(mask.float()).save(f"mask_{out_frame_idx}.png")
        
    return masks

class MotionAccuracyMetric(BaseMetric):
    """
    Using the difference of median of estimated optical-flow to measure the motion accuracy.
    """

    def __init__(self, generate_type: str) -> None:
        super().__init__()
        
        args = {
            "cfg": "worldscore/benchmark/metrics/third_party/SEA-RAFT/config/eval/spring-M.json",
            "path": "worldscore/benchmark/metrics/checkpoints/Tartan-C-T-TSKH-spring540x960-M.pth",
        }
        args = argparse.Namespace(**args)
        args = parse_args(args)
        
        # load optical flow model
        model = RAFT(args)
        load_ckpt(model, args.path)
        model.to(self._device)
        model.eval()
        self._optical_flow_model = model
        self._optical_flow_args = args

        self._score_thresh = 0.0
        self._per_obj_png_file = True

        GlobalHydra.instance().clear()
        hydra.initialize(config_path=".")
        
        # if we use per-object PNG files, they could possibly overlap in inputs and outputs
        hydra_overrides_extra = [
            "++model.non_overlap_masks=" + ("false" if self._per_obj_png_file else "true")
        ]
        self._predictor = build_sam2_video_predictor(
            config_file="sam2/configs/sam2.1/sam2.1_hiera_b+.yaml",
            ckpt_path="worldscore/benchmark/metrics/checkpoints/sam2.1_hiera_base_plus.pt",
            apply_postprocessing=False,
            hydra_overrides_extra=hydra_overrides_extra,
        )
        
        self._generate_type = generate_type
        if self._generate_type == "t2v":
            # Grounding DINO
            args = {
                "config_file": "worldscore/benchmark/metrics/third_party/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                "grounded_checkpoint": "worldscore/benchmark/metrics/checkpoints/groundingdino_swint_ogc.pth",
                "sam_version": "vit_h",
                "sam_checkpoint": "worldscore/benchmark/metrics/checkpoints/sam_vit_h_4b8939.pth",
                "sam_hq_checkpoint": None,
                "use_sam_hq": False,
                "box_threshold": 0.4,
                "text_threshold": 0.4,
                "bert_base_uncased_path": None,
                "device": "cuda",
            }
            args = argparse.Namespace(**args)
            
            # load model        
            self._grounding_dino_model = load_model(args.config_file, args.grounded_checkpoint, args.bert_base_uncased_path, args.device)
            self._grounding_dino_args = args
    
    def load_image(self, imfile):
        image = cv2.imread(imfile)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        image = image[None].to(self._device)
        return image
    
    def forward_flow(self, image1, image2):
        with torch.amp.autocast(device_type="cuda"):
            output = self._optical_flow_model(image1, image2, iters=self._optical_flow_args.iters, test_mode=True)
        flow_final = output['flow'][-1]
        info_final = output['info'][-1]
        return flow_final, info_final

    def _compute_flow(self, image1, image2):
        print(f"computing flow...")
        
        img1 = F.interpolate(image1, scale_factor=2 ** self._optical_flow_args.scale, mode='bilinear', align_corners=False)
        img2 = F.interpolate(image2, scale_factor=2 ** self._optical_flow_args.scale, mode='bilinear', align_corners=False)
        H, W = img1.shape[2:]
        flow, info = self.forward_flow(img1, img2)
        flow_down = F.interpolate(flow, scale_factor=0.5 ** self._optical_flow_args.scale, mode='bilinear', align_corners=False) * (0.5 ** self._optical_flow_args.scale)
        info_down = F.interpolate(info, scale_factor=0.5 ** self._optical_flow_args.scale, mode='area')
        
        flow = flow_down.cpu().numpy().squeeze().transpose(1, 2, 0)
        return flow
    
    def _compute_scores(
        self, 
        rendered_images: List[str],
        masked_images: List[str],
        objects_list: List[str],
    ) -> float:

        video_dir = os.path.dirname(rendered_images[0])
        mask_dir = os.path.dirname(masked_images[0])
            
        masks = vos_inference(
            predictor=self._predictor,
            video_dir=video_dir,
            input_mask=mask_dir,
            score_thresh=self._score_thresh,
            per_obj_png_file=self._per_obj_png_file,
        )
        
        scores = []
        
        with torch.no_grad():
            images = rendered_images
            for i, (imfile1, imfile2, mask) in enumerate(zip(images[:-1], images[1:], masks[:-1])):
                print(f"processing {imfile1}, {imfile2}...")
                image1, image2 = self.load_image(imfile1), self.load_image(imfile2)
                if mask.shape != image1.shape[2:]:
                    mask = mask_resize(mask, image1.shape[2:])
                flow = self._compute_flow(image1, image2)
                flow_magnitude = np.sqrt((flow[..., 0] ** 2 + flow[..., 1] ** 2))   
                object_flow_magnitude = flow_magnitude[mask]
                background_flow_magnitude = flow_magnitude[~mask]
                motion_alignment = torch.from_numpy(object_flow_magnitude).max() - torch.from_numpy(background_flow_magnitude).max()
                scores.append(float(motion_alignment.item()))
            
        score = sum(scores) / len(scores)
        return score
    

