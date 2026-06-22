import argparse
import re

import torch
from PIL import Image
from typing import List

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

from worldscore.benchmark.metrics.base_metrics import BaseMetric


# Load spaCy's small English model
# make sure to run: python -m spacy download en_core_web_sm
import spacy
nlp = spacy.load("en_core_web_sm")


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

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
    
class ObjectDetectionMetric(BaseMetric):
    """
    Object Detection Metric
    Return: detection success rate
    Range: [0, 1] higher the better
    """
    
    def __init__(self) -> None:
        super().__init__()
        
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
        self._model = load_model(args.config_file, args.grounded_checkpoint, args.bert_base_uncased_path, args.device)
        self._args = args
        
    def _compute_scores(
        self, 
        rendered_images: List[str], 
        text_prompt: str
    ) -> float:
        
        prompt_list = text_prompt.split(", ")
        prompt_list = prompt_list[1:]
        prompt_list = [p.lower() for p in prompt_list]
        prompt_string = ""
        for prompt in prompt_list:
            if prompt_string != "":
                prompt_string += ", "
            prompt_string += prompt 
        text_prompt = prompt_string
        
        scores = []
        for image_path in rendered_images:
            image_pil, image = load_image(image_path)
            
            # run model
            boxes_filt, pred_phrases = get_grounding_output(
                self._model, image, text_prompt, self._args.box_threshold, self._args.text_threshold, device=self._args.device
            )
            print(pred_phrases)
            
            count = 0
            result_cache = set()
            remaining_prompts = set(prompt_list)
            
            # First pass: exact matches
            for phrase in pred_phrases:
                result = re.sub(r'\(.*?\)', '', phrase).strip().lower()
                result = standardize_string(result) 
                if result in result_cache or result not in remaining_prompts:
                    continue
                result_cache.add(result)
                if result in remaining_prompts:
                    count += 1
                    remaining_prompts.remove(result)
            
            # Second pass: partial matches
            for phrase in pred_phrases:
                result = re.sub(r'\(.*?\)', '', phrase).strip().lower()
                result = standardize_string(result)
                if result not in result_cache:
                    point, matched_prompts = get_match_point(result, list(remaining_prompts))
                    count += point    
                    remaining_prompts -= matched_prompts            
            
            rate = min(count / len(prompt_list), 1)

            scores.append(rate)

        score = sum(scores) / len(scores)
        return score

        
        
        
        

