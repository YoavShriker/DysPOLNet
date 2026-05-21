from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageOps

GATE_PASS_THRESHOLD = 0.55
GATE_BLOCK_THRESHOLD = 0.35

# MediaPipe face-mesh (468-point) mouth indices:
# 61, 291: outer mouth corners (left, right); 0, 17: upper / lower mid-lip.
MOUTH_LANDMARK_IDXS = (61, 291, 0, 17)

# A mouth that covers at least this fraction of the frame is treated as a
# close-up; smaller than this is a face/selfie and we crop down to the mouth.
FACE_CROP_MOUTH_AREA_THRESHOLD = 0.15
FACE_CROP_PAD_FACTOR = 1.4

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

POSITIVE_PROMPTS = [
    "a close-up clinical photograph of the inside of a human mouth showing oral mucosa, tongue, or lips",
    "an intraoral photograph of a lesion on the buccal mucosa",
    "a close-up photograph of the inside of a human mouth with teeth and gums",
]

NEGATIVE_PROMPTS = [
    "a photograph of human skin on a hand or arm",
    "a photograph of a face from a distance, not a close-up of the mouth",
    "a document, screenshot, or text image",
    "a photograph of an animal",
    "a landscape or outdoor scene",
    "a photograph of food on a plate",
    "an abstract pattern or texture",
]


@dataclass
class GateResult:
    decision: str
    p_oral: float
    method: str
    face_detected: bool
    crop_box: Optional[tuple]


def detect_face_and_mouth(pil_image, face_mesh):
    img_np = np.asarray(ImageOps.exif_transpose(pil_image).convert("RGB"))
    results = face_mesh.process(img_np)
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0].landmark
    h, w = img_np.shape[:2]
    xs = [landmarks[i].x * w for i in MOUTH_LANDMARK_IDXS]
    ys = [landmarks[i].y * h for i in MOUTH_LANDMARK_IDXS]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    mouth_w = max(x_max - x_min, 1.0)
    mouth_h = max(y_max - y_min, 1.0)
    return {
        "img_size": (w, h),
        "mouth_bbox": (x_min, y_min, x_max, y_max),
        "mouth_area_frac": (mouth_w * mouth_h) / (w * h),
    }


def crop_box_around_mouth(img_size, mouth_bbox, pad_factor=FACE_CROP_PAD_FACTOR):
    w, h = img_size
    x_min, y_min, x_max, y_max = mouth_bbox
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    half_w = (x_max - x_min) * pad_factor / 2
    half_h = (y_max - y_min) * pad_factor / 2
    left = max(0, int(round(cx - half_w)))
    top = max(0, int(round(cy - half_h)))
    right = min(w, int(round(cx + half_w)))
    bottom = min(h, int(round(cy + half_h)))
    return (left, top, right, bottom)


def clip_oral_probability(pil_image, clip_bundle):
    import torch
    img = ImageOps.exif_transpose(pil_image).convert("RGB")
    image_input = clip_bundle["preprocess"](img).unsqueeze(0)
    with torch.no_grad():
        image_features = clip_bundle["model"].encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = (image_features @ clip_bundle["text_features"].T) * 100.0
        probs = torch.softmax(logits, dim=-1).squeeze(0)
    n_pos = clip_bundle["n_positive"]
    return float(probs[:n_pos].sum().item())


def decide_from_p_oral(p_oral):
    if p_oral >= GATE_PASS_THRESHOLD:
        return "pass"
    if p_oral >= GATE_BLOCK_THRESHOLD:
        return "warn"
    return "block"


def evaluate_gate(pil_image, face_detector_fn, clip_score_fn):
    face_info = face_detector_fn(pil_image)
    if face_info is not None:
        if face_info["mouth_area_frac"] >= FACE_CROP_MOUTH_AREA_THRESHOLD:
            return GateResult(
                decision="pass",
                p_oral=0.9,
                method="face_close_up",
                face_detected=True,
                crop_box=None,
            )
        crop_box = crop_box_around_mouth(face_info["img_size"], face_info["mouth_bbox"])
        return GateResult(
            decision="warn",
            p_oral=0.7,
            method="face_crop",
            face_detected=True,
            crop_box=crop_box,
        )

    p_oral = clip_score_fn(pil_image)
    return GateResult(
        decision=decide_from_p_oral(p_oral),
        p_oral=p_oral,
        method="clip",
        face_detected=False,
        crop_box=None,
    )


def build_face_mesh():
    import mediapipe as mp
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
    )


def build_clip_bundle():
    import open_clip
    import torch
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    text_tokens = tokenizer(POSITIVE_PROMPTS + NEGATIVE_PROMPTS)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return {
        "model": model,
        "preprocess": preprocess,
        "text_features": text_features,
        "n_positive": len(POSITIVE_PROMPTS),
    }
