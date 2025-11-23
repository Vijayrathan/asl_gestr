import base64
import json
import os
import sys
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification

MODEL_NAME = os.environ.get("ASL_MODEL_NAME", "prithivMLmods/Alphabet-Sign-Language-Detection")


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _load_artifacts():
    _log(f"Loading HuggingFace ASL model: {MODEL_NAME}")
    model = SiglipForImageClassification.from_pretrained(MODEL_NAME)
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    return model, processor


MODEL, PROCESSOR = _load_artifacts()

# Class mapping from model indices to letters
LABELS = {
    "0": "A", "1": "B", "2": "C", "3": "D", "4": "E", "5": "F", "6": "G", "7": "H", "8": "I", "9": "J",
    "10": "K", "11": "L", "12": "M", "13": "N", "14": "O", "15": "P", "16": "Q", "17": "R", "18": "S", "19": "T",
    "20": "U", "21": "V", "22": "W", "23": "X", "24": "Y", "25": "Z"
}


def _decode_image(image_b64: str) -> Image.Image:
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    image_bytes = base64.b64decode(image_b64)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("Unable to decode image data")

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Convert to PIL Image
    image = Image.fromarray(img_rgb).convert("RGB")
    return image


def predict_letter(image_b64: str) -> Tuple[str, float]:
    image = _decode_image(image_b64)
    
    # Preprocess image using HuggingFace processor
    inputs = PROCESSOR(images=image, return_tensors="pt")
    
    # Run inference
    with torch.no_grad():
        outputs = MODEL(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze()
    
    # Get predicted class
    class_id = int(torch.argmax(probs).item())
    confidence = float(probs[class_id].item())
    
    # Map class ID to letter
    letter = LABELS.get(str(class_id), "?")
    letter = letter.strip().upper()
    
    return letter, confidence


def main() -> None:
    _log("Classifier worker ready")

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            _log(f"Invalid JSON received: {exc}")
            continue

        request_id = payload.get("id")
        image_b64 = payload.get("image")

        response = {"id": request_id}

        if not image_b64:
            response["error"] = "Missing image data"
        else:
            try:
                letter, confidence = predict_letter(image_b64)
                response["letter"] = letter
                response["confidence"] = confidence
            except Exception as exc:  # noqa: BLE001
                response["error"] = str(exc)

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _log("Classifier worker shutting down")

