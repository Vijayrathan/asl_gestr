import base64
import json
import os
import sys
from typing import Tuple

import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = os.environ.get("ASL_MODEL_PATH", "asl_model.keras")
CLASS_MAP_PATH = os.environ.get("ASL_CLASS_MAP_PATH", "asl_class_map.npy")
IMG_SIZE = int(os.environ.get("ASL_IMG_SIZE", "224"))


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _load_artifacts() -> Tuple[tf.keras.Model, dict]:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    if not os.path.exists(CLASS_MAP_PATH):
        raise FileNotFoundError(f"Class map not found at {CLASS_MAP_PATH}")

    _log(f"Loading ASL model from {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    _log(f"Loading class map from {CLASS_MAP_PATH}")
    class_map = np.load(CLASS_MAP_PATH, allow_pickle=True).item()
    return model, class_map


MODEL, CLASS_MAP = _load_artifacts()


def _decode_image(image_b64: str) -> np.ndarray:
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    image_bytes = base64.b64decode(image_b64)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Unable to decode image data")

    return img


def _preprocess(image_bgr: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)


def predict_letter(image_b64: str) -> Tuple[str, float]:
    frame = _decode_image(image_b64)
    input_tensor = _preprocess(frame)
    preds = MODEL.predict(input_tensor, verbose=0)
    class_id = int(np.argmax(preds))
    confidence = float(preds[0][class_id])
    letter = CLASS_MAP.get(class_id, "?")

    if not isinstance(letter, str):
        letter = str(letter)

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

