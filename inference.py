import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification

# --------------------------------------------------
# Load model + processor
# --------------------------------------------------

MODEL_NAME = "prithivMLmods/Alphabet-Sign-Language-Detection"

print("Loading HuggingFace model...")
model = SiglipForImageClassification.from_pretrained(MODEL_NAME)
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

# Class mapping from model indices to letters
LABELS = {
    "0": "A", "1": "B", "2": "C", "3": "D", "4": "E", "5": "F", "6": "G", "7": "H", "8": "I", "9": "J",
    "10": "K", "11": "L", "12": "M", "13": "N", "14": "O", "15": "P", "16": "Q", "17": "R", "18": "S", "19": "T",
    "20": "U", "21": "V", "22": "W", "23": "X", "24": "Y", "25": "Z"
}

# --------------------------------------------------
# Preprocess a frame for HuggingFace model
# --------------------------------------------------
def preprocess(frame):
    # Convert BGR (OpenCV) to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert to PIL Image
    image = Image.fromarray(frame_rgb).convert("RGB")
    # Process using HuggingFace processor
    inputs = processor(images=image, return_tensors="pt")
    return inputs

# --------------------------------------------------
# Webcam Live Loop
# --------------------------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

print("Starting webcam... press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Prepare frame
    inputs = preprocess(frame)

    # Predict ASL letter
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze()
    
    class_id = int(torch.argmax(probs).item())
    confidence = float(probs[class_id].item())
    class_label = LABELS.get(str(class_id), "?")

    # Draw prediction on video
    cv2.putText(
        frame,
        f"Prediction: {class_label} ({confidence:.2f})",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    # Show window
    cv2.imshow("ASL Live Detection", frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
