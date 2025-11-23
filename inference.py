import cv2
import numpy as np
import tensorflow as tf

# --------------------------------------------------
# Load model + class map
# --------------------------------------------------

MODEL_PATH = "asl_model.keras"          # your new Keras model format
CLASS_MAP_PATH = "asl_class_map.npy"

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Loading class map...")
class_map = np.load(CLASS_MAP_PATH, allow_pickle=True).item()

IMG_SIZE = 224

# --------------------------------------------------
# Preprocess a frame for MobileNetV2
# --------------------------------------------------
def preprocess(frame):
    # Resize to 224×224
    frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    # Convert BGR (OpenCV) to RGB (TensorFlow)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    # Normalize to [0,1]
    frame_norm = frame_rgb.astype(np.float32) / 255.0
    # Add batch dimension
    return np.expand_dims(frame_norm, axis=0)

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
    x = preprocess(frame)

    # Predict ASL letter
    preds = model.predict(x, verbose=0)
    class_id = np.argmax(preds)
    class_label = class_map[class_id]

    # Draw prediction on video
    cv2.putText(
        frame,
        f"Prediction: {class_label}",
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
