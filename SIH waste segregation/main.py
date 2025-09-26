import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter  # lightweight for Raspberry Pi

# ---------------------------
# 1. Load TFLite model
# ---------------------------
interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# ---------------------------
# 2. Load labels
# ---------------------------
def load_labels(label_path):
    with open(label_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

labels = load_labels("labels.txt")

# ---------------------------
# 3. Preprocess frame (quantized)
# ---------------------------
def preprocess_frame(frame, input_shape):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_shape[1], input_shape[2]))
    img = np.array(img, dtype=np.uint8)  # quantized model expects uint8
    return np.expand_dims(img, axis=0)

# ---------------------------
# 4. Capture video from Pi camera / webcam
# ---------------------------
cap = cv2.VideoCapture(0)  # 0 = default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_data = preprocess_frame(frame, input_shape)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred_class = int(np.argmax(output_data[0]))
    confidence = float(np.max(output_data[0]))
    pred_label = labels[pred_class] if pred_class < len(labels) else "Unknown"

    # Overlay result on frame
    text = f"{pred_label} ({confidence:.2f})"
    cv2.putText(frame, text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Waste Classifier - Pi Camera", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
