import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter

# ---------------------------
# 1. Load TFLite model
# ---------------------------
interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input & output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------------------
# 2. Load labels
# ---------------------------
def load_labels(label_path):
    with open(label_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

labels = load_labels("labels.txt")  # path to your labels.txt

# ---------------------------
# 3. Preprocess image
# ---------------------------
def preprocess_image(img_path, input_shape):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((input_shape[1], input_shape[2]))  # (width, height)
    img_array = np.array(img, dtype=np.float32)

    # normalize if model expects float input
    if input_details[0]['dtype'] == np.float32:
        img_array = img_array / 255.0  

    # Add batch dimension: (1, height, width, channels)
    return np.expand_dims(img_array, axis=0)

# ---------------------------
# 4. Run inference
# ---------------------------
input_shape = input_details[0]['shape']  # e.g., [1, 224, 224, 3]
image = preprocess_image("test.jpg", input_shape)

interpreter.set_tensor(input_details[0]['index'], image)
interpreter.invoke()

# ---------------------------
# 5. Get prediction
# ---------------------------
output_data = interpreter.get_tensor(output_details[0]['index'])
pred_class = int(np.argmax(output_data[0]))  # predicted class index
confidence = float(np.max(output_data[0]))   # confidence score

pred_label = labels[pred_class] if pred_class < len(labels) else "Unknown"

print("Predicted Class:", pred_label)
print("Confidence:", confidence)
