import cv2
from inference_sdk import InferenceHTTPClient
import numpy as np
import tempfile
from PIL import Image
import os

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="ErfkmyS3AMjcC5KPQs9F"
)

cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        pil_image.save(temp_file, format="JPEG")
        temp_file_path = temp_file.name
    
    result = CLIENT.infer(temp_file_path, model_id="playing-cards-ow27d/4")
    os.remove(temp_file_path)

    if result and 'predictions' in result:
        for prediction in result['predictions']:
            x_center = prediction['x'] * frame.shape[1]
            y_center = prediction['y'] * frame.shape[0]
            width = prediction['width'] * frame.shape[1]
            height = prediction['height'] * frame.shape[0]
            confidence = prediction['confidence']
            label = prediction['class']

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label_text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
