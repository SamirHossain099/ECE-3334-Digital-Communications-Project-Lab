import cv2
from inference_sdk import InferenceHTTPClient
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="ErfkmyS3AMjcC5KPQs9F" 
)
MODEL_ID = "playing-cards-ow27d/4"
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
    resized_frame = cv2.resize(frame, (640, 640))
    result = CLIENT.infer(resized_frame, model_id=MODEL_ID)
    predictions = result.get('predictions', [])
    for prediction in predictions:
        x = int(prediction['x'] - prediction['width'] / 2)
        y = int(prediction['y'] - prediction['height'] / 2)
        w = int(prediction['width'])
        h = int(prediction['height'])
        class_name = prediction['class']
        confidence = prediction['confidence']
        cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{class_name} ({confidence:.2f})"
        cv2.putText(resized_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow("Playing Card Detection", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()