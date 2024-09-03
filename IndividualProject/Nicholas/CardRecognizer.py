import cv2
from inference_sdk import InferenceHTTPClient
# Initialize the inference client with the correct API URL and key
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="ErfkmyS3AMjcC5KPQs9F"  # Use your actual API key here
)
# Define the model ID as specified by the documentation
MODEL_ID = "playing-cards-ow27d/4"
# Open the webcam
cap = cv2.VideoCapture(0)
while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
    # Resize the frame to the expected input size of the model
    resized_frame = cv2.resize(frame, (640, 640))
    # Perform inference using the client
    try:
        # Perform inference on the resized frame
        result = CLIENT.infer(resized_frame, model_id=MODEL_ID)
        # Extract predictions from the result
        predictions = result.get('predictions', [])
        # Draw bounding boxes and labels on the frame
        for prediction in predictions:
            x = int(prediction['x'] - prediction['width'] / 2)
            y = int(prediction['y'] - prediction['height'] / 2)
            w = int(prediction['width'])
            h = int(prediction['height'])
            class_name = prediction['class']
            confidence = prediction['confidence']
            # Draw the bounding box
            cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Display class and confidence
            label = f"{class_name} ({confidence:.2f})"
            cv2.putText(resized_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # Show the annotated frame
        cv2.imshow("Playing Card Detection", resized_frame)
    except Exception as e:
        print(f"Inference error: {e}")
    # Exit on pressing the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


