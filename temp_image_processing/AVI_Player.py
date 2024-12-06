import cv2
import sys

def play_video(file_path):
    # Create a VideoCapture object
    while True:
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        cap = cv2.VideoCapture(file_path)

        # Check if the video opened successfully
        if not cap.isOpened():
            print("Error: Could not open video.")
            sys.exit()

        # Read until the video is completed
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                # Display the resulting frame
                cv2.imshow('Frame', frame)

                # Press Q on keyboard to exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

play_video(file_path='temp_image_processing/Worm1.avi')