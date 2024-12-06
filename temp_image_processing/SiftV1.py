import cv2
import numpy as np

def sample_frames(video_file, num_frames):
    """
    Sample evenly spaced frames from a video file.

    Args:
        video_file (str): Path to the video file.
        num_frames (int): Number of frames to sample.

    Returns:
        list: List of sampled grayscale frames.
    """
    cap = cv2.VideoCapture(video_file)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Total frames in video: {total_frames}")
    
    # Adjust num_frames if it's greater than total_frames
    if num_frames > total_frames:
        num_frames = total_frames
        print(f"Adjusted number of frames to sample: {num_frames}")

    # Calculate intervals for sampling
    interval = total_frames / num_frames
    print(f"Frame sampling interval: {interval}")
    
    # Extract frames
    for i in range(num_frames):
        frame_idx = int(i * interval)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Ensure the frame is in grayscale
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
            print(f"Sampled frame {i} at index {frame_idx}")
        else:
            print(f"Failed to read frame at index {frame_idx}")
            break

    cap.release()
    print(f"Total sampled frames: {len(frames)}")
    return frames

def detect_sift_features(frame):
    """
    Detect SIFT features in a grayscale frame.

    Args:
        frame (numpy.ndarray): Grayscale image.

    Returns:
        tuple: Keypoints and descriptors.
    """
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(frame, None)
    print(f"Detected {len(keypoints)} keypoints")
    return keypoints, descriptors

def find_repeated_features(keypoints_list, descriptors_list, spatial_threshold=2):
    """
    Find repeated features across multiple frames.

    Args:
        keypoints_list (list): List of keypoints for each frame.
        descriptors_list (list): List of descriptors for each frame.
        spatial_threshold (float): Distance threshold to consider features as repeated.

    Returns:
        list: List of repeated feature points for each frame.
    """
    bf = cv2.BFMatcher()
    num_frames = len(keypoints_list)
    repeated_features = [[] for _ in range(num_frames)]

    print("Finding repeated features across frames...")
    for i in range(num_frames - 1):
        kp1 = keypoints_list[i]
        des1 = descriptors_list[i]
        for j in range(i + 1, num_frames):
            kp2 = keypoints_list[j]
            des2 = descriptors_list[j]

            # Match descriptors between frames using KNN
            matches = bf.knnMatch(des1, des2, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    # Calculate spatial distance between keypoints
                    pt1 = kp1[m.queryIdx].pt
                    pt2 = kp2[m.trainIdx].pt
                    distance = cv2.norm(pt1, pt2, cv2.NORM_L2)
                    if distance < spatial_threshold:
                        repeated_features[i].append((int(pt1[0]), int(pt1[1])))
                        repeated_features[j].append((int(pt2[0]), int(pt2[1])))
                        good_matches.append(m)
            print(f"Frame {i} vs Frame {j}: {len(good_matches)} good matches found")

    # Remove duplicate points in each frame
    for idx in range(num_frames):
        repeated_features[idx] = list(set(repeated_features[idx]))
        print(f"Frame {idx} has {len(repeated_features[idx])} repeated features")

    return repeated_features

def preprocess_frame_for_contours(frame):
    """
    Preprocess the frame to enhance contour detection.

    Args:
        frame (numpy.ndarray): Grayscale image.

    Returns:
        numpy.ndarray: Preprocessed binary image.
    """
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    # Use Otsu's thresholding to automatically determine the threshold value
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert the image if necessary (objects should be white on black background)
    # Check the mean to decide if inversion is needed
    # if np.mean(thresh) > 127:
    #     thresh = cv2.bitwise_not(thresh)
    #     print("Inverted thresholded image for proper contour detection.")

    # Apply morphological operations to separate connected objects
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Optional: Apply dilation to ensure contours are closed
    dilation = cv2.dilate(opening, kernel, iterations=1)

    return dilation

def find_relevant_contours(preprocessed_frame):
    """
    Find contours in the preprocessed frame.

    Args:
        preprocessed_frame (numpy.ndarray): Binary image for contour detection.

    Returns:
        list: Detected contours.
    """
    contours, _ = cv2.findContours(preprocessed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Detected {len(contours)} contours after preprocessing.")
    return contours

def remove_objects_associated_with_features(frames, repeated_features):
    """
    Remove objects associated with repeated features from each frame.

    Args:
        frames (list): List of grayscale frames.
        repeated_features (list): List of repeated feature points for each frame.

    Returns:
        list: List of processed frames with objects removed.
    """
    processed_frames = []

    print("Removing objects associated with repeated features...")
    for idx, frame in enumerate(frames):
        print(f"\nProcessing Frame {idx}")

        # Display the original frame with repeated features
        frame_with_features = cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2BGR)
        for pt in repeated_features[idx]:
            cv2.circle(frame_with_features, pt, 3, (0, 0, 255), -1)
        cv2.imshow(f'Repeated Features - Frame {idx}', frame_with_features)
        cv2.waitKey(1)  # Briefly display the image

        # Preprocess the frame to enhance contour detection
        preprocessed = preprocess_frame_for_contours(frame)

        # Display the preprocessed binary image
        cv2.imshow(f'Preprocessed - Frame {idx}', preprocessed)
        cv2.waitKey(1)

        # Find contours in the preprocessed frame
        contours = find_relevant_contours(preprocessed)

        # Display contours
        frame_with_contours = cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(frame_with_contours, contours, -1, (0, 255, 0), 1)
        cv2.imshow(f'Contours - Frame {idx}', frame_with_contours)
        cv2.waitKey(1)

        # Create a mask for objects to remove
        mask = np.zeros_like(frame, dtype=np.uint8)

        for pt in repeated_features[idx]:
            for contour in contours:
                if cv2.pointPolygonTest(contour, pt, False) >= 0:
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    print(f"Marked contour for point {pt} in Frame {idx}")
                    break  # Assuming one contour per point

        # Display mask
        cv2.imshow(f'Mask - Frame {idx}', mask)
        cv2.waitKey(1)

        # Remove the objects by setting them to 255 (white)
        frame_processed = frame.copy()
        frame_processed[mask == 255] = 255
        processed_frames.append(frame_processed)
        print(f"Frame {idx} processed")

        # Display processed frame with objects removed
        cv2.imshow(f'Processed Frame {idx}', frame_processed)
        cv2.waitKey(1)

    print("\nAll frames processed")
    return processed_frames

def main():
    video_path = 'temp_image_processing/Worm3.avi'
    num_frames = 5  # Adjust as needed for debugging
    print("Sampling frames...")
    frames = sample_frames(video_path, num_frames)
    
    keypoints_list = []
    descriptors_list = []
    
    print("\nDetecting SIFT features in all sampled frames...")
    for idx, frame in enumerate(frames):
        print(f"\nDetecting features in Frame {idx}")
        keypoints, descriptors = detect_sift_features(frame)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)
    
    print("\nFinding repeated features across all frames...")
    repeated_features = find_repeated_features(keypoints_list, descriptors_list)
    
    # Debug: Print repeated features
    for idx, features in enumerate(repeated_features):
        print(f"Repeated features in Frame {idx}: {len(features)} points")
    
    print("\nRemoving objects associated with repeated features...")
    processed_frames = remove_objects_associated_with_features(frames, repeated_features)
    
    print("\nDisplaying the processed frames. Press any key to proceed to the next frame or 'q' to quit.")
    for idx, frame in enumerate(processed_frames):
        cv2.imshow(f'Final Processed Frame {idx}', frame)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()