import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_card(image):
    # Step 1: Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Edge Detection
    edges = cv2.Canny(blurred, 50, 150)

    # Step 4: Find Contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found.")
        return None
    
    # Step 5: Find the largest contour, which should be the card
    largest_contour = max(contours, key=cv2.contourArea)

    # Step 6: Approximate the contour to get a quadrilateral (the card shape)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) != 4:
        print("Could not find a four-cornered contour for the card.")
        return None
    
    # Step 7: Order points and apply perspective transformation to get a top-down view
    # Get the points in a consistent order: top-left, top-right, bottom-right, bottom-left
    pts = np.array([point[0] for point in approx], dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect = np.array([
        pts[np.argmin(s)],    # Top-left point
        pts[np.argmin(diff)], # Top-right point
        pts[np.argmax(s)],    # Bottom-right point
        pts[np.argmax(diff)]  # Bottom-left point
    ], dtype="float32")

    # Calculate the width and height of the new image
    width_a = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
    width_b = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
    height_b = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # Define the destination points for the perspective transform
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    # Compute the perspective transform matrix and apply it
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

    # Display the processed card
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    plt.title("Aligned and Cropped Card")
    plt.axis('off')
    plt.show()

    return warped

def main():
    # Load image (adjust path as needed)
    filename = r'e:\Laptop\Work\Study\Uni - TTU\6) Fall 24 - Sixth Semester\Fall 2024 TTU Image Processing (ECE-4367-001) Full Term\Projects\Project 4\I1.tif'
    image = cv2.imread(filename)
    if image is None:
        print("Image not found.")
        return
    
    # Process the card
    processed_card = process_card(image)
    if processed_card is None:
        print("Failed to process the card.")
        return

    # Further template matching steps would go here

if __name__ == "__main__":
    main()
