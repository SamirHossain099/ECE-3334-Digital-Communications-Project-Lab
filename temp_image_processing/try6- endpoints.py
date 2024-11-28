import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from scipy.interpolate import splprep, splev
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.decomposition import PCA
from skimage.morphology import medial_axis
from skimage.util import invert

# -------------------- Helper Functions -------------------- #

def process_video(video_path):
    """
    Process the first frame of the video to segment and straighten the C. elegans worm.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open video file.")
        return

    # Read the first frame
    ret, frame = cap.read()
    cap.release()
    if not ret:
        messagebox.showerror("Error", "Cannot read video file.")
        return

    # Resize frame for consistency
    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("Original Frame", frame)
    cv2.waitKey(0)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale Image", gray)
    cv2.waitKey(0)

    # Enhance contrast using histogram equalization
    gray_equalized = cv2.equalizeHist(gray)
    cv2.imshow("Contrast Enhanced Image", gray_equalized)
    cv2.waitKey(0)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray_equalized, (5, 5), 0)
    cv2.imshow("Blurred Image", blurred)
    cv2.waitKey(0)

    # Apply adaptive thresholding to create binary image
    block_size = 15  # Must be odd number >=3
    C = 3           # Constant subtracted from the mean
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, C)
    cv2.imshow("Binary Image", binary)
    cv2.waitKey(0)

    # Morphological operations to remove small noise and fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    cv2.imshow("Morphologically Cleaned Image", opened)
    cv2.waitKey(0)

    # Find contours using OpenCV
    contours, hierarchy = cv2.findContours(
        opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Total Contours Found: {len(contours)}")

    if not contours:
        messagebox.showerror("Error", "No contours found.")
        return

    # Filter contours based on length and elongation
    worm_contour = None
    max_length = 0
    for idx, cnt in enumerate(contours):
        length = cv2.arcLength(cnt, closed=False)
        x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(cnt)
        aspect_ratio = float(h_cnt) / w_cnt if w_cnt != 0 else 0

        print(f"Contour {idx+1}: Length={length:.2f}, Aspect Ratio={aspect_ratio:.2f}")

        if length > max_length and aspect_ratio > 0.5:
            max_length = length
            worm_contour = cnt

    if worm_contour is None:
        messagebox.showerror("Error", "Worm contour could not be detected.")
        return

    print(f"Selected Worm Contour Length: {max_length:.2f}")

    # Create mask from the worm contour
    worm_mask = np.zeros_like(gray)
    cv2.drawContours(worm_mask, [worm_contour], -1, 255, -1)

    # Apply the mask to extract the worm
    segmented_worm = cv2.bitwise_and(frame, frame, mask=worm_mask)

    # Get the bounding rectangle of the worm contour
    x, y, w, h = cv2.boundingRect(worm_contour)
    x, y = max(x, 0), max(y, 0)
    x_end, y_end = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])

    # Crop the segmented worm image to the bounding rectangle
    cropped_worm = segmented_worm[y:y_end, x:x_end]

    # ----------- New Code: Further Opening and Closing ----------- #
    # Convert cropped worm to grayscale if needed
    cropped_gray = cv2.cvtColor(cropped_worm, cv2.COLOR_BGR2GRAY)
    
    # Apply further morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    further_opened = cv2.morphologyEx(cropped_gray, cv2.MORPH_OPEN, kernel, iterations=1)
    further_closed = cv2.morphologyEx(further_opened, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Re-apply the mask to the cropped worm
    final_mask = np.where(further_closed > 0, 255, 0).astype(np.uint8)
    refined_worm = cv2.bitwise_and(cropped_worm, cropped_worm, mask=final_mask)
    cv2.imshow("Refined Segmented Worm", refined_worm)
    cv2.waitKey(0)

    # ----------- Straighten the Worm ----------- #
    # Assuming 'refined_worm' is your segmented worm image
    straightened_worm = straighten_worm(refined_worm)  

    cv2.imshow("Straightened Worm", straightened_worm)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Display the final result in the GUI
    display_result(frame, straighten_worm)


def trace_boundary(binary_image):
    """
    Trace the boundary of a binary object using Moore's Neighbor Tracing algorithm.

    Parameters:
        binary_image (numpy.ndarray): Binary image containing the object (worm).

    Returns:
        boundary_coords (list of tuples): List of (row, col) coordinates of the boundary pixels.
    """
    # Ensure the binary image is binary (0 and 1)
    binary_image = (binary_image > 0).astype(np.uint8)

    # Define the neighborhood (Moore neighborhood)
    neighborhood = [(-1, -1), (-1, 0), (-1, 1),
                    ( 0, 1),  (1, 1),  (1, 0),
                    ( 1, -1), (0, -1)]

    # Pad the image to handle borders
    padded_image = np.pad(binary_image, pad_width=1, mode='constant', constant_values=0)

    # Find the starting point (first foreground pixel)
    rows, cols = np.where(padded_image == 1)
    if len(rows) == 0:
        print("No foreground pixels found.")
        return []

    start_point = (rows[0], cols[0])

    boundary_coords = [start_point]
    current_point = start_point
    backtrack_direction = 7  # Start with the last neighbor

    while True:
        # Look for the next boundary point
        found_next = False
        for i in range(len(neighborhood)):
            # Calculate the neighbor index (clockwise)
            neighbor_idx = (backtrack_direction + 1 + i) % 8
            dr, dc = neighborhood[neighbor_idx]
            neighbor_point = (current_point[0] + dr, current_point[1] + dc)

            if padded_image[neighbor_point] == 1:
                # Found the next boundary point
                boundary_coords.append(neighbor_point)
                current_point = neighbor_point
                backtrack_direction = (neighbor_idx + 5) % 8  # Update backtrack direction
                found_next = True
                break

        if not found_next:
            # No next boundary point found; boundary is complete
            break

        if current_point == start_point and len(boundary_coords) > 1:
            # Returned to the starting point; boundary is complete
            break

    # Adjust for padding
    boundary_coords = [(r - 1, c - 1) for r, c in boundary_coords]

    return boundary_coords

def find_worm_ends(boundary_coords):
    """
    Identify the two ends of the worm based on boundary coordinates.

    Parameters:
        boundary_coords (list of tuples): List of (row, col) coordinates of the boundary pixels.

    Returns:
        end1 (tuple): Coordinates of the first end.
        end2 (tuple): Coordinates of the second end.
    """
    if not boundary_coords:
        print("No boundary coordinates provided.")
        return None, None

    # Convert to NumPy array for easier processing
    boundary_array = np.array(boundary_coords)  # Shape: (N, 2) -> (row, col)

    # Perform PCA to find the main axis of the worm
    pca = PCA(n_components=2)
    pca.fit(boundary_array)

    # Project boundary points onto the principal axis
    projections = pca.transform(boundary_array)[:, 0]  # Only the first principal component

    # Identify the two points with the maximum and minimum projections
    end1_idx = np.argmax(projections)
    end2_idx = np.argmin(projections)

    end1 = tuple(boundary_array[end1_idx])
    end2 = tuple(boundary_array[end2_idx])

    return end1, end2

def visualize_boundary_and_ends(refined_worm, boundary_coords, end1, end2):
    """
    Visualize the boundary with the identified ends.

    Parameters:
        refined_worm (numpy.ndarray): Original worm image (color).
        boundary_coords (list of tuples): Boundary coordinates.
        end1 (tuple): First end coordinate.
        end2 (tuple): Second end coordinate.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(refined_worm, cv2.COLOR_BGR2RGB))
    boundary_array = np.array(boundary_coords)
    plt.plot(boundary_array[:, 1], boundary_array[:, 0], 'r-', linewidth=1)  # Boundary in red

    if end1:
        plt.plot(end1[1], end1[0], 'go', markersize=8, label='End 1')  # Green dot
    if end2:
        plt.plot(end2[1], end2[0], 'bo', markersize=8, label='End 2')  # Blue dot

    plt.title("Boundary with Identified Ends")
    plt.legend()
    plt.axis('off')
    plt.show()

def straighten_worm(refined_worm):
    """
    Straighten the worm by tracing its boundary, identifying its ends, and mapping the centerline.

    Parameters:
        refined_worm (numpy.ndarray): Input image of the segmented and refined worm (color).

    Returns:
        straightened_worm (numpy.ndarray): Output image of the straightened worm.
    """
    # -------------------- Step 1: Grayscale and Binary Conversion -------------------- #
    gray = cv2.cvtColor(refined_worm, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Display Binary Worm Mask
    plt.figure(figsize=(6, 6))
    plt.imshow(binary, cmap='gray')
    plt.title("Step 1: Binary Worm Mask")
    plt.axis('off')
    plt.show()

    # -------------------- Step 2: Skeletonization -------------------- #
    try:
        skeleton = cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    except AttributeError:
        raise AttributeError("cv2.ximgproc.thinning is not available. Install opencv-contrib-python.")

    # Display Skeleton
    plt.figure(figsize=(6, 6))
    plt.imshow(skeleton, cmap='gray')
    plt.title("Step 2: Skeleton of Worm")
    plt.axis('off')
    plt.show()

    # -------------------- Step 3: Boundary Tracing -------------------- #
    boundary_coords = trace_boundary(binary)

    # Display Boundary Traced
    plt.figure(figsize=(6, 6))
    plt.imshow(binary, cmap='gray')
    boundary_array = np.array(boundary_coords)
    plt.plot(boundary_array[:, 1], boundary_array[:, 0], 'r-', linewidth=1)
    plt.title("Step 3: Boundary Traced")
    plt.axis('off')
    plt.show()

    # -------------------- Step 4: Identify Worm Ends -------------------- #
    end1, end2 = find_worm_ends(boundary_coords)

    # Display Boundary with Ends
    visualize_boundary_and_ends(refined_worm, boundary_coords, end1, end2)

    # -------------------- Step 5: Straighten the Worm -------------------- #
    # For demonstration, we'll align end1 to the left and end2 to the right along the x-axis
    # Compute the angle to rotate the worm so that the backbone aligns horizontally

    if end1 is None or end2 is None:
        print("Cannot straighten the worm without identifying both ends.")
        return refined_worm

    # Calculate the angle between the line connecting the ends and the x-axis
    dy = end2[0] - end1[0]
    dx = end2[1] - end1[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Rotate the image to align the backbone horizontally
    (h, w) = refined_worm.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(refined_worm, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    # Update the boundary coordinates based on rotation
    boundary_coords_rotated = []
    for coord in boundary_coords:
        r, c = coord
        vec = np.array([c, r, 1])
        rotated_vec = M @ vec
        rotated_c, rotated_r = rotated_vec[:2]
        boundary_coords_rotated.append((rotated_r, rotated_c))

    # Re-identify ends after rotation
    end1_rotated, end2_rotated = find_worm_ends(boundary_coords_rotated)

    # Visualization after Rotation
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
    boundary_array_rotated = np.array(boundary_coords_rotated)
    plt.plot(boundary_array_rotated[:, 1], boundary_array_rotated[:, 0], 'r-', linewidth=1)

    if end1_rotated:
        plt.plot(end1_rotated[1], end1_rotated[0], 'go', markersize=8, label='End 1')
    if end2_rotated:
        plt.plot(end2_rotated[1], end2_rotated[0], 'bo', markersize=8, label='End 2')

    plt.title("Step 5: Rotated Worm with Identified Ends")
    plt.legend()
    plt.axis('off')
    plt.show()

    # -------------------- Step 6: Crop and Translate -------------------- #
    # Optionally, crop the rotated image to include the entire worm
    # Find bounding box based on boundary coordinates
    boundary_array_rotated = np.array(boundary_coords_rotated)
    min_r, min_c = boundary_array_rotated.min(axis=0)
    max_r, max_c = boundary_array_rotated.max(axis=0)

    # Add some padding
    padding = 10
    min_r = max(int(min_r) - padding, 0)
    min_c = max(int(min_c) - padding, 0)
    max_r = min(int(max_r) + padding, rotated.shape[0])
    max_c = min(int(max_c) + padding, rotated.shape[1])

    cropped = rotated[min_r:max_r, min_c:max_c]

    # Display Cropped and Rotated Image
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    plt.title("Step 6: Cropped and Rotated Worm")
    plt.axis('off')
    plt.show()

    # -------------------- Step 7: Final Straightened Worm -------------------- #
    # The rotated and cropped image should have the worm's backbone aligned horizontally
    # For better visualization, we can normalize the image height

    # Optionally, resize the image to standard dimensions
    standard_height = 200
    aspect_ratio = cropped.shape[1] / cropped.shape[0]
    standard_width = int(standard_height * aspect_ratio)
    straightened_worm = cv2.resize(cropped, (standard_width, standard_height), interpolation=cv2.INTER_LINEAR)

    # Display the Final Straightened Worm
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(refined_worm, cv2.COLOR_BGR2RGB))
    plt.title('Original Worm')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(straightened_worm, cv2.COLOR_BGR2RGB))
    plt.title('Straightened Worm')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return straightened_worm

def display_result(original_frame, straightened_worm):
    """
    Display the original and straightened worm images in the GUI.
    """
    # Convert images to RGB for displaying in Tkinter
    original_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
    straightened_rgb = cv2.cvtColor(straightened_worm, cv2.COLOR_BGR2RGB)

    # Convert images to PIL format
    original_pil = Image.fromarray(original_rgb)
    straightened_pil = Image.fromarray(straightened_rgb)

    # Resize images for display
    original_pil = original_pil.resize((400, 300), Image.Resampling.LANCZOS)

    # For the straightened worm, maintain aspect ratio
    max_width, max_height = 400, 300
    width, height = straightened_pil.size
    scaling_factor = min(max_width / width, max_height / height)
    new_width, new_height = int(width * scaling_factor), int(height * scaling_factor)
    straightened_pil = straightened_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Convert images to ImageTk format
    original_tk = ImageTk.PhotoImage(original_pil)
    straightened_tk = ImageTk.PhotoImage(straightened_pil)

    # Display images in the GUI
    original_label.config(image=original_tk)
    original_label.image = original_tk
    segmented_label.config(image=straightened_tk)
    segmented_label.image = straightened_tk

def load_video():
    """
    Open a file dialog to select a video and process it.
    """
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
    )
    if video_path:
        process_video(video_path)

# -------------------- GUI Setup -------------------- #

# Create the main application window
root = tk.Tk()
root.title("C. elegans Worm Segmentation and Straightening")
root.geometry("850x350")

# Create a frame for the buttons
button_frame = tk.Frame(root)
button_frame.pack(side=tk.TOP, pady=10)

# Load Video Button
load_btn = tk.Button(
    button_frame,
    text="Load Video",
    command=load_video,
    font=("Helvetica", 14)
)
load_btn.pack(side=tk.LEFT, padx=20)

# Create frames for displaying images
image_frame = tk.Frame(root)
image_frame.pack(side=tk.BOTTOM, pady=10)

original_label = tk.Label(image_frame, text="Original Frame", font=("Helvetica", 12))
original_label.pack(side=tk.LEFT, padx=10)

segmented_label = tk.Label(image_frame, text="Straightened Worm", font=("Helvetica", 12))
segmented_label.pack(side=tk.LEFT, padx=10)

# Start the GUI event loop
root.mainloop()
