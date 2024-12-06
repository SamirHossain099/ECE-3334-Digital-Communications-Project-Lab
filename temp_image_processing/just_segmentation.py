import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.decomposition import PCA
from PIL import Image, ImageTk
from tkinter import Toplevel
from scipy.interpolate import splev, splprep
from skimage.morphology import skeletonize
import networkx as nx
import threading

# -------------------- Helper Functions -------------------- #

def display_results(title, image, scale=1, parent=None):
    """
    Display an image in a new Tkinter window with optional scaling.

    Parameters:
        title (str): Title for the display window.
        image (numpy.ndarray): Image to display.
        scale (int or float): Scaling factor for the display window.
        parent (Tkinter widget, optional): Parent window for modal behavior.
    """
    if image is None or image.size == 0:
        print(f"Cannot display {title}: Image is invalid.")
        return

    # Calculate new dimensions based on the scaling factor
    height, width = image.shape[:2]
    new_dimensions = (int(width * scale), int(height * scale))
    resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_LINEAR)

    # Convert BGR (OpenCV) to RGB (PIL)
    if len(resized_image.shape) == 3 and resized_image.shape[2] == 3:
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = resized_image

    pil_image = Image.fromarray(image_rgb)
    tk_image = ImageTk.PhotoImage(pil_image)

    # Create a new top-level window
    window = Toplevel(parent)
    window.title(title)

    # Create a label to hold the image
    label = tk.Label(window, image=tk_image)
    label.image = tk_image  # Keep a reference to prevent garbage collection
    label.pack()

    # Optionally, set the window size to fit the image
    window.geometry(f"{new_dimensions[0]}x{new_dimensions[1]}")

def get_ordered_skeleton_points(worm_mask):
    """
    Given a binary worm mask, skeletonize it and return a list of skeleton points
    ordered along the main path of the worm from one end to the other.
    """

    # Skeletonize
    worm_bool = worm_mask > 0
    skeleton = skeletonize(worm_bool)

    # Extract skeleton coordinates
    sk_points = np.argwhere(skeleton)  # (y, x)
    if len(sk_points) < 2:
        return None

    # Build a graph where each pixel is a node, edges connect neighboring pixels
    G = nx.Graph()
    for (y, x) in sk_points:
        G.add_node((y, x))
    # Connect neighbors
    # Consider 8-connectivity or 4-connectivity. Usually 8 is safer for a skeleton.
    for (y, x) in sk_points:
        for ny in range(y-1, y+2):
            for nx_ in range(x-1, x+2):
                if (ny, nx_) in G and not (ny == y and nx_ == x):
                    G.add_edge((y,x), (ny,nx_))

    # Find endpoints: skeleton endpoints have only one neighbor
    endpoints = [n for n in G.nodes if G.degree(n) == 1]

    if len(endpoints) < 2:
        # If we cannot find endpoints, try picking the furthest two nodes as endpoints
        # This is a fallback scenario.
        nodes_list = list(G.nodes)
        dist_mat = {}
        # Compute shortest paths from an arbitrary node
        # Just pick first node for BFS
        start_node = nodes_list[0]
        lengths = nx.single_source_shortest_path_length(G, start_node)
        # furthest node from start
        end1 = max(lengths, key=lengths.get)
        # furthest node from end1
        lengths2 = nx.single_source_shortest_path_length(G, end1)
        end2 = max(lengths2, key=lengths2.get)
        endpoints = [end1, end2]

    # Get the shortest path between endpoints - this gives us an ordered set of points
    path = nx.shortest_path(G, endpoints[0], endpoints[1])
    # path is a list of (y, x) from one endpoint to the other

    return path

# -------------------- Main Functions -------------------- #

def segment_worm(frame, parent=None):
    """
    Segment the worm from the video frame.

    Parameters:
        frame (numpy.ndarray): The video frame.
        parent (Tkinter widget, optional): Parent window for image displays.

    Returns:
        segmented_worm (numpy.ndarray): Image of the segmented worm.
    """
    print("Segmenting worm...")
    # Resize frame for consistency
    frame = cv2.resize(frame, (640, 480))
    display_results("Original Frame", frame, scale=2, parent=parent)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    display_results("Grayscale Image", gray, scale=2, parent=parent)

    # Enhance contrast using histogram equalization
    gray_equalized = cv2.equalizeHist(gray)
    display_results("Contrast Enhanced Image", gray_equalized, scale=2, parent=parent)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray_equalized, (5, 5), 0)
    display_results("Blurred Image", blurred, scale=2, parent=parent)

    # Apply adaptive thresholding to create binary image
    block_size = 15  # Must be odd number >=3
    C = 3           # Constant subtracted from the mean
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, C)
    display_results("Binary Image", binary, scale=2, parent=parent)

    # Morphological operations to remove small noise and fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    display_results("Morphologically Cleaned Image", opened, scale=2, parent=parent)

    # Find contours using OpenCV
    contours, hierarchy = cv2.findContours(
        opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Total Contours Found: {len(contours)}")

    if not contours:
        messagebox.showerror("Error", "No contours found.")
        return None

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
        return None

    print(f"Selected Worm Contour Length: {max_length:.2f}")

    # Create mask from the worm contour
    worm_mask = np.zeros_like(gray)
    cv2.drawContours(worm_mask, [worm_contour], -1, 255, -1)
    display_results("Worm Mask", worm_mask, scale=2, parent=parent)

    # Apply the mask to extract the worm
    segmented_worm = cv2.bitwise_and(frame, frame, mask=worm_mask)
    display_results("Segmented Worm", segmented_worm, scale=2, parent=parent)

    # Get the bounding rectangle of the worm contour
    x, y, w, h = cv2.boundingRect(worm_contour)
    x, y = max(x, 0), max(y, 0)
    x_end, y_end = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])

    # Crop the segmented worm image to the bounding rectangle
    cropped_worm = segmented_worm[y:y_end, x:x_end]
    display_results("Cropped Worm", cropped_worm, scale=2, parent=parent)

    return cropped_worm

# In your process_video function, after segment_worm:
def process_video(video_path, parent=None):
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open video file.")
        return None, None, None, None

    ret, frame = cap.read()
    cap.release()
    if not ret:
        messagebox.showerror("Error", "Cannot read video file.")
        return None, None, None, None

    segmented_worm = segment_worm(frame, parent=parent)
    if segmented_worm is None:
        return None, None, None, None

    return

# -------------------- GUI Integration -------------------- #

def load_video():
    """
    Open a file dialog to select a video and process it.
    """
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
    )
    if video_path:
        # Start processing in a separate thread to keep GUI responsive
        threading.Thread(target=process_video, args=(video_path, root), daemon=True).start()

# -------------------- Main Function -------------------- #

def main():
    # Create the main application window
    global root
    root = tk.Tk()
    root.title("C. elegans Worm Segmentation and Slicing")
    root.geometry("1250x350")  # Adjusted width to accommodate additional image

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

    global original_label, slices_label
    original_label = tk.Label(image_frame, text="Original Frame", font=("Helvetica", 12))
    original_label.pack(side=tk.LEFT, padx=10)

    slices_label = tk.Label(image_frame, text="Slices", font=("Helvetica", 12))
    slices_label.pack(side=tk.LEFT, padx=10)

    # Start the GUI event loop
    root.mainloop()

if __name__ == "__main__":
    main()
