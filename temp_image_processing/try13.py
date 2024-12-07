import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from tkinter import Toplevel
from scipy.interpolate import splev, splprep
from skimage.morphology import skeletonize
import networkx as nx
import threading

# -------------------- Helper Functions -------------------- #

def display_results(title, image, scale=1, parent=None):
    """
    Display an image in a Tkinter window.

    Parameters:
    - title (str): Title of the window.
    - image (np.array): Image to display.
    - scale (int): Scaling factor for display.
    - parent: Parent Tkinter window.
    """
    if image is None or image.size == 0:
        print(f"Cannot display {title}: Image is invalid.")
        return

    height, width = image.shape[:2]
    new_dimensions = (int(width * scale), int(height * scale))
    resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_LINEAR)

    if len(resized_image.shape) == 3 and resized_image.shape[2] == 3:
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = resized_image

    pil_image = Image.fromarray(image_rgb)
    tk_image = ImageTk.PhotoImage(pil_image)
    window = Toplevel(parent)
    window.title(title)
    label = tk.Label(window, image=tk_image)
    label.image = tk_image  
    label.pack()
    window.geometry(f"{new_dimensions[0]}x{new_dimensions[1]}")

def get_boundary_points_from_contour(contour):
    """
    Extract boundary points from a contour.

    Parameters:
    - contour (np.array): Contour points.

    Returns:
    - np.array: Nx2 array of (x, y) boundary points.
    """
    contour = contour.squeeze()
    return contour  # contour Nx2 (x,y)

def skeletonize_worm(worm_mask):
    """
    Skeletonize the binary worm mask after applying morphological closing to ensure continuity.

    Parameters:
    - worm_mask (np.array): Binary mask of the worm.

    Returns:
    - np.array: Skeletonized worm mask.
    """
    # Apply morphological closing to fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_mask = cv2.morphologyEx(worm_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    print("Applied morphological closing to the worm mask.")

    # Skeletonize
    skeleton = skeletonize(closed_mask > 0).astype(np.uint8) * 255
    print("Skeletonization complete.")
    return skeleton

def build_graph_from_skeleton(skeleton):
    """
    Build a graph from the skeleton where each skeleton pixel is a node connected to its neighbors.

    Parameters:
    - skeleton (np.array): Skeletonized worm mask.

    Returns:
    - networkx.Graph: Graph representing the skeleton.
    """
    G = nx.Graph()
    rows, cols = skeleton.shape
    ys, xs = np.where(skeleton > 0)
    points = list(zip(xs, ys))
    for (x, y) in points:
        G.add_node((x, y))
        # Check 8-connected neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx_, ny_ = x + dx, y + dy
                if 0 <= nx_ < cols and 0 <= ny_ < rows:
                    if skeleton[ny_, nx_] > 0:
                        neighbor = (nx_, ny_)
                        if neighbor in G.nodes:
                            G.add_edge((x, y), neighbor)
    print("Graph construction from skeleton complete.")
    return G

def find_skeleton_endpoints(skeleton):
    """
    Find endpoints in the skeleton. Endpoints are pixels with only one neighbor.

    Parameters:
    - skeleton (np.array): Skeletonized worm mask.

    Returns:
    - list: List of endpoint coordinates as tuples (x, y).
    """
    endpoints = []
    rows, cols = skeleton.shape
    ys, xs = np.where(skeleton > 0)
    for (x, y) in zip(xs, ys):
        count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx_, ny_ = x + dx, y + dy
                if 0 <= nx_ < cols and 0 <= ny_ < rows:
                    if skeleton[ny_, nx_] > 0:
                        count += 1
        if count == 1:
            endpoints.append((x, y))
    print(f"Detected {len(endpoints)} skeleton endpoints.")
    return endpoints

def extract_backbone_from_skeleton(skeleton):
    """
    Extract the backbone from the skeleton by finding the longest path between two furthest endpoints.

    Parameters:
    - skeleton (np.array): Skeletonized worm mask.

    Returns:
    - np.array or None: Array of backbone control points as (x, y) or None if extraction fails.
    """
    G = build_graph_from_skeleton(skeleton)
    endpoints = find_skeleton_endpoints(skeleton)
    
    if len(endpoints) < 2:
        print("Not enough endpoints detected in skeleton.")
        return None
    
    # If more than two endpoints, choose the pair with the maximum Euclidean distance
    if len(endpoints) > 2:
        print(f"Multiple endpoints detected. Selecting the two furthest apart.")
        max_distance = 0
        endpoint_pair = (endpoints[0], endpoints[1])
        for i in range(len(endpoints)):
            for j in range(i+1, len(endpoints)):
                dist = np.sqrt((endpoints[j][0] - endpoints[i][0])**2 + (endpoints[j][1] - endpoints[i][1])**2)
                if dist > max_distance:
                    max_distance = dist
                    endpoint_pair = (endpoints[i], endpoints[j])
        print(f"Selected endpoints: {endpoint_pair[0]} and {endpoint_pair[1]}")
    else:
        endpoint_pair = (endpoints[0], endpoints[1])
        print(f"Selected endpoints: {endpoint_pair[0]} and {endpoint_pair[1]}")
    
    try:
        path = nx.shortest_path(G, source=endpoint_pair[0], target=endpoint_pair[1])
    except nx.NetworkXNoPath:
        print("No path found between skeleton endpoints.")
        return None
    
    backbone_cp = np.array(path)
    print(f"Backbone extracted with {len(backbone_cp)} control points.")
    return backbone_cp

def refine_endpoints(backbone_cp, worm_mask):
    """
    Refine the endpoints of the backbone based on cross-sectional widths.
    If one end is significantly wider, trim a few points from that end.

    Parameters:
    - backbone_cp (np.array): Backbone control points as (x, y).
    - worm_mask (np.array): Binary mask of the worm.

    Returns:
    - np.array: Refined backbone control points.
    """
    if backbone_cp is None or len(backbone_cp) < 2:
        return backbone_cp

    # Parameters for endpoint refinement
    num_samples = min(5, len(backbone_cp))  # number of points near each end to sample width
    half_width_estimate = 30  # how far to check normal directions for width measurement

    # Fit spline to backbone to get tangent vectors
    if len(backbone_cp) < 4:
        # Not enough points for spline, skip
        print("Not enough backbone points for spline fitting. Skipping endpoint refinement.")
        return backbone_cp

    try:
        tck, u = splprep([backbone_cp[:, 0], backbone_cp[:, 1]], s=0, k=min(3, len(backbone_cp)-1))
    except Exception as e:
        print(f"Spline fitting error: {e}. Skipping endpoint refinement.")
        return backbone_cp

    unew = np.linspace(0, 1, len(backbone_cp))
    x_spline, y_spline = splev(unew, tck)
    dx, dy = splev(unew, tck, der=1)
    length = np.sqrt(dx**2 + dy**2)
    dx /= (length + 1e-12)
    dy /= (length + 1e-12)

    def measure_width(i):
        cx, cy = x_spline[i], y_spline[i]
        nx, ny = -dy[i], dx[i]
        w_left, w_right = 0, 0
        rows, cols = worm_mask.shape
        for w in range(1, half_width_estimate):
            sx_left = int(round(cx - w*nx))
            sy_left = int(round(cy - w*ny))
            sx_right = int(round(cx + w*nx))
            sy_right = int(round(cy + w*ny))
            # Check left side
            if 0 <= sy_left < rows and 0 <= sx_left < cols:
                if worm_mask[sy_left, sx_left] == 255:
                    w_left = w
                else:
                    break
            else:
                break
            # Check right side
            if 0 <= sy_right < rows and 0 <= sx_right < cols:
                if worm_mask[sy_right, sx_right] == 255:
                    w_right = w
                else:
                    break
            else:
                break
        return w_left + w_right

    # Measure widths near the start and end of the backbone
    start_widths = [measure_width(i) for i in range(num_samples)]
    end_widths = [measure_width(len(backbone_cp)-1 - i) for i in range(num_samples)]

    avg_start_width = np.mean(start_widths)
    avg_end_width = np.mean(end_widths)

    print(f"Endpoint refinement: avg_start_width={avg_start_width:.2f}, avg_end_width={avg_end_width:.2f}")

    # Heuristic: if one end is significantly wider, trim a few points from that end
    width_threshold = 10  # Adjust based on expected worm width differences

    trimmed_cp = backbone_cp.copy()
    trim_points = 0

    if avg_start_width > avg_end_width + width_threshold:
        # Start is wider; trim a few points from the start
        trim_points = 5  # Number of points to trim
        if len(trimmed_cp) > trim_points + 2:
            trimmed_cp = trimmed_cp[trim_points:]
            print(f"Trimming {trim_points} points from the start of the backbone.")
    elif avg_end_width > avg_start_width + width_threshold:
        # End is wider; trim a few points from the end
        trim_points = 5
        if len(trimmed_cp) > trim_points + 2:
            trimmed_cp = trimmed_cp[:-trim_points]
            print(f"Trimming {trim_points} points from the end of the backbone.")

    return trimmed_cp

def straighten_channel(channel, backbone_cp, half_width=20):
    """
    Straighten a single channel based on backbone control points.

    Parameters:
    - channel (np.array): Single channel image.
    - backbone_cp (np.array): Backbone control points as (x, y).
    - half_width (int): Half-width for sampling perpendicular to the backbone.

    Returns:
    - np.array or None: Straightened channel or None if unsuccessful.
    """
    print("Straightening a channel...")
    gray = channel.copy()
    rows, cols = gray.shape

    if len(backbone_cp) < 4:
        print("Not enough backbone points for spline fitting.")
        return None

    try:
        tck, u = splprep([backbone_cp[:,0], backbone_cp[:,1]], s=0, k=3)
    except Exception as e:
        print(f"Error fitting spline to backbone: {e}")
        return None

    unew = np.linspace(0, 1, 200)
    x_spline, y_spline = splev(unew, tck)
    dx, dy = splev(unew, tck, der=1)

    length = np.sqrt(dx**2 + dy**2)
    dx /= (length + 1e-12)
    dy /= (length + 1e-12)

    straight_height = len(unew)
    straight_width = 2 * half_width
    straightened_channel = np.zeros((straight_height, straight_width), dtype=gray.dtype)

    for i in range(straight_height):
        cx, cy = x_spline[i], y_spline[i]
        nx, ny = -dy[i], dx[i]
        for w in range(-half_width, half_width):
            sx = cx + w * nx
            sy = cy + w * ny
            sx_int = int(round(sx))
            sy_int = int(round(sy))
            if 0 <= sy_int < rows and 0 <= sx_int < cols:
                straightened_channel[i, w + half_width] = gray[sy_int, sx_int]

    nonzero_count = np.count_nonzero(straightened_channel)
    print(f"Nonzero pixel count in straightened channel: {nonzero_count}")

    if nonzero_count == 0:
        print("Straightened channel is empty.")
        return None

    return straightened_channel

def extract_backbone_skeleton(worm_mask):
    """
    Extract the backbone using skeletonization and graph-based longest path.

    Parameters:
    - worm_mask (np.array): Binary mask of the worm.

    Returns:
    - np.array or None: Array of backbone control points as (x, y) or None if extraction fails.
    """
    print("Performing skeletonization...")
    skeleton = skeletonize_worm(worm_mask)
    display_results("Skeleton", skeleton, scale=2)

    print("Extracting backbone from skeleton...")
    backbone_cp = extract_backbone_from_skeleton(skeleton)
    if backbone_cp is None:
        print("Failed to extract backbone from skeleton.")
        return None

    # Refine endpoints based on cross-sectional widths
    backbone_cp = refine_endpoints(backbone_cp, worm_mask)

    return backbone_cp

def straighten_worm(segmented_worm, backbone_cp, half_width=20):
    """
    Straighten the worm image based on the extracted backbone.
    Additional steps:
    - Rotate the straightened image by 90 degrees to make it horizontal.
    - Apply filtering on a binary version to clean up artifacts.
    - Calculate and print the length of the straightened worm.

    Parameters:
    - segmented_worm (np.array): Original segmented worm image.
    - backbone_cp (np.array): Backbone control points as (x, y).
    - half_width (int): Half-width for sampling perpendicular to the backbone.

    Returns:
    - tuple or None: Cleaned and rotated straightened worm color image and its length, or None if unsuccessful.
    """
    print("Straightening worm...")
    if len(segmented_worm.shape) == 3 and segmented_worm.shape[2] == 3:
        # Split into color channels
        b_channel, g_channel, r_channel = cv2.split(segmented_worm)
    else:
        # Handle grayscale or other
        b_channel, g_channel, r_channel = [segmented_worm.copy()], [segmented_worm.copy()], [segmented_worm.copy()]

    # Straighten each color channel
    straightened_b = straighten_channel(b_channel, backbone_cp, half_width)
    straightened_g = straighten_channel(g_channel, backbone_cp, half_width)
    straightened_r = straighten_channel(r_channel, backbone_cp, half_width)

    if straightened_b is None or straightened_g is None or straightened_r is None:
        print("Straightened color channels are empty.")
        return None

    # Merge color channels back
    straightened_color = cv2.merge([straightened_b, straightened_g, straightened_r])

    # Rotate the straightened color image by 90 degrees clockwise
    rotated_color = cv2.rotate(straightened_color, cv2.ROTATE_90_CLOCKWISE)
    print("Rotated the straightened worm by 90 degrees.")

    # Create a straightened mask
    worm_mask = cv2.cvtColor(segmented_worm, cv2.COLOR_BGR2GRAY)
    straightened_mask = straighten_channel(worm_mask, backbone_cp, half_width)

    if straightened_mask is None:
        print("Straightened mask is empty.")
        return None

    # Rotate the mask in the same way
    rotated_mask = cv2.rotate(straightened_mask, cv2.ROTATE_90_CLOCKWISE)
    print("Rotated the straightened mask by 90 degrees.")

    # Binarize the rotated mask
    _, binary_rotated_mask = cv2.threshold(rotated_mask, 1, 255, cv2.THRESH_BINARY)
    print("Converted rotated mask to binary image.")

    # Apply morphological opening to clean the mask
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned_mask = cv2.morphologyEx(binary_rotated_mask, cv2.MORPH_OPEN, kernel_clean, iterations=1)
    print("Applied morphological opening to clean the mask.")

    # Apply median filtering to further clean the mask
    cleaned_mask = cv2.medianBlur(cleaned_mask, 5)
    print("Applied median filtering to clean the mask.")

    # Apply the cleaned mask to the rotated color image
    cleaned_worm = cv2.bitwise_and(rotated_color, rotated_color, mask=cleaned_mask)
    print("Applied cleaned mask to the rotated color image.")

    # Calculate the length of the straightened worm
    worm_length_pixels = np.sum(np.any(cleaned_mask > 0, axis=0))
    print(f"Straightened worm length: {worm_length_pixels} pixels.")

    return cleaned_worm, worm_length_pixels

# -------------------- Main Functions -------------------- #

def segment_worm(frame, parent=None):
    """
    Segment the worm from the input frame.

    Parameters:
    - frame (np.array): Input video frame.
    - parent: Parent Tkinter window.

    Returns:
    - tuple: (worm_mask_filtered5, worm_contour, cropped_worm, x, y)
    """
    print("Segmenting worm...")
    frame = cv2.resize(frame, (640, 480))
    display_results("Original Frame", frame, scale=2, parent=parent)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    display_results("Grayscale Image", gray, scale=2, parent=parent)

    gray_equalized = cv2.equalizeHist(gray)
    display_results("Contrast Enhanced Image", gray_equalized, scale=2, parent=parent)

    blurred = cv2.GaussianBlur(gray_equalized, (3, 3), 0)
    display_results("Blurred Image", blurred, scale=2, parent=parent)

    block_size = 15  
    C = 3
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, C)
    display_results("Binary Image", binary, scale=2, parent=parent)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    display_results("Morphologically Cleaned Image", opened, scale=2, parent=parent)

    contours, hierarchy = cv2.findContours(
        opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Total Contours Found: {len(contours)}")

    if not contours:
        messagebox.showerror("Error", "No contours found.")
        return None, None, None, None, None

    worm_contour = None
    max_length = 0
    for idx, cnt in enumerate(contours):
        length = cv2.arcLength(cnt, closed=False)
        x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(cnt)
        aspect_ratio = float(h_cnt)/w_cnt if w_cnt != 0 else 0
        print(f"Contour {idx+1}: Length={length:.2f}, Aspect Ratio={aspect_ratio:.2f}")
        if length > max_length and aspect_ratio > 0.5:
            max_length = length
            worm_contour = cnt

    if worm_contour is None:
        messagebox.showerror("Error", "Worm contour could not be detected.")
        return None, None, None, None, None

    print(f"Selected Worm Contour Length: {max_length:.2f}")

    worm_mask = np.zeros_like(gray)
    cv2.drawContours(worm_mask, [worm_contour], -1, 255, -1)
    display_results("Worm Mask", worm_mask, scale=2, parent=parent)

    # Apply median filter to smooth the mask
    worm_mask_filtered5 = cv2.medianBlur(worm_mask, 5)
    display_results("Worm Mask Median Filter5", worm_mask_filtered5, scale=2, parent=parent)

    segmented_worm = cv2.bitwise_and(frame, frame, mask=worm_mask_filtered5)
    display_results("Segmented Worm", segmented_worm, scale=2, parent=parent)

    # Get the bounding rectangle of the worm contour
    x, y, w, h = cv2.boundingRect(worm_contour)
    x, y = max(x, 0), max(y, 0)
    x_end, y_end = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])

    # Crop the segmented worm image to the bounding rectangle
    cropped_worm = segmented_worm[y:y_end, x:x_end]
    display_results("Cropped Worm", cropped_worm, scale=2, parent=parent)

    print(f"Cropped worm shape: {cropped_worm.shape}, bounding box: ({x},{y}) to ({x_end},{y_end})")

    return worm_mask_filtered5, worm_contour, cropped_worm, x, y

def process_video(video_path, parent=None):
    """
    Process the input video to segment the worm, extract the backbone, and straighten the worm image.

    Parameters:
    - video_path (str): Path to the video file.
    - parent: Parent Tkinter window.
    """
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open video file.")
        return
    ret, frame = cap.read()
    cap.release()
    if not ret:
        messagebox.showerror("Error", "Cannot read video file.")
        return

    result = segment_worm(frame, parent=parent)
    if result is None:
        print("Worm segmentation failed.")
        return
    worm_mask, worm_contour, cropped_worm, x, y = result

    if worm_mask is None or worm_contour is None or cropped_worm is None:
        print("Worm mask or contour could not be generated.")
        return

    # Extract backbone using skeletonization
    backbone_cp = extract_backbone_skeleton(worm_mask)
    if backbone_cp is not None and len(backbone_cp) > 0:
        # Visualize backbone control points on the worm mask
        disp = cv2.cvtColor(worm_mask, cv2.COLOR_GRAY2BGR)
        for p in backbone_cp.astype(int):
            if np.all(np.isfinite(p)):
                cv2.circle(disp, (p[0], p[1]), 2, (0, 0, 255), -1)
        display_results("Skeleton-Based Backbone Control Points", disp, scale=2, parent=parent)

        # Adjust backbone coordinates to cropped frame
        backbone_cp_adjusted = backbone_cp.copy()
        backbone_cp_adjusted[:,0] -= x
        backbone_cp_adjusted[:,1] -= y

        # Ensure adjusted coordinates are within cropped image bounds
        backbone_cp_adjusted[:,0] = np.clip(backbone_cp_adjusted[:,0], 0, cropped_worm.shape[1]-1)
        backbone_cp_adjusted[:,1] = np.clip(backbone_cp_adjusted[:,1], 0, cropped_worm.shape[0]-1)

        # Straighten worm using adjusted backbone coordinates
        straightened, worm_length = straighten_worm(cropped_worm, backbone_cp_adjusted, half_width=20)
        if straightened is not None:
            display_results("Straightened Worm", straightened, scale=2, parent=parent)
            print(f"Straightened worm length: {worm_length} pixels.")
        else:
            print("Straightened worm is None or empty. Check debug prints for clues.")
    else:
        print("No valid backbone could be computed.")
        messagebox.showerror("Error", "Backbone computation failed.")

# -------------------- GUI Integration -------------------- #

def load_video():
    """
    Open a file dialog to select a video file and process it in a separate thread.
    """
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
    )
    if video_path:
        threading.Thread(target=process_video, args=(video_path, root), daemon=True).start()

def main():
    """
    Initialize the GUI application.
    """
    global root
    root = tk.Tk()
    root.title("C. elegans Worm Segmentation and Skeleton-Based Backbone Detection")
    root.geometry("1250x350")

    button_frame = tk.Frame(root)
    button_frame.pack(side=tk.TOP, pady=10)

    load_btn = tk.Button(
        button_frame,
        text="Load Video",
        command=load_video,
        font=("Helvetica", 14)
    )
    load_btn.pack(side=tk.LEFT, padx=20)

    image_frame = tk.Frame(root)
    image_frame.pack(side=tk.BOTTOM, pady=10)

    global original_label, slices_label
    original_label = tk.Label(image_frame, text="Original Frame", font=("Helvetica", 12))
    original_label.pack(side=tk.LEFT, padx=10)

    slices_label = tk.Label(image_frame, text="Slices", font=("Helvetica", 12))
    slices_label.pack(side=tk.LEFT, padx=10)

    root.mainloop()

if __name__ == "__main__":
    main()
