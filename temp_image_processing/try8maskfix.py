import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.decomposition import PCA
from PIL import Image, ImageTk
from tkinter import Toplevel
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import KDTree
from scipy.interpolate import CubicSpline
from shapely.geometry import LineString, Polygon
import threading

# Initialize global variables to store rotated centerline, endpoints, and slices

global_rotated_centerline = []
global_rotated_endpoints = ()
global_slices = []

# -------------------- Helper Functions -------------------- #

def trace_boundary(binary_image):
    """
    Trace the boundary of a binary object using Moore's Neighbor Tracing algorithm.

    Parameters:
        binary_image (numpy.ndarray): Binary image containing the object (worm).

    Returns:
        boundary_coords (list of tuples): List of (row, col) coordinates of the boundary pixels.
    """
    print("Tracing boundary...")
    # Ensure the binary image is binary (0 and 1)
    binary_image = (binary_image > 0).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        print("No contours found.")
        return []

    # Assume the largest contour is the boundary
    boundary_contour = max(contours, key=cv2.contourArea)
    boundary_coords = boundary_contour.squeeze().tolist()  # Convert to list of (x, y)
    boundary_coords = [(y, x) for x, y in boundary_coords]  # Convert to (row, col)

    print(f"Total boundary points traced: {len(boundary_coords)}")
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
    print("Finding worm ends...")
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

    print(f"End 1 coordinates: {end1}")
    print(f"End 2 coordinates: {end2}")
    return end1, end2

def rotate_image_to_align_endpoints(image, end1, end2, parent=None):
    """
    Rotate the image so that the line connecting end1 and end2 is horizontal.

    Parameters:
        image (numpy.ndarray): The image to rotate.
        end1 (tuple): Coordinates of the first worm end (row, col).
        end2 (tuple): Coordinates of the second worm end (row, col).
        parent (Tkinter widget, optional): Parent window for image displays.

    Returns:
        rotated (numpy.ndarray): The rotated image with worm horizontally aligned.
        M (numpy.ndarray): The rotation matrix used for transforming coordinates.
    """
    print("Starting rotation to align worm endpoints horizontally...")
    
    # Calculate the angle between the two ends
    dy = end2[0] - end1[0]
    dx = end2[1] - end1[1]
    angle = np.degrees(np.arctan2(dy, dx))
    print(f"Calculated rotation angle: {angle:.2f} degrees")

    # Get image dimensions
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Compute the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate the sine and cosine of rotation angle
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # Compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to account for translation
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]

    # Perform the rotation
    rotated = cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_LINEAR, 
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    print("Image rotated successfully.")

    # Display the rotated image and wait for key press
    display_results("Rotated Worm", rotated, scale=2, parent=parent)

    return rotated, M

def find_centerline(binary_image, end1, end2, original_image, parent=None):
    """
    Find the centerline path from end1 to end2 using the skeleton.

    Parameters:
        binary_image (numpy.ndarray): Binary image containing the object (worm).
        end1 (tuple): Coordinates of the first end (row, col).
        end2 (tuple): Coordinates of the second end (row, col).
        original_image (numpy.ndarray): Original segmented worm image for visualization.
        parent (Tkinter widget, optional): Parent window for image displays.

    Returns:
        centerline_coords (list of tuples): Ordered list of (row, col) coordinates representing the centerline.
        centerline_image (numpy.ndarray): Original image with the centerline overlaid.
    """
    print("Starting centerline detection...")

    # Compute the skeleton using thinning
    try:
        skeleton = cv2.ximgproc.thinning(binary_image, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    except AttributeError:
        messagebox.showerror("Error", "cv2.ximgproc.thinning is not available. Install opencv-contrib-python.")
        return [], original_image

    # Create a graph from the skeleton
    skeleton_coords = np.column_stack(np.where(skeleton > 0))
    if skeleton_coords.size == 0:
        print("Skeletonization resulted in no pixels.")
        return [], original_image

    coord_indices = {tuple(coord): idx for idx, coord in enumerate(map(tuple, skeleton_coords))}

    row_indices = []
    col_indices = []
    data = []

    # Define 8-connected neighbors
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                (0, 1), (1, 1), (1, 0),
                (1, -1), (0, -1)]

    for idx, (r, c) in enumerate(skeleton_coords):
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if (nr, nc) in coord_indices:
                neighbor_idx = coord_indices[(nr, nc)]
                row_indices.append(idx)
                col_indices.append(neighbor_idx)
                data.append(1)  # Assuming uniform weight

    n_nodes = len(skeleton_coords)
    adjacency_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(n_nodes, n_nodes))

    # Find indices of end1 and end2 in skeleton_coords
    try:
        start_idx = coord_indices[end1]
        end_idx = coord_indices[end2]
    except KeyError:
        messagebox.showerror("Error", "Endpoints not found in the skeleton.")
        return [], original_image

    # Compute the shortest path from start to end
    dist_matrix, predecessors = shortest_path(csgraph=adjacency_matrix, directed=False, indices=start_idx, return_predecessors=True)

    if np.isinf(dist_matrix[end_idx]):
        messagebox.showerror("Error", "No path found between the endpoints.")
        return [], original_image

    # Reconstruct the path
    path = []
    current = end_idx
    while current != -9999:
        path.append(current)
        current = predecessors[current]
    path = path[::-1]  # Reverse to get path from start to end

    centerline_coords = [tuple(skeleton_coords[idx]) for idx in path]

    # Create an image to visualize the centerline
    centerline_image = original_image.copy()
    for coord in centerline_coords:
        cv2.circle(centerline_image, (coord[1], coord[0]), 1, (0, 255, 0), -1)  # Green for centerline

    # Optionally, draw lines between consecutive points for better visualization
    for i in range(1, len(centerline_coords)):
        cv2.line(centerline_image, 
                 (centerline_coords[i-1][1], centerline_coords[i-1][0]),
                 (centerline_coords[i][1], centerline_coords[i][0]),
                 (0, 255, 0), 2)

    display_results("Centerline", centerline_image, scale=2, parent=parent)

    print("Centerline detection completed.")
    return centerline_coords, centerline_image

def transform_coordinates(coords, rotation_matrix):
    """
    Transform a list of (row, col) coordinates using the rotation matrix.

    Parameters:
        coords (list of tuples): List of (row, col) coordinates.
        rotation_matrix (numpy.ndarray): 2x3 rotation matrix.

    Returns:
        transformed_coords (list of tuples): List of transformed (row, col) coordinates.
    """
    transformed_coords = []
    for coord in coords:
        row, col = coord
        # Convert (row, col) to (x, y) for transformation
        x, y = col, row
        transformed = rotation_matrix @ np.array([x, y, 1]).reshape((3, 1))
        transformed_x, transformed_y = transformed[0, 0], transformed[1, 0]
        transformed_coords.append((int(transformed_y), int(transformed_x)))  # Convert back to (row, col)
    return transformed_coords

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
        print("Empty image. Cannot display.")
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

    # Wait for a key press
    cv2.waitKey(1)
    cv2.destroyAllWindows()

def compute_centerline_spline(centerline_coords):
    """
    Fit a cubic spline to the centerline coordinates.

    Parameters:
        centerline_coords (list of tuples): List of (row, col) coordinates.

    Returns:
        spline_x (CubicSpline): Spline function for x-coordinate (col).
        spline_y (CubicSpline): Spline function for y-coordinate (row).
        s_vals (numpy.ndarray): Parameter values along the spline.
        total_length (float): Total length of the centerline.
    """
    # Remove consecutive duplicate points to ensure s is strictly increasing
    print("Removing consecutive duplicate points from centerline...")
    unique_centerline = [centerline_coords[0]]
    for coord in centerline_coords[1:]:
        if coord != unique_centerline[-1]:
            unique_centerline.append(coord)

    if len(unique_centerline) < 2:
        print("Insufficient unique centerline points after removing duplicates.")
        return None, None, None, None

    # Extract x and y coordinates
    y_coords = np.array([coord[0] for coord in unique_centerline])
    x_coords = np.array([coord[1] for coord in unique_centerline])

    # Compute cumulative distance (s) along the centerline
    distances = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2)
    s = np.hstack(([0], np.cumsum(distances)))
    total_length = s[-1]

    # Check for strictly increasing s
    if not np.all(np.diff(s) > 0):
        print("Cumulative distance s is not strictly increasing.")
        return None, None, None, None

    # Fit cubic splines to x and y as functions of s
    spline_x = CubicSpline(s, x_coords)
    spline_y = CubicSpline(s, y_coords)

    print("Spline fitting completed successfully.")
    return spline_x, spline_y, s, total_length

def compute_tangent_normal_vectors(spline_x, spline_y, s_vals):
    """
    Compute tangent and normal vectors at each point along the spline.

    Parameters:
        spline_x (CubicSpline): Spline function for x-coordinate.
        spline_y (CubicSpline): Spline function for y-coordinate.
        s_vals (numpy.ndarray): Array of parameter values along the spline.

    Returns:
        tangents (numpy.ndarray): Array of tangent vectors at each point.
        normals (numpy.ndarray): Array of normal vectors at each point.
    """
    # Compute first derivatives
    dx_ds = spline_x.derivative()(s_vals)
    dy_ds = spline_y.derivative()(s_vals)

    # Tangent vectors
    tangents = np.vstack((dx_ds, dy_ds)).T
    tangents_norm = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents_unit = tangents / tangents_norm

    # Normal vectors (rotate tangents by 90 degrees)
    normals_unit = np.hstack((-tangents_unit[:, [1]], tangents_unit[:, [0]]))

    return tangents_unit, normals_unit

def slice_worm(rotated_image, transformed_centerline, rotated_end1, rotated_end2, parent=None):
    """
    Slice the worm along its centerline using regular sampling and normal vectors.

    Parameters:
        rotated_image (numpy.ndarray): Rotated image of the worm.
        transformed_centerline (list of tuples): Rotated centerline coordinates.
        rotated_end1 (tuple): Rotated coordinates of the first end (x, y).
        rotated_end2 (tuple): Rotated coordinates of the second end (x, y).
        parent (Tkinter widget, optional): Parent window for image displays.

    Returns:
        slices (list of tuples): List of tuples containing slice endpoints [(pt1, pt2), ...].
        slice_image (numpy.ndarray): Image showing only the endpoints and slices.
    """
    print("Starting worm slicing with regular sampling and normal vectors...")

    # Create an empty image to draw slices
    slice_image = np.zeros_like(rotated_image)

    # Convert the rotated image to grayscale and binary for boundary detection
    gray_rotated = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
    _, binary_rotated = cv2.threshold(gray_rotated, 10, 255, cv2.THRESH_BINARY)

    # Find contours to get the boundary
    contours, _ = cv2.findContours(binary_rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found in rotated image.")
        return [], slice_image

    # Assuming the largest contour is the worm's boundary
    worm_contour = max(contours, key=cv2.contourArea)
    worm_contour = worm_contour.squeeze()

    # Create a binary mask of the worm
    worm_mask = np.zeros_like(binary_rotated)
    cv2.drawContours(worm_mask, [worm_contour], -1, 255, thickness=cv2.FILLED)

    # Fit splines to the centerline
    spline_fit = compute_centerline_spline(transformed_centerline)
    if spline_fit[0] is None:
        messagebox.showerror("Error", "Spline fitting failed due to insufficient unique centerline points.")
        return [], slice_image

    spline_x, spline_y, s_vals, total_length = spline_fit

    # Compute tangent and normal vectors
    tangents, normals = compute_tangent_normal_vectors(spline_x, spline_y, s_vals)

    # Define slicing parameters
    sampling_interval = 1  # pixels between slices
    slice_length = 20       # total length of each slice (can be adaptive)

    # Generate regularly spaced sample points along the centerline
    num_samples = int(total_length // sampling_interval)
    sampled_s_vals = np.linspace(0, total_length, num_samples)

    slices = []
    for s in sampled_s_vals:
        # Compute current point on spline (x, y)
        x_current = spline_x(s)
        y_current = spline_y(s)
        pt_current = (int(np.round(x_current)), int(np.round(y_current)))  # (x, y)

        # Compute derivatives for tangent
        dx_ds = spline_x.derivative()(s)
        dy_ds = spline_y.derivative()(s)

        # Compute tangent vector
        tangent = np.array([dx_ds, dy_ds])
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm == 0:
            print(f"Zero tangent vector at s={s}. Skipping slice.")
            continue
        tangent_unit = tangent / tangent_norm

        # Compute normal vector by rotating tangent by 90 degrees
        normal_unit = np.array([-tangent_unit[1], tangent_unit[0]])

        # Define slice endpoints along the normal vector
        half_length = slice_length / 2
        pt1 = pt_current + normal_unit * half_length
        pt2 = pt_current - normal_unit * half_length

        # Convert to integer coordinates
        pt1 = (int(np.round(pt1[0])), int(np.round(pt1[1])))
        pt2 = (int(np.round(pt2[0])), int(np.round(pt2[1])))

        # Check and clip the points within the worm mask
        def clip_point(pt):
            x, y = pt
            x = np.clip(x, 0, worm_mask.shape[1] - 1)
            y = np.clip(y, 0, worm_mask.shape[0] - 1)
            if worm_mask[y, x] == 0:
                # Find the nearest boundary point along the normal
                line = LineString([(pt_current[0], pt_current[1]), (x, y)])
                intersection = line.intersection(Polygon(worm_contour))
                if intersection.is_empty:
                    return pt
                if isinstance(intersection, LineString):
                    intersection = list(intersection.coords)[-1]
                elif isinstance(intersection, (tuple, list, np.ndarray)):
                    intersection = intersection
                else:
                    intersection = (x, y)
                return (int(np.round(intersection[0])), int(np.round(intersection[1])))
            return pt

        pt1 = clip_point(pt1)
        pt2 = clip_point(pt2)

        # Append the slice endpoints
        slices.append((pt1, pt2))

        # Draw the slice on the slice_image
        cv2.line(slice_image, pt1, pt2, (255, 255, 255), 1)

    # Draw the endpoints
    rotated_end1_xy = (rotated_end1[0], rotated_end1[1])  # (x, y)
    rotated_end2_xy = (rotated_end2[0], rotated_end2[1])

    cv2.circle(slice_image, rotated_end1_xy, 5, (255, 255, 255), -1)
    cv2.circle(slice_image, rotated_end2_xy, 5, (255, 255, 255), -1)

    # Display the slice image
    display_results("Worm Slices", slice_image, scale=2, parent=parent)

    print(f"Total slices created: {len(slices)}")
    return slices, slice_image

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
    # display_results("Grayscale Image", gray, scale=2, parent=parent)

    # Enhance contrast using histogram equalization
    gray_equalized = cv2.equalizeHist(gray)
    # display_results("Contrast Enhanced Image", gray_equalized, scale=2, parent=parent)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray_equalized, (5, 5), 0)
    # display_results("Blurred Image", blurred, scale=2, parent=parent)

    # Apply adaptive thresholding to create binary image
    block_size = 15  # Must be odd number >=3
    C = 3           # Constant subtracted from the mean
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, C)
    # display_results("Binary Image", binary, scale=2, parent=parent)

    # Morphological operations to remove small noise and fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    # display_results("Morphologically Cleaned Image", opened, scale=2, parent=parent)

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

    filter5 = worm_mask.copy()
    
    worm_mask_filtered5 = cv2.medianBlur(filter5, 5)
    display_results("Worm Mask Median Filter5", worm_mask_filtered5, scale=2, parent=parent)
    
    # Apply the mask to extract the worm
    segmented_worm = cv2.bitwise_and(frame, frame, mask=worm_mask_filtered5)
    display_results("Segmented Worm", segmented_worm, scale=2, parent=parent)

    # Get the bounding rectangle of the worm contour
    x, y, w, h = cv2.boundingRect(worm_contour)
    x, y = max(x, 0), max(y, 0)
    x_end, y_end = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])

    # Crop the segmented worm image to the bounding rectangle
    cropped_worm = segmented_worm[y:y_end, x:x_end]
    display_results("Cropped Worm", cropped_worm, scale=2, parent=parent)

    return cropped_worm

def straighten_worm(segmented_worm, parent=None):
    """
    Trace the boundary, find the ends of the segmented worm, find the centerline,
    rotate the image so that the worm is horizontally aligned, and overlay the centerline.

    Additionally, output the rotated centerline coordinates and rotated endpoints.

    Parameters:
        segmented_worm (numpy.ndarray): Image of the segmented worm.
        parent (Tkinter widget, optional): Parent window for image displays.

    Returns:
        rotated (numpy.ndarray): The rotated image with the worm horizontally aligned.
        transformed_centerline (list of tuples): Rotated centerline coordinates.
        rotated_end1 (tuple): Rotated coordinates of the first end.
        rotated_end2 (tuple): Rotated coordinates of the second end.
    """
    print("Starting straighten_worm function...")

    # -------------------- Step 1: Grayscale and Binary Conversion -------------------- #
    gray = cv2.cvtColor(segmented_worm, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    display_results("Binary Worm Mask", binary, scale=2, parent=parent)

    # -------------------- Step 2: Boundary Tracing -------------------- #
    boundary_coords = trace_boundary(binary)
    if not boundary_coords:
        messagebox.showerror("Error", "Boundary tracing failed.")
        return segmented_worm, None, None, None

    # -------------------- Step 3: Visualize Boundary Traced -------------------- #
    boundary_image = segmented_worm.copy()
    for coord in boundary_coords:
        cv2.circle(boundary_image, (coord[1], coord[0]), 1, (0, 0, 255), -1)  # Red for boundary
    display_results("Boundary Traced", boundary_image, scale=2, parent=parent)

    # -------------------- Step 4: Find Worm Ends -------------------- #
    end1, end2 = find_worm_ends(boundary_coords)

    if end1 and end2:
        # Mark ends on the image
        cv2.circle(boundary_image, (end1[1], end1[0]), 5, (0, 255, 0), -1)  # Green for end1
        cv2.circle(boundary_image, (end2[1], end2[0]), 5, (255, 0, 0), -1)  # Blue for end2
        display_results("Worm Ends Identified", boundary_image, scale=2, parent=parent)
    else:
        print("Could not identify both ends of the worm.")
        messagebox.showerror("Error", "Worm ends could not be identified.")
        return segmented_worm, None, None, None

    # -------------------- Step 5: Find Centerline -------------------- #
    centerline_coords, centerline_image = find_centerline(binary, end1, end2, segmented_worm, parent=parent)

    if not centerline_coords:
        messagebox.showerror("Error", "Centerline detection failed.")
        return segmented_worm, None, None, None

    print("Centerline detected successfully.")

    # -------------------- Step 6: Rotate Image to Align Ends Horizontally -------------------- #
    rotated, rotation_matrix = rotate_image_to_align_endpoints(segmented_worm, end1, end2, parent=parent)

    # -------------------- Step 7: Transform Centerline and Endpoints -------------------- #
    # Transform centerline coordinates based on rotation matrix
    transformed_centerline = transform_coordinates(centerline_coords, rotation_matrix)

    # Transform endpoints
    rotated_end1 = transform_coordinates([end1], rotation_matrix)[0]
    rotated_end2 = transform_coordinates([end2], rotation_matrix)[0]

    print(f"Rotated End1: {rotated_end1}")
    print(f"Rotated End2: {rotated_end2}")

    # -------------------- Step 8: Overlay Centerline on Rotated Image -------------------- #
    # Overlay the transformed centerline on the rotated image
    for coord in transformed_centerline:
        cv2.circle(rotated, (coord[1], coord[0]), 1, (0, 255, 0), -1)  # Green for centerline

    # Optionally, draw lines between consecutive points for better visualization
    for i in range(1, len(transformed_centerline)):
        cv2.line(rotated,
                 (transformed_centerline[i-1][1], transformed_centerline[i-1][0]),
                 (transformed_centerline[i][1], transformed_centerline[i][0]),
                 (0, 255, 0), 2)

    # Mark rotated endpoints on the rotated image
    cv2.circle(rotated, (rotated_end1[1], rotated_end1[0]), 5, (0, 255, 255), -1)  # Yellow for rotated end1
    cv2.circle(rotated, (rotated_end2[1], rotated_end2[0]), 5, (255, 255, 0), -1)  # Cyan for rotated end2

    # Display the final rotated image with centerline and rotated endpoints
    display_results("Rotated Worm with Centerline and Rotated Ends", rotated, scale=2, parent=parent)

    print("straighten_worm function completed with rotation, centerline overlay, and rotated endpoints.")
    return rotated, transformed_centerline, rotated_end1, rotated_end2

def process_video(video_path, parent=None):
    """
    Master function to process the video, segment the worm, and straighten it.

    Additionally, outputs the rotated centerline and rotated endpoints.

    Parameters:
        video_path (str): Path to the video file.
        parent (Tkinter widget, optional): Parent window for image displays.

    Returns:
        rotated (numpy.ndarray): The rotated image with the worm horizontally aligned.
        transformed_centerline (list of tuples): Rotated centerline coordinates.
        rotated_end1 (tuple): Rotated coordinates of the first end.
        rotated_end2 (tuple): Rotated coordinates of the second end.
    """
    print(f"Processing video: {video_path}")
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open video file.")
        return None, None, None, None

    # Read the first frame
    ret, frame = cap.read()
    cap.release()
    if not ret:
        messagebox.showerror("Error", "Cannot read video file.")
        return None, None, None, None

    # Segment the worm
    segmented_worm = segment_worm(frame, parent=parent)
    if segmented_worm is None:
        return None, None, None, None

    # # Straighten the worm (trace boundary, find ends, find centerline, rotate)
    # straightened_worm, transformed_centerline, rotated_end1, rotated_end2 = straighten_worm(segmented_worm, parent=parent)
    # if straightened_worm is None:
    #     return None, None, None, None

    # # Slice the worm
    # slices, slice_image = slice_worm(straightened_worm, transformed_centerline, rotated_end1, rotated_end2, parent=parent)

    # # Display the slices image in the GUI
    # display_result_slices(slice_image)

    # # Output the additional data (rotated_centerline, rotated_end1, rotated_end2, slices)
    # # These can be stored or processed further as needed
    # print(f"Rotated Centerline Coordinates: {transformed_centerline}")
    # print(f"Rotated Endpoints: {rotated_end1}, {rotated_end2}")
    # print(f"Number of slices: {len(slices)}")

    # # Store the outputs globally or pass them to other components as needed
    # global global_rotated_centerline, global_rotated_endpoints, global_slices
    # global_rotated_centerline = transformed_centerline
    # global_rotated_endpoints = (rotated_end1, rotated_end2)
    # global_slices = slices

    # # Display the final result in the GUI
    # display_result(frame, slice_image)  # Display the slices image instead of straightened worm

    # return straightened_worm, transformed_centerline, rotated_end1, rotated_end2

# -------------------- GUI Integration -------------------- #

def display_result(original_frame, slices_image):
    """
    Display the original frame and slices image in the GUI.

    Parameters:
        original_frame (numpy.ndarray): Original video frame.
        slices_image (numpy.ndarray): Image showing the slices and endpoints.
    """
    # Convert images to RGB for displaying in Tkinter
    original_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
    slices_rgb = cv2.cvtColor(slices_image, cv2.COLOR_BGR2RGB)

    # Convert images to PIL format
    original_pil = Image.fromarray(original_rgb)
    slices_pil = Image.fromarray(slices_rgb)

    # Resize images for display
    original_pil = original_pil.resize((400, 300), Image.Resampling.LANCZOS)

    # For the slices image, maintain aspect ratio
    max_width, max_height = 400, 300
    width, height = slices_pil.size
    scaling_factor = min(max_width / width, max_height / height)
    new_width, new_height = int(width * scaling_factor), int(height * scaling_factor)
    slices_pil = slices_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Convert images to ImageTk format
    original_tk = ImageTk.PhotoImage(original_pil)
    slices_tk = ImageTk.PhotoImage(slices_pil)

    # Display images in the GUI
    original_label.config(image=original_tk)
    original_label.image = original_tk
    slices_label.config(image=slices_tk)
    slices_label.image = slices_tk

def display_result_slices(slice_image):
    """
    Display the slices image in the GUI.

    Parameters:
        slice_image (numpy.ndarray): Image showing the slices and endpoints.
    """
    # Convert image to RGB for displaying in Tkinter
    slices_rgb = cv2.cvtColor(slice_image, cv2.COLOR_BGR2RGB)

    # Convert image to PIL format
    slices_pil = Image.fromarray(slices_rgb)

    # Resize image for display
    max_width, max_height = 400, 300
    width, height = slices_pil.size
    scaling_factor = min(max_width / width, max_height / height)
    new_width, new_height = int(width * scaling_factor), int(height * scaling_factor)
    slices_pil = slices_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Convert image to ImageTk format
    slices_tk = ImageTk.PhotoImage(slices_pil)

    # Display image in the GUI
    slices_label.config(image=slices_tk)
    slices_label.image = slices_tk

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
