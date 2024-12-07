import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.decomposition import PCA
from PIL import Image, ImageTk
from tkinter import Toplevel
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from scipy.interpolate import splev, splprep
from skimage.morphology import skeletonize
import networkx as nx
import threading
import random
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# -------------------- Helper Functions -------------------- #

def display_results(title, image, scale=1, parent=None):
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
    contour = contour.squeeze()
    return contour  # contour Nx2 (x,y)

def random_points_within_mask(mask, num_points=200):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=int)
    if len(xs) < num_points:
        num_points = len(xs)
    idx = np.random.choice(len(xs), num_points, replace=False)
    sampled = np.column_stack((xs[idx], ys[idx]))  # (x,y)
    return sampled

def build_mst(points):
    dist_matrix = cdist(points, points)
    G = nx.Graph()
    for i in range(len(points)):
        G.add_node(i)

    edges = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            edges.append((i,j,dist_matrix[i,j]))
    edges = sorted(edges, key=lambda e: e[2])

    uf = list(range(len(points)))
    def find(x):
        while uf[x]!=x:
            uf[x]=uf[uf[x]]
            x=uf[x]
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        uf[ra]=rb

    MST = nx.Graph()
    for e in edges:
        a,b,w=e
        if find(a)!=find(b):
            union(a,b)
            MST.add_edge(a,b,weight=w)
        if MST.number_of_edges()==len(points)-1:
            break
    return MST

def longest_path_in_mst(MST):
    start_node = list(MST.nodes)[0]
    dist = nx.single_source_dijkstra_path_length(MST, start_node)
    end1 = max(dist, key=dist.get)
    dist2 = nx.single_source_dijkstra_path_length(MST, end1)
    end2 = max(dist2, key=dist2.get)
    path = nx.shortest_path(MST, end1, end2)
    return path

def closest_point_on_boundary(pt, boundary_points):
    dists = np.sum((boundary_points - pt)**2, axis=1)
    idx = np.argmin(dists)
    return boundary_points[idx]

def divide_boundary(boundary_points, c_first, c_last):
    bp = boundary_points
    dists_first = np.sum((bp - c_first)**2, axis=1)
    idx_first = np.argmin(dists_first)
    dists_last = np.sum((bp - c_last)**2, axis=1)
    idx_last = np.argmin(dists_last)

    if idx_first < idx_last:
        BL = bp[idx_first:idx_last+1]
        BR = np.concatenate((bp[idx_last:], bp[:idx_first+1]), axis=0)
    else:
        BL = bp[idx_last:idx_first+1]
        BR = np.concatenate((bp[idx_first:], bp[:idx_last+1]), axis=0)
    return BL, BR

def refine_control_points(control_points, BL, BR, iterations=5):
    treeL = KDTree(BL)
    treeR = KDTree(BR)
    cp = control_points.copy()
    for _ in range(iterations):
        _, idxL = treeL.query(cp)
        _, idxR = treeR.query(cp)
        cL = BL[idxL]
        cR = BR[idxR]
        cp = (cL + cR)/2.0
    return cp

def uniform_spacing(control_points, desired_num_points=None):
    cp = control_points
    dists = np.sqrt(np.sum(np.diff(cp, axis=0)**2, axis=1))
    dists = np.where(dists == 0, 1e-8, dists)
    length = np.sum(dists)
    if desired_num_points is None:
        desired_num_points = len(cp)

    cumdist = np.insert(np.cumsum(dists), 0, 0)
    new_positions = np.linspace(0, length, desired_num_points)
    new_cp = []
    for pos in new_positions:
        idx = np.searchsorted(cumdist, pos) - 1
        idx = max(0, min(idx, len(cp) - 2))
        t = (pos - cumdist[idx]) / dists[idx]
        p = cp[idx] * (1 - t) + cp[idx + 1] * t
        new_cp.append(p)
    return np.array(new_cp)

def refine_endpoints(backbone_cp, worm_mask):
    """
    Refine the endpoints of the backbone based on worm_mask properties.
    For example, ensure that the backbone reaches the actual ends by analyzing cross-sectional widths.
    This is a heuristic and may need tuning based on specific data characteristics.
    """
    if backbone_cp is None or len(backbone_cp) < 2:
        return backbone_cp

    # Parameters for endpoint refinement
    num_samples = min(5, len(backbone_cp))  # number of points near each end to sample width
    half_width_estimate = 30  # how far to check normal directions for width measurement

    # Fit spline to backbone to get tangent vectors
    if len(backbone_cp) < 4:
        # Not enough points for spline, skip
        return backbone_cp

    try:
        tck, u = splprep([backbone_cp[:, 0], backbone_cp[:, 1]], s=0, k=min(3, len(backbone_cp)-1))
    except:
        # Spline fitting error, return original
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
            if 0 <= sy_left < rows and 0 <= sx_left < cols:
                if worm_mask[sy_left, sx_left] == 255:
                    w_left = w
                else:
                    break
            else:
                break
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

    # Heuristic: if one end is significantly narrower, consider it as the tail and ensure backbone reaches it
    # For simplicity, here we just print the widths. Implement trimming or extension if needed.

    # Example Logic (can be refined based on data):
    # If the start is much wider, trim a few points from the start
    # If the end is much wider, trim a few points from the end

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

def bdb_plus_backbone(worm_mask, worm_contour):
    print("Computing BDB+ Backbone...")
    boundary_points = get_boundary_points_from_contour(worm_contour)
    sampled_pts = random_points_within_mask(worm_mask, num_points=200)

    if len(sampled_pts) < 2:
        print("Not enough sampled points for MST.")
        return None

    # Include extreme boundary points to ensure backbone spans the entire worm
    min_x_idx = np.argmin(boundary_points[:,0])
    max_x_idx = np.argmax(boundary_points[:,0])
    min_y_idx = np.argmin(boundary_points[:,1])
    max_y_idx = np.argmax(boundary_points[:,1])
    extreme_pts = boundary_points[[min_x_idx, max_x_idx, min_y_idx, max_y_idx]]
    sampled_pts = np.vstack([sampled_pts, extreme_pts])

    MST = build_mst(sampled_pts)
    path = longest_path_in_mst(MST)
    init_cp = sampled_pts[path]

    c_first = init_cp[0]
    c_last = init_cp[-1]
    BL, BR = divide_boundary(boundary_points, c_first, c_last)

    internal_cp = init_cp[1:-1]
    refined_cp = refine_control_points(internal_cp, BL, BR, iterations=6)

    if len(refined_cp) < 2:
        print("Refined control points too few.")
        return None

    first_cp = refined_cp[0]
    last_cp = refined_cp[-1]
    BL2, BR2 = divide_boundary(boundary_points, first_cp, last_cp)

    first_div = (closest_point_on_boundary(first_cp, BL2) + closest_point_on_boundary(first_cp, BR2))/2.0
    last_div = (closest_point_on_boundary(last_cp, BL2) + closest_point_on_boundary(last_cp, BR2))/2.0
    refined_cp = np.vstack([first_div, refined_cp, last_div])
    final_cp = uniform_spacing(refined_cp)

    print(f"Backbone computed with {len(final_cp)} control points.")

    # Refine endpoints to ensure backbone reaches the worm's ends
    final_cp = refine_endpoints(final_cp, worm_mask)

    return final_cp

def straighten_worm(segmented_worm, backbone_cp, half_width=20):
    print("Straightening worm...")
    if len(segmented_worm.shape) == 3:
        gray = cv2.cvtColor(segmented_worm, cv2.COLOR_BGR2GRAY)
    else:
        gray = segmented_worm.copy()
    rows, cols = gray.shape
    print(f"Segmented worm shape: {gray.shape}")

    if len(backbone_cp) < 4:
        print("Not enough points for spline fitting.")
        return None

    try:
        tck, u = splprep([backbone_cp[:,0], backbone_cp[:,1]], s=0, k=3)
    except Exception as e:
        print(f"Error fitting spline: {e}")
        return None

    unew = np.linspace(0, 1, 200)
    x_spline, y_spline = splev(unew, tck)
    dx, dy = splev(unew, tck, der=1)

    length = np.sqrt(dx**2 + dy**2)
    dx /= (length + 1e-12)
    dy /= (length + 1e-12)

    straight_height = len(unew)
    straight_width = 2 * half_width
    straightened_worm = np.zeros((straight_height, straight_width), dtype=gray.dtype)

    for i in range(straight_height):
        cx, cy = x_spline[i], y_spline[i]
        nx, ny = -dy[i], dx[i]
        for w in range(-half_width, half_width):
            sx = cx + w * nx
            sy = cy + w * ny
            sx_int = int(round(sx))
            sy_int = int(round(sy))
            if 0 <= sy_int < rows and 0 <= sx_int < cols:
                straightened_worm[i, w+half_width] = gray[sy_int, sx_int]

    nonzero_count = np.count_nonzero(straightened_worm)
    print(f"Nonzero pixel count in straightened image: {nonzero_count}")
    return straightened_worm if nonzero_count > 0 else None

# -------------------- Main Functions -------------------- #

# def segment_worm(frame, parent=None):
#     print("Segmenting worm...")
#     frame = cv2.resize(frame, (640, 480))
#     # display_results("Original Frame", frame, scale=2, parent=parent)

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # display_results("Grayscale Image", gray, scale=2, parent=parent)
#     plt.hist(gray.ravel(), 256, [0, 256], log = True)
#     plt.show()

#     gray_equalized = cv2.equalizeHist(gray)
#     display_results("Contrast Enhanced Image", gray_equalized, scale=2, parent=parent)

#     blurred = cv2.GaussianBlur(gray_equalized, (3, 3), 0)
#     display_results("Blurred Image", blurred, scale=2, parent=parent)

#     block_size = 15  
#     C = 3
#     binary = cv2.adaptiveThreshold(
#         blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY_INV, block_size, C)
#     display_results("Binary Image", binary, scale=2, parent=parent)

#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
#     display_results("Morphologically Cleaned Image", opened, scale=2, parent=parent)

#     contours, hierarchy = cv2.findContours(
#         opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     print(f"Total Contours Found: {len(contours)}")

#     if not contours:
#         messagebox.showerror("Error", "No contours found.")
#         return None, None, None, None, None

#     worm_contour = None
#     max_length = 0
#     for idx, cnt in enumerate(contours):
#         length = cv2.arcLength(cnt, closed=False)
#         x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(cnt)
#         aspect_ratio = float(h_cnt)/w_cnt if w_cnt != 0 else 0
#         print(f"Contour {idx+1}: Length={length:.2f}, Aspect Ratio={aspect_ratio:.2f}")
#         if length > max_length and aspect_ratio > 0.5:
#             max_length = length
#             worm_contour = cnt

#     if worm_contour is None:
#         messagebox.showerror("Error", "Worm contour could not be detected.")
#         return None, None, None, None, None

#     print(f"Selected Worm Contour Length: {max_length:.2f}")

#     worm_mask = np.zeros_like(gray)
#     cv2.drawContours(worm_mask, [worm_contour], -1, 255, -1)
#     display_results("Worm Mask", worm_mask, scale=2, parent=parent)

#     # Apply median filter to smooth the mask
#     worm_mask_filtered5 = cv2.medianBlur(worm_mask, 5)
#     display_results("Worm Mask Median Filter5", worm_mask_filtered5, scale=2, parent=parent)

#     segmented_worm = cv2.bitwise_and(frame, frame, mask=worm_mask_filtered5)
#     display_results("Segmented Worm", segmented_worm, scale=2, parent=parent)

#     # Get the bounding rectangle of the worm contour
#     x, y, w, h = cv2.boundingRect(worm_contour)
#     x, y = max(x, 0), max(y, 0)
#     x_end, y_end = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])

#     # Crop the segmented worm image to the bounding rectangle
#     cropped_worm = segmented_worm[y:y_end, x:x_end]
#     display_results("Cropped Worm", cropped_worm, scale=2, parent=parent)

#     print(f"Cropped worm shape: {cropped_worm.shape}, bounding box: ({x},{y}) to ({x_end},{y_end})")

#     return worm_mask_filtered5, worm_contour, cropped_worm, x, y

# def segment_worm(frame, OriginalSettings=True, parent=None):
#     print("Segmenting worm...")
#     frame = cv2.resize(frame, (640, 480))
#     display_results("Original Frame", frame, scale=2, parent=parent)

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     display_results("Grayscale Image", gray, scale=2, parent=parent)

#     gray_equalized = cv2.equalizeHist(gray)
#     display_results("Contrast Enhanced Image", gray_equalized, scale=2, parent=parent)


#     blurred = cv2.GaussianBlur(gray_equalized, (3, 3), 1)
#     display_results("Blurred Image", blurred, scale=2, parent=parent)

#     block_size = 15  
#     C = 3
#     binary = cv2.adaptiveThreshold(
#         blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY_INV, block_size, C)
#     display_results("Binary Image", binary, scale=2, parent=parent)

#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     print(f"Kernel: {kernel}")
#     opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
#     display_results("Morphologically Cleaned Image", opened, scale=2, parent=parent)

#     contours, hierarchy = cv2.findContours(
#         opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     print(f"Total Contours Found: {len(contours)}")

#     if not contours:
#         messagebox.showerror("Error", "No contours found.")
#         return None, None, None, None, None

#     worm_contour = None
#     max_length = 0
#     for idx, cnt in enumerate(contours):
#         length = cv2.arcLength(cnt, closed=False)
#         x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(cnt)
#         aspect_ratio = float(h_cnt)/w_cnt if w_cnt != 0 else 0
#         print(f"Contour {idx+1}: Length={length:.2f}, Aspect Ratio={aspect_ratio:.2f}")
#         if length > max_length and aspect_ratio > 0.5:
#             max_length = length
#             worm_contour = cnt

#     if worm_contour is None:
#         messagebox.showerror("Error", "Worm contour could not be detected.")
#         return None, None, None, None, None

#     print(f"Selected Worm Contour Length: {max_length:.2f}")

#     worm_mask = np.zeros_like(gray)
#     cv2.drawContours(worm_mask, [worm_contour], -1, 255, -1)
#     display_results("Worm Mask", worm_mask, scale=2, parent=parent)

#     # Apply median filter to smooth the mask
#     worm_mask_filtered5 = cv2.medianBlur(worm_mask, 7)
#     display_results("Worm Mask Median Filter5", worm_mask_filtered5, scale=2, parent=parent)

#     segmented_worm = cv2.bitwise_and(frame, frame, mask=worm_mask_filtered5)

#     display_results("Segmented Worm", segmented_worm, scale=2, parent=parent)

#     # Get the bounding rectangle of the worm contour
#     x, y, w, h = cv2.boundingRect(worm_contour)
#     x, y = max(x, 0), max(y, 0)
#     x_end, y_end = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])

#     # Crop the segmented worm image to the bounding rectangle
#     cropped_worm = segmented_worm[y:y_end, x:x_end]
#     display_results("Cropped Worm", cropped_worm, scale=2, parent=parent)

#     print(f"Cropped worm shape: {cropped_worm.shape}, bounding box: ({x},{y}) to ({x_end},{y_end})")

#     return worm_mask_filtered5, worm_contour, cropped_worm, x, y
def segment_worm(frame, OriginalSettings=True, parent=None):
    print("Segmenting worm...")
    frame = cv2.resize(frame, (640, 480))
    display_results("Original Frame", frame, scale=2, parent=parent)
    if OriginalSettings:
        # Original settings
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        plt.hist(gray.ravel(), 256, [0, 256], log=True)
        plt.show()

        gray_equalized = cv2.equalizeHist(gray)
        display_results("Contrast Enhanced Image", gray_equalized, scale=2, parent=parent)

        blurred = cv2.GaussianBlur(gray_equalized, (3, 3), 0)
        display_results("Blurred Image", blurred, scale=2, parent=parent)

        kernel_size = (3, 3)
        iterations = 2
        median_blur_size = 5

    else:
        # Alternative settings
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_results("Grayscale Image", gray, scale=2, parent=parent)

        gray_equalized = cv2.equalizeHist(gray)
        display_results("Contrast Enhanced Image", gray_equalized, scale=2, parent=parent)

        blurred = cv2.GaussianBlur(gray_equalized, (3, 3), 1)
        display_results("Blurred Image", blurred, scale=2, parent=parent)

        kernel_size = (5, 5)
        iterations = 1
        median_blur_size = 7

    block_size = 15
    C = 3
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, C)
    display_results("Binary Image", binary, scale=2, parent=parent)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)
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
        aspect_ratio = float(h_cnt) / w_cnt if w_cnt != 0 else 0
        # print(f"Contour {idx+1}: Length={length:.2f}, Aspect Ratio={aspect_ratio:.2f}")
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
    worm_mask_filtered5 = cv2.medianBlur(worm_mask, median_blur_size)
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

def histogram_matching_from_images(image1_path, image2_path, image3):
    """
    Matches the histogram of the mismatched image to the reference histograms of the other two.
    
    Parameters:
        image1_path (str): File path for the first image.
        image2_path (str): File path for the second image.
        image3_path (str): File path for the third (potentially mismatched) image.
    
    Returns:
        np.ndarray: The remapped image (mismatched image adjusted to match the reference histograms).
    """
    # Load the images
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Check if images were loaded correctly
    if image1 is None or image2 is None or image3 is None:
        raise ValueError("One or more images could not be loaded. Check the file paths.")

    # Calculate histograms (256 bins for 8-bit grayscale)
    hist1, _ = np.histogram(image1.ravel(), bins=256, range=(0, 256), density=True)
    hist2, _ = np.histogram(image2.ravel(), bins=256, range=(0, 256), density=True)
    hist3, _ = np.histogram(image3.ravel(), bins=256, range=(0, 256), density=True)

    # Normalize histograms and calculate cumulative distribution functions (CDFs)
    cdf1 = np.cumsum(hist1)
    cdf2 = np.cumsum(hist2)
    cdf3 = np.cumsum(hist3)

    # Identify the mismatched histogram using the Kolmogorov-Smirnov test
    ks1 = ks_2samp(hist1, hist3).statistic
    ks2 = ks_2samp(hist2, hist3).statistic

    print(ks1, ks2)

    mismatched_cdf = cdf3 if ks1 > 0.30 or ks2 > 0.30 else None

    OriginalSetting = True

    if mismatched_cdf is None:
        OriginalSetting = True
        print("Histograms are already similar. No adjustment needed.")
        return image3, OriginalSetting
    else: 
        OriginalSetting = False
        print("Histograms are not similar. Adjusting the mismatched image...")

    # Create the reference CDF as the average of the two similar histograms
    reference_cdf = (cdf1 + cdf2) / 2

    # Compute the mapping function
    mapping = np.interp(mismatched_cdf, reference_cdf, np.arange(256))

    # Apply the mapping to the mismatched image
    remapped_image = cv2.LUT(image3, mapping.astype(np.uint8))

    return remapped_image, OriginalSetting

def process_video(video_path, parent=None):
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
  
    frame, OriginalSetting = histogram_matching_from_images("temp_image_processing/Worm1.png", "temp_image_processing/Worm3.png", frame)

    result = segment_worm(frame, OriginalSetting, parent=parent)
    if result is None:
        print("Worm segmentation failed.")
        return
    worm_mask, worm_contour, cropped_worm, x, y = result

    if worm_mask is None or worm_contour is None or cropped_worm is None:
        print("Worm mask or contour could not be generated.")
        return

    backbone_cp = bdb_plus_backbone(worm_mask, worm_contour)
    if backbone_cp is not None and len(backbone_cp) > 0:
        disp = cv2.cvtColor(worm_mask, cv2.COLOR_GRAY2BGR)
        for p in backbone_cp.astype(int):
            if np.all(np.isfinite(p)):
                cv2.circle(disp, (p[0], p[1]), 3, (0, 0, 255), -1)
        display_results("BDB+ Backbone Control Points", disp, scale=2, parent=parent)

        # Adjust backbone coordinates to cropped frame
        backbone_cp_adjusted = backbone_cp.copy()
        backbone_cp_adjusted[:,0] -= x
        backbone_cp_adjusted[:,1] -= y

        # Ensure adjusted coordinates are within cropped image bounds
        backbone_cp_adjusted[:,0] = np.clip(backbone_cp_adjusted[:,0], 0, cropped_worm.shape[1]-1)
        backbone_cp_adjusted[:,1] = np.clip(backbone_cp_adjusted[:,1], 0, cropped_worm.shape[0]-1)

        # Straighten worm using adjusted backbone coordinates
        straightened = straighten_worm(cropped_worm, backbone_cp_adjusted, half_width=20)
        if straightened is not None:
            display_results("Straightened Worm", straightened, scale=2, parent=parent)
        else:
            print("Straightened worm is None or empty. Check debug prints for clues.")
    else:
        print("No valid backbone could be computed.")
        messagebox.showerror("Error", "Backbone computation failed.")

# -------------------- GUI Integration -------------------- #

def load_video():
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
    )
    if video_path:
        threading.Thread(target=process_video, args=(video_path, root), daemon=True).start()

def main():
    global root
    root = tk.Tk()
    root.title("C. elegans Worm Segmentation, BDB+ Backbone Detection and Straightening (Refined)")
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
