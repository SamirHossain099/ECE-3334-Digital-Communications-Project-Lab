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

def bdb_plus_backbone(worm_mask, worm_contour):
    print("Computing BDB+ Backbone...")
    boundary_points = get_boundary_points_from_contour(worm_contour)
    sampled_pts = random_points_within_mask(worm_mask, num_points=200)

    if len(sampled_pts) < 2:
        print("Not enough sampled points for MST.")
        return None

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

def segment_worm(frame, parent=None):
    print("Segmenting worm...")
    frame = cv2.resize(frame, (640, 480))
    display_results("Original Frame", frame, scale=2, parent=parent)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    display_results("Grayscale Image", gray, scale=2, parent=parent)

    gray_equalized = cv2.equalizeHist(gray)
    display_results("Contrast Enhanced Image", gray_equalized, scale=2, parent=parent)

    blurred = cv2.GaussianBlur(gray_equalized, (5, 5), 0)
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
        return None, None, None

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
        return None, None, None

    print(f"Selected Worm Contour Length: {max_length:.2f}")

    worm_mask = np.zeros_like(gray)
    cv2.drawContours(worm_mask, [worm_contour], -1, 255, -1)
    display_results("Worm Mask", worm_mask, scale=2, parent=parent)

    filter5 = worm_mask.copy()

    worm_mask_filtered5 = cv2.medianBlur(filter5, 5)
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

    return worm_mask_filtered5, worm_contour, cropped_worm, x, y

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

    result = segment_worm(frame, parent=parent)
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
    root.title("C. elegans Worm Segmentation, BDB+ Backbone Detection and Straightening")
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
