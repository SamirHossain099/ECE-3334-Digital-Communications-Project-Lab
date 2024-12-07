import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from tkinter import Toplevel
from scipy.interpolate import splev, splprep
from skimage.morphology import skeletonize
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import threading

# Toggle debug mode here
DEBUG_MODE = True

# -------------------- Helper Functions -------------------- #

def display_results(title, image, scale=1, parent=None):
    if image is None or image.size == 0:
        if DEBUG_MODE:
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

def skeletonize_worm(worm_mask):
    if DEBUG_MODE:
        print("Step 3 (Skeletonization): Performing morphological closing...")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_mask = cv2.morphologyEx(worm_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    if DEBUG_MODE:
        print("Applied morphological closing to the worm mask.")

    skeleton = skeletonize(closed_mask > 0).astype(np.uint8) * 255
    if DEBUG_MODE:
        print("Skeletonization complete.")
    return skeleton

def build_graph_from_skeleton(skeleton):
    G = nx.Graph()
    rows, cols = skeleton.shape
    ys, xs = np.where(skeleton > 0)
    points = list(zip(xs, ys))
    for (x, y) in points:
        G.add_node((x, y))
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
    if DEBUG_MODE:
        print("Built graph from skeleton.")
    return G

def find_skeleton_endpoints(skeleton):
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
    if DEBUG_MODE:
        print(f"Detected {len(endpoints)} skeleton endpoints.")
    return endpoints

def extract_backbone_from_skeleton(skeleton):
    G = build_graph_from_skeleton(skeleton)
    endpoints = find_skeleton_endpoints(skeleton)
    
    if len(endpoints) < 2:
        if DEBUG_MODE:
            print("Not enough endpoints detected in skeleton.")
        return None
    
    if len(endpoints) > 2:
        if DEBUG_MODE:
            print("Multiple endpoints found. Selecting two furthest apart.")
        max_distance = 0
        endpoint_pair = (endpoints[0], endpoints[1])
        for i in range(len(endpoints)):
            for j in range(i+1, len(endpoints)):
                dist = np.sqrt((endpoints[j][0] - endpoints[i][0])**2 + (endpoints[j][1] - endpoints[i][1])**2)
                if dist > max_distance:
                    max_distance = dist
                    endpoint_pair = (endpoints[i], endpoints[j])
        if DEBUG_MODE:
            print(f"Selected endpoints: {endpoint_pair[0]} and {endpoint_pair[1]}")
    else:
        endpoint_pair = (endpoints[0], endpoints[1])
        if DEBUG_MODE:
            print(f"Selected endpoints: {endpoint_pair[0]} and {endpoint_pair[1]}")

    try:
        path = nx.shortest_path(G, source=endpoint_pair[0], target=endpoint_pair[1])
    except nx.NetworkXNoPath:
        if DEBUG_MODE:
            print("No path found between endpoints.")
        return None
    
    backbone_cp = np.array(path)
    if DEBUG_MODE:
        print(f"Backbone extracted with {len(backbone_cp)} points.")
    return backbone_cp

def refine_endpoints(backbone_cp, worm_mask):
    if backbone_cp is None or len(backbone_cp) < 2:
        return backbone_cp
    num_samples = min(5, len(backbone_cp))
    half_width_estimate = 30

    if len(backbone_cp) < 4:
        if DEBUG_MODE:
            print("Not enough backbone points for spline fitting. Skipping endpoint refinement.")
        return backbone_cp

    try:
        tck, u = splprep([backbone_cp[:, 0], backbone_cp[:, 1]], s=0, k=min(3, len(backbone_cp)-1))
    except Exception as e:
        if DEBUG_MODE:
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
            if not (0 <= sy_left < rows and 0 <= sx_left < cols and worm_mask[sy_left, sx_left] == 255):
                break
            w_left = w
            if not (0 <= sy_right < rows and 0 <= sx_right < cols and worm_mask[sy_right, sx_right] == 255):
                break
            w_right = w
        return w_left + w_right

    start_widths = [measure_width(i) for i in range(num_samples)]
    end_widths = [measure_width(len(backbone_cp)-1 - i) for i in range(num_samples)]

    avg_start_width = np.mean(start_widths)
    avg_end_width = np.mean(end_widths)

    if DEBUG_MODE:
        print(f"Endpoint refinement: avg_start_width={avg_start_width:.2f}, avg_end_width={avg_end_width:.2f}")

    width_threshold = 10  
    trimmed_cp = backbone_cp.copy()
    if avg_start_width > avg_end_width + width_threshold:
        trim_points = 5
        if len(trimmed_cp) > trim_points + 2:
            trimmed_cp = trimmed_cp[trim_points:]
            if DEBUG_MODE:
                print("Trimmed start of the backbone.")
    elif avg_end_width > avg_start_width + width_threshold:
        trim_points = 5
        if len(trimmed_cp) > trim_points + 2:
            trimmed_cp = trimmed_cp[:-trim_points]
            if DEBUG_MODE:
                print("Trimmed end of the backbone.")

    return trimmed_cp

def extract_backbone_skeleton(worm_mask, parent=None):
    if DEBUG_MODE:
        print("Step 3: Skeletonization and Backbone Extraction")
    skeleton = skeletonize_worm(worm_mask)
    display_results("Step 3a: Skeleton", skeleton, scale=2, parent=parent)

    if DEBUG_MODE:
        print("Extracting backbone from skeleton...")
    backbone_cp = extract_backbone_from_skeleton(skeleton)
    if backbone_cp is None:
        if DEBUG_MODE:
            print("Failed to extract backbone.")
        return None

    # Show backbone points on skeleton
    disp = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    for p in backbone_cp.astype(int):
        cv2.circle(disp, (p[0], p[1]), 2, (0, 255, 0), -1)
    display_results("Step 3b: Backbone Points on Skeleton", disp, scale=2, parent=parent)

    # Refine endpoints
    backbone_cp = refine_endpoints(backbone_cp, worm_mask)

    # Step 4: Spline Fitting and Extension
    if backbone_cp is not None and len(backbone_cp) > 3:
        try:
            tck, u = splprep([backbone_cp[:,0], backbone_cp[:,1]], s=0, k=min(3, len(backbone_cp)-1))
            unew = np.linspace(0, 1, 200)
            x_spline, y_spline = splev(unew, tck)
            dx, dy = splev(unew, tck, der=1)

            length = np.sqrt(dx**2 + dy**2)
            dx /= (length + 1e-12)
            dy /= (length + 1e-12)

            # Extend the backbone at both ends
            start_extension = 3
            end_extension = 3

            x_start_extended = x_spline[0] - start_extension * dx[0]
            y_start_extended = y_spline[0] - start_extension * dy[0]
            x_end_extended = x_spline[-1] + end_extension * dx[-1]
            y_end_extended = y_spline[-1] + end_extension * dy[-1]

            extended_backbone = np.vstack([
                [x_start_extended, y_start_extended],
                np.column_stack((x_spline, y_spline)),
                [x_end_extended, y_end_extended]
            ])
            backbone_cp = extended_backbone

            # Display the spline and extended backbone
            spline_img = cv2.cvtColor(worm_mask, cv2.COLOR_GRAY2BGR)
            for i in range(len(x_spline)-1):
                cv2.line(spline_img, (int(x_spline[i]), int(y_spline[i])),
                                    (int(x_spline[i+1]), int(y_spline[i+1])), (0,0,255), 1)
            cv2.circle(spline_img, (int(x_start_extended), int(y_start_extended)), 3, (255,0,0), -1)
            cv2.circle(spline_img, (int(x_end_extended), int(y_end_extended)), 3, (255,0,0), -1)
            display_results("Step 4: Spline Fitted and Extended", spline_img, scale=2, parent=parent)

        except Exception as e:
            if DEBUG_MODE:
                print(f"Spline fitting error in extension step: {e}")

    return backbone_cp

def straighten_channel(channel, backbone_cp, half_width=20, visualization_img=None):
    if DEBUG_MODE:
        print("Step 5: Straightening Channels (Sampling Lines) ...")
    gray = channel.copy()
    rows, cols = gray.shape

    if len(backbone_cp) < 4:
        if DEBUG_MODE:
            print("Not enough backbone points for spline.")
        return None

    try:
        tck, u = splprep([backbone_cp[:,0], backbone_cp[:,1]], s=0, k=3)
    except Exception as e:
        if DEBUG_MODE:
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

    # Show sampling lines on the original worm image
    if DEBUG_MODE and visualization_img is not None:
        vis_img = visualization_img.copy()

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

        # Display all or every nth slicing line
        if DEBUG_MODE and visualization_img is not None and i % 2 == 0:
            p1 = (int(cx - half_width*nx), int(cy - half_width*ny))
            p2 = (int(cx + half_width*nx), int(cy + half_width*ny))
            cv2.line(vis_img, p1, p2, (0,255,0), 1)

    if DEBUG_MODE and visualization_img is not None:
        display_results("Step 5: Sampling Lines on Worm Image", vis_img, scale=2)

    return straightened_channel

def determine_half_width(backbone_cp, worm_mask):
    # Optional step if you want to automate half_width
    if DEBUG_MODE:
        print("Determining half_width automatically based on worm thickness.")
    tck, u = splprep([backbone_cp[:,0], backbone_cp[:,1]], s=0, k=min(3, len(backbone_cp)-1))
    unew = np.linspace(0, 1, 200)
    x_spline, y_spline = splev(unew, tck)
    dx, dy = splev(unew, tck, der=1)

    length = np.sqrt(dx**2 + dy**2)
    dx /= (length + 1e-12)
    dy /= (length + 1e-12)

    rows, cols = worm_mask.shape
    max_width = 0

    for i in range(len(x_spline)):
        cx, cy = x_spline[i], y_spline[i]
        nx, ny = -dy[i], dx[i]

        dist_left, dist_right = 0, 0
        for w in range(1, 500):
            sx_left = int(round(cx - w*nx))
            sy_left = int(round(cy - w*ny))
            if sx_left < 0 or sx_left >= cols or sy_left < 0 or sy_left >= rows or worm_mask[sy_left, sx_left] == 0:
                break
            dist_left = w

        for w in range(1, 500):
            sx_right = int(round(cx + w*nx))
            sy_right = int(round(cy + w*ny))
            if sx_right < 0 or sx_right >= cols or sy_right < 0 or sy_right >= rows or worm_mask[sy_right, sx_right] == 0:
                break
            dist_right = w

        width = dist_left + dist_right
        if width > max_width:
            max_width = width

    half_width = int(np.ceil(max_width / 2)) + 5
    return half_width

def straighten_worm(segmented_worm, backbone_cp, half_width=20, parent=None):
    if DEBUG_MODE:
        print("Step 6: Straightening the Worm using the Backbone ...")
    if len(segmented_worm.shape) == 3 and segmented_worm.shape[2] == 3:
        b_channel, g_channel, r_channel = cv2.split(segmented_worm)
    else:
        b_channel, g_channel, r_channel = [segmented_worm.copy()], [segmented_worm.copy()], [segmented_worm.copy()]

    # Show sampling lines on the segmented worm image
    straightened_b = straighten_channel(b_channel, backbone_cp, half_width, visualization_img=segmented_worm)
    straightened_g = straighten_channel(g_channel, backbone_cp, half_width)
    straightened_r = straighten_channel(r_channel, backbone_cp, half_width)

    if straightened_b is None or straightened_g is None or straightened_r is None:
        if DEBUG_MODE:
            print("One of the channels could not be straightened.")
        return None

    straightened_color = cv2.merge([straightened_b, straightened_g, straightened_r])
    if DEBUG_MODE:
        display_results("Step 6a: Straightened Worm Before Rotation", straightened_color, scale=2, parent=parent)

    # Rotate the straightened color image
    rotated_color = cv2.rotate(straightened_color, cv2.ROTATE_90_CLOCKWISE)
    if DEBUG_MODE:
        print("Rotated the straightened worm by 90 degrees.")

    # Straighten mask as well
    worm_mask = cv2.cvtColor(segmented_worm, cv2.COLOR_BGR2GRAY)
    straightened_mask = straighten_channel(worm_mask, backbone_cp, half_width)

    if straightened_mask is None:
        if DEBUG_MODE:
            print("Straightened mask is empty.")
        return None

    rotated_mask = cv2.rotate(straightened_mask, cv2.ROTATE_90_CLOCKWISE)
    if DEBUG_MODE:
        print("Rotated the straightened mask by 90 degrees.")

    # Binarize mask
    _, binary_rotated_mask = cv2.threshold(rotated_mask, 1, 255, cv2.THRESH_BINARY)
    if DEBUG_MODE:
        print("Binarized rotated mask.")

    # Clean mask
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned_mask = cv2.morphologyEx(binary_rotated_mask, cv2.MORPH_OPEN, kernel_clean, iterations=1)
    cleaned_mask = cv2.medianBlur(cleaned_mask, 5)
    if DEBUG_MODE:
        print("Cleaned mask with morphological operations and median filter.")

    cleaned_worm = cv2.bitwise_and(rotated_color, rotated_color, mask=cleaned_mask)
    if DEBUG_MODE:
        print("Applied cleaned mask to get final straightened worm image.")

    # Measure length from backbone
    distances = np.sqrt(np.sum(np.diff(backbone_cp, axis=0)**2, axis=1))
    worm_length = np.sum(distances)
    if DEBUG_MODE:
        print(f"Measured worm length from backbone: {worm_length:.2f} pixels.")

    display_results("Step 7: Final Straightened Worm", cleaned_worm, scale=2, parent=parent)

    return cleaned_worm, worm_length

def segment_worm(frame, OriginalSettings=True, parent=None):
    # Step 1: Display Original Frame
    display_results("Step 1: Original Frame", frame, scale=2, parent=parent)
    if DEBUG_MODE:
        print("Segmenting worm...")

    frame = cv2.resize(frame, (640, 480))
    if OriginalSettings:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if DEBUG_MODE:
            print("Showing histogram...")
            plt.hist(gray.ravel(), 256, [0, 256], log=True)
            plt.show()

        gray_equalized = cv2.equalizeHist(gray)
        display_results("Step 2a: Contrast Enhanced Image", gray_equalized, scale=2, parent=parent)

        blurred = cv2.GaussianBlur(gray_equalized, (3, 3), 0)
        display_results("Step 2b: Blurred Image", blurred, scale=2, parent=parent)

        kernel_size = (3, 3)
        iterations = 2
        median_blur_size = 5
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_results("Step 2a: Grayscale Image", gray, scale=2, parent=parent)

        gray_equalized = cv2.equalizeHist(gray)
        display_results("Step 2b: Contrast Enhanced Image", gray_equalized, scale=2, parent=parent)

        blurred = cv2.GaussianBlur(gray_equalized, (3, 3), 1)
        display_results("Step 2c: Blurred Image", blurred, scale=2, parent=parent)

        kernel_size = (5, 5)
        iterations = 1
        median_blur_size = 7

    block_size = 15
    C = 3
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, C)
    display_results("Step 2d: Binary Image", binary, scale=2, parent=parent)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)
    display_results("Step 2e: Morphologically Cleaned Image", opened, scale=2, parent=parent)

    contours, hierarchy = cv2.findContours(
        opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if DEBUG_MODE:
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
        if length > max_length and aspect_ratio > 0.5:
            max_length = length
            worm_contour = cnt

    if worm_contour is None:
        messagebox.showerror("Error", "Worm contour could not be detected.")
        return None, None, None, None, None

    if DEBUG_MODE:
        print(f"Selected Worm Contour Length: {max_length:.2f}")

    worm_mask = np.zeros_like(gray)
    cv2.drawContours(worm_mask, [worm_contour], -1, 255, -1)
    display_results("Step 2f: Worm Mask", worm_mask, scale=2, parent=parent)

    worm_mask_filtered5 = cv2.medianBlur(worm_mask, median_blur_size)
    display_results("Step 2g: Worm Mask Median Filter5", worm_mask_filtered5, scale=2, parent=parent)

    segmented_worm = cv2.bitwise_and(frame, frame, mask=worm_mask_filtered5)
    display_results("Step 2h: Segmented Worm", segmented_worm, scale=2, parent=parent)

    x, y, w, h = cv2.boundingRect(worm_contour)
    x, y = max(x, 0), max(y, 0)
    x_end, y_end = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])

    cropped_worm = segmented_worm[y:y_end, x:x_end]
    display_results("Step 2i: Cropped Worm", cropped_worm, scale=2, parent=parent)
    if DEBUG_MODE:
        print(f"Cropped worm shape: {cropped_worm.shape}, bounding box: ({x},{y}) to ({x_end},{y_end})")

    return worm_mask_filtered5, worm_contour, cropped_worm, x, y

def histogram_matching_from_images(image1_path, image2_path, image3):
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    if image1 is None or image2 is None or image3 is None:
        raise ValueError("One or more images could not be loaded.")

    hist1, _ = np.histogram(image1.ravel(), bins=256, range=(0, 256), density=True)
    hist2, _ = np.histogram(image2.ravel(), bins=256, range=(0, 256), density=True)
    hist3, _ = np.histogram(image3.ravel(), bins=256, range=(0, 256), density=True)

    cdf1 = np.cumsum(hist1)
    cdf2 = np.cumsum(hist2)
    cdf3 = np.cumsum(hist3)

    ks1 = ks_2samp(hist1, hist3).statistic
    ks2 = ks_2samp(hist2, hist3).statistic

    if DEBUG_MODE:
        print("Histogram KS stats:", ks1, ks2)

    mismatched_cdf = cdf3 if ks1 > 0.30 or ks2 > 0.30 else None
    OriginalSetting = True
    mapping = None

    if mismatched_cdf is None:
        OriginalSetting = True
        if DEBUG_MODE:
            print("No histogram adjustment needed.")
        return image3, OriginalSetting, mapping
    else:
        OriginalSetting = False
        if DEBUG_MODE:
            print("Adjusting histogram of the input image.")
        reference_cdf = (cdf1 + cdf2) / 2
        mapping = np.interp(mismatched_cdf, reference_cdf, np.arange(256)).astype(np.uint8)
        remapped_image = cv2.LUT(image3, mapping)
        return remapped_image, OriginalSetting, mapping

def process_video(video_path, parent=None):
    if DEBUG_MODE:
        print("Step 1: Loading and Processing Video")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open video file.")
        return
    ret, frame = cap.read()
    cap.release()
    if not ret:
        messagebox.showerror("Error", "Cannot read video file.")
        return

    frame, OriginalSetting, mapping = histogram_matching_from_images("temp_image_processing/Worm1.png", "temp_image_processing/Worm3.png", frame)

    result = segment_worm(frame, OriginalSetting, parent=parent)
    if result is None:
        if DEBUG_MODE:
            print("Worm segmentation failed.")
        return
    worm_mask, worm_contour, cropped_worm, x, y = result

    if worm_mask is None or worm_contour is None or cropped_worm is None:
        if DEBUG_MODE:
            print("No worm mask or contour generated.")
        return

    backbone_cp = extract_backbone_skeleton(worm_mask, parent=parent)
    if backbone_cp is not None and len(backbone_cp) > 0:
        backbone_cp_adjusted = backbone_cp.copy()
        backbone_cp_adjusted[:,0] -= x
        backbone_cp_adjusted[:,1] -= y
        backbone_cp_adjusted[:,0] = np.clip(backbone_cp_adjusted[:,0], 0, cropped_worm.shape[1]-1)
        backbone_cp_adjusted[:,1] = np.clip(backbone_cp_adjusted[:,1], 0, cropped_worm.shape[0]-1)

        half_width = 20  # or use determine_half_width if needed

        straightened, worm_length = straighten_worm(cropped_worm, backbone_cp_adjusted, half_width=half_width, parent=parent)
        if straightened is not None:
            if not OriginalSetting and mapping is not None:
                # Construct inverse mapping
                inverse_mapping = np.zeros(256, dtype=np.uint8)
                for t in range(256):
                    diff = np.abs(mapping.astype(int) - t)
                    i = np.argmin(diff)
                    inverse_mapping[t] = i

                # Apply inverse mapping
                if straightened.ndim == 3 and straightened.shape[2] == 3:
                    b, g, r = cv2.split(straightened)
                    b_restored = cv2.LUT(b, inverse_mapping)
                    g_restored = cv2.LUT(g, inverse_mapping)
                    r_restored = cv2.LUT(r, inverse_mapping)
                    straightened = cv2.merge([b_restored, g_restored, r_restored])
                else:
                    straightened = cv2.LUT(straightened, inverse_mapping)

                display_results("Final Straightened Worm with Original Intensity", straightened, scale=2, parent=parent)
            else:
                display_results("Final Straightened Worm (No Intensity Change)", straightened, scale=2, parent=parent)

            if DEBUG_MODE:
                print(f"Final worm length: {worm_length} pixels.")
        else:
            if DEBUG_MODE:
                print("Straightened worm is None.")
    else:
        if DEBUG_MODE:
            print("No valid backbone computed.")
        messagebox.showerror("Error", "Backbone computation failed.")

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
