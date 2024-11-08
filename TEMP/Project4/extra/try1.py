import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    # Resize for consistency
    scale_percent = 800 / image.shape[1]
    image = cv2.resize(image, None, fx=scale_percent, fy=scale_percent, interpolation=cv2.INTER_AREA)
    # Convert to grayscale and blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    return image, edges

def find_card_contour(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

def order_points(pts):
    # Order points in a consistent order: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    return rect

def get_warped_image(image, contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) != 4:
        return None
    rect = order_points(approx.reshape(4, 2))
    (tl, tr, br, bl) = rect
    # Compute the width and height
    widthA = np.hypot(br[0] - bl[0], br[1] - bl[1])
    widthB = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
    maxWidth = int(max(widthA, widthB))
    heightA = np.hypot(tr[0] - br[0], tr[1] - br[1])
    heightB = np.hypot(tl[0] - bl[0], tl[1] - bl[1])
    maxHeight = int(max(heightA, heightB))
    # Destination points
    dst = np.array([[0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype="float32")
    # Perspective transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def preprocess_warped_image(warped):
    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # Apply binary inversion thresholding
    _, thresh = cv2.threshold(gray_warped, 200, 255, cv2.THRESH_BINARY_INV)
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Remove small noise
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    # Close gaps
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closing

def connected_components_analysis(binary_image):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    return num_labels, labels, stats, centroids

def extract_features(symbol_binary):
    # Compute Hu Moments
    moments = cv2.moments(symbol_binary)
    hu_moments = cv2.HuMoments(moments).flatten()
    # Compute aspect ratio
    x, y, w, h = cv2.boundingRect(symbol_binary)
    aspect_ratio = w / float(h)
    # Count number of holes
    contours, hierarchy = cv2.findContours(symbol_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    num_holes = 0
    for idx, cnt in enumerate(contours):
        if hierarchy[0][idx][3] != -1:
            num_holes += 1
    # Skeletonization can be added to analyze strokes
    return {
        'hu_moments': hu_moments,
        'aspect_ratio': aspect_ratio,
        'num_holes': num_holes,
        # Add more features as needed
    }

def recognize_symbol(features):
    aspect_ratio = features['aspect_ratio']
    num_holes = features['num_holes']
    # Example rule-based recognition
    if num_holes == 0:
        if aspect_ratio < 0.5:
            return '1'  # Possibly '1' or 'I' for 'Ace'
        elif aspect_ratio >= 0.8:
            return 'Spade or Club'
    elif num_holes == 1:
        if aspect_ratio >= 0.8:
            return 'Heart or Diamond'
        else:
            return 'Number with a hole (e.g., 4, 6, 9)'
    # Add more rules based on observed features
    return 'Unknown'

def process_image(image_path):
    # Steps 1 and 2
    image, edges = load_and_preprocess_image(image_path)
    card_contour = find_card_contour(edges)
    if card_contour is None:
        print("Card contour not found.")
        return
    warped = get_warped_image(image, card_contour)
    if warped is None:
        print("Could not warp image.")
        return
    # Step 3
    binary_image = preprocess_warped_image(warped)
    # Step 4
    num_labels, labels, stats, centroids = connected_components_analysis(binary_image)
    # Step 5 and 6
    h, w = warped.shape[:2]
    symbols = []
    for i in range(1, num_labels):  # Skip background
        x, y, w_comp, h_comp, area = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                                     stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], \
                                     stats[i, cv2.CC_STAT_AREA]
        if 100 < area < 1000:  # Adjust thresholds as needed
            symbol_binary = binary_image[y:y+h_comp, x:x+w_comp]
            features = extract_features(symbol_binary)
            symbol_name = recognize_symbol(features)
            symbols.append((symbol_name, (x, y, w_comp, h_comp)))
    # Output recognized symbols
    for symbol_name, (x, y, w_comp, h_comp) in symbols:
        print(f"Recognized Symbol: {symbol_name} at position ({x}, {y})")

def visualize_recognition(warped, symbols):
    for symbol_name, (x, y, w_comp, h_comp) in symbols:
        cv2.rectangle(warped, (x, y), (x + w_comp, y + h_comp), (0, 255, 0), 2)
        cv2.putText(warped, symbol_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    plt.title("Recognized Symbols")
    plt.axis('off')
    plt.show()
