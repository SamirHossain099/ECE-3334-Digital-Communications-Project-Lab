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
    # Find contours in the edged image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Assume the largest contour is the card
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

def order_points(pts):
    # Order points in a consistent order: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]      # Top-left
    rect[2] = pts[np.argmax(s)]      # Bottom-right
    rect[1] = pts[np.argmin(diff)]   # Top-right
    rect[3] = pts[np.argmax(diff)]   # Bottom-left

    return rect

def get_warped_image(image, contour):
    # Apply perspective transformation to get a top-down view of the card
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

def extract_corner_regions(warped):
    # Define the fraction of the card to crop
    corner_frac_width = 0.18  # Adjusted based on symbol size
    corner_frac_height = 0.18

    h, w = warped.shape[:2]
    corner_width = int(w * corner_frac_width)
    corner_height = int(h * corner_frac_height)

    # Extract top-left corner
    corner_roi = warped[0:corner_height, 0:corner_width]

    return corner_roi

def preprocess_roi(roi):
    # Convert to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh_roi = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    processed_roi = cv2.morphologyEx(thresh_roi, cv2.MORPH_OPEN, kernel, iterations=1)
    processed_roi = cv2.morphologyEx(processed_roi, cv2.MORPH_CLOSE, kernel, iterations=1)

    return processed_roi

def extract_symbols_from_roi(processed_roi):
    # Find contours in the ROI
    contours, hierarchy = cv2.findContours(processed_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    symbols = []
    h, w = processed_roi.shape

    for cnt in contours:
        x, y, w_comp, h_comp = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        # Define size criteria (adjust thresholds as needed)
        if area > 50 and h_comp < h * 0.9 and w_comp < w * 0.9:
            # Ensure the contour is within the ROI bounds
            symbol = processed_roi[y:y+h_comp, x:x+w_comp]
            symbols.append({'image': symbol, 'contour': cnt})

    # Sort symbols from top to bottom
    symbols = sorted(symbols, key=lambda s: s['contour'][0][0][1])

    # Assume the top symbol is the rank, and the next is the suit
    return symbols

def analyze_contour_properties(contour, symbol_image):
    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h

    # Contour area
    area = cv2.contourArea(contour)

    # Bounding rectangle area
    rect_area = w * h

    # Extent
    extent = float(area) / rect_area if rect_area != 0 else 0

    # Convex hull and solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area != 0 else 0

    # Equivalent diameter
    equi_diameter = np.sqrt(4 * area / np.pi) if area != 0 else 0

    # Orientation
    angle = 0
    if len(contour) >= 5:
        _, _, angle = cv2.fitEllipse(contour)

    # Number of convexity defects
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    if hull_indices is not None and len(hull_indices) > 3:
        defects = cv2.convexityDefects(contour, hull_indices)
        num_defects = defects.shape[0] if defects is not None else 0
    else:
        num_defects = 0

    # Moments
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()

    features = {
        'aspect_ratio': aspect_ratio,
        'extent': extent,
        'solidity': solidity,
        'equi_diameter': equi_diameter,
        'orientation': angle,
        'num_defects': num_defects,
        'hu_moments': hu_moments,
        'area': area
    }

    return features

def recognize_rank(features):
    aspect_ratio = features['aspect_ratio']
    solidity = features['solidity']
    num_defects = features['num_defects']
    area = features['area']

    # Implement rules based on contour properties
    if aspect_ratio < 0.5 and solidity > 0.5:
        if num_defects == 0:
            return 'A'  # Ace
        elif num_defects == 1:
            return '4'  # Possibly 4
        elif num_defects > 1:
            return '8'  # Possibly 8
    elif 0.5 <= aspect_ratio <= 0.8:
        if area > 1000:
            return 'K'  # King
        else:
            return '7'  # Possibly 7
    elif aspect_ratio > 0.8:
        return 'J'  # Jack or 10
    # Add more rules based on observed features
    return 'Unknown Rank'

def recognize_suit(features):
    aspect_ratio = features['aspect_ratio']
    solidity = features['solidity']
    num_defects = features['num_defects']

    # Implement rules based on contour properties
    if solidity > 0.9:
        if num_defects == 0:
            return 'Hearts'  # Solid shape, no defects
        elif num_defects >= 1:
            return 'Diamonds'  # May have slight defects
    elif 0.5 < solidity <= 0.9:
        if num_defects >= 2:
            return 'Spades'  # More complex shape
        elif num_defects == 1:
            return 'Clubs'  # Has one defect due to stem
    # Add more rules based on observed features
    return 'Unknown Suit'

def process_image(image_path):
    # Step 1: Load and preprocess image
    image, edges = load_and_preprocess_image(image_path)

    # Step 2: Card detection and perspective correction
    card_contour = find_card_contour(edges)
    if card_contour is None:
        print("Card contour not found.")
        return None

    warped = get_warped_image(image, card_contour)
    if warped is None:
        print("Warped image could not be obtained.")
        return None

    # Step 3: Extract corner regions
    corner_roi = extract_corner_regions(warped)

    # Step 4: Preprocess the ROI
    processed_roi = preprocess_roi(corner_roi)

    # Step 5: Extract symbols from ROI
    symbols = extract_symbols_from_roi(processed_roi)
    if len(symbols) < 2:
        print("Not enough symbols found in ROI.")
        return None

    # Step 6: Contour analysis and recognition
    rank_symbol_data = symbols[0]
    suit_symbol_data = symbols[1]

    rank_features = analyze_contour_properties(rank_symbol_data['contour'], rank_symbol_data['image'])
    suit_features = analyze_contour_properties(suit_symbol_data['contour'], suit_symbol_data['image'])

    recognized_rank = recognize_rank(rank_features)
    recognized_suit = recognize_suit(suit_features)

    return warped, recognized_rank, recognized_suit

def visualize_recognition(warped, recognized_rank, recognized_suit):
    # Annotate the warped image with recognized rank and suit
    cv2.putText(warped, f'Rank: {recognized_rank}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(warped, f'Suit: {recognized_suit}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    plt.title("Recognized Rank and Suit")
    plt.axis('off')
    plt.show()

def main():
    image_path = r'e:\Laptop\Work\Study\Uni - TTU\6) Fall 24 - Sixth Semester\Fall 2024 TTU Image Processing (ECE-4367-001) Full Term\Projects\Project 4\I6.jpg'
    result = process_image(image_path)
    if result is None:
        print("Failed to process image.")
        return

    warped, recognized_rank, recognized_suit = result

    print(f"Recognized Rank: {recognized_rank}")
    print(f"Recognized Suit: {recognized_suit}")

    visualize_recognition(warped, recognized_rank, recognized_suit)

if __name__ == "__main__":
    main()
