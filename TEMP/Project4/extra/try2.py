import os
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
    corner_frac_width = 0.2  # 20% of the width
    corner_frac_height = 0.2  # 20% of the height

    h, w = warped.shape[:2]
    corner_width = int(w * corner_frac_width)
    corner_height = int(h * corner_frac_height)

    # Extract top-left corner
    rank_roi = warped[0:corner_height, 0:corner_width]

    # Optionally, extract bottom-right corner (if needed)
    # suit_roi = warped[h - corner_height:h, w - corner_width:w]

    return rank_roi

def preprocess_roi(roi):
    # Convert to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh_roi = cv2.threshold(gray_roi, 150, 255, cv2.THRESH_BINARY_INV)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
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
        if area > 50:
            symbol = processed_roi[y:y+h_comp, x:x+w_comp]
            symbols.append(symbol)

    # Sort symbols from top to bottom
    symbols = sorted(symbols, key=lambda s: s.shape[0], reverse=True)

    # Assume the first symbol is rank, second is suit (adjust if necessary)
    return symbols

def extract_features(symbol):
    # Resize symbol to a standard size
    symbol = cv2.resize(symbol, (70, 125))

    # Compute Hu Moments
    moments = cv2.moments(symbol)
    hu_moments = cv2.HuMoments(moments).flatten()

    # Compute aspect ratio
    x, y, w_comp, h_comp = cv2.boundingRect(symbol)
    aspect_ratio = w_comp / float(h_comp)

    # Count number of holes
    contours, hierarchy = cv2.findContours(symbol, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    num_holes = 0
    if hierarchy is not None:
        for idx, cnt in enumerate(contours):
            if hierarchy[0][idx][3] != -1:
                num_holes += 1

    # Compute solidity
    area = cv2.contourArea(contours[0])
    hull = cv2.convexHull(contours[0])
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area != 0 else 0

    features = {
        'hu_moments': hu_moments,
        'aspect_ratio': aspect_ratio,
        'num_holes': num_holes,
        'solidity': solidity,
    }

    return features

def build_reference_hu_moments(symbols_folder):
    hu_moments_dict = {}
    for filename in os.listdir(symbols_folder):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            symbol_name = os.path.splitext(filename)[0]
            filepath = os.path.join(symbols_folder, filename)
            symbol_image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if symbol_image is None:
                continue
            # Preprocess the symbol image similarly
            _, thresh_symbol = cv2.threshold(symbol_image, 150, 255, cv2.THRESH_BINARY_INV)
            # Resize to standard size
            symbol_resized = cv2.resize(thresh_symbol, (70, 125))
            # Compute Hu Moments
            moments = cv2.moments(symbol_resized)
            hu_moments = cv2.HuMoments(moments).flatten()
            hu_moments_dict[symbol_name] = hu_moments
    return hu_moments_dict

def recognize_rank(features, rank_hu_moments):
    extracted_hu = features['hu_moments']
    min_score = float('inf')
    recognized_rank = 'Unknown Rank'
    for rank_name, hu_moments in rank_hu_moments.items():
        score = np.sum(np.abs(extracted_hu - hu_moments))
        if score < min_score:
            min_score = score
            recognized_rank = rank_name
    return recognized_rank

def recognize_suit(features, suit_hu_moments):
    extracted_hu = features['hu_moments']
    min_score = float('inf')
    recognized_suit = 'Unknown Suit'
    for suit_name, hu_moments in suit_hu_moments.items():
        score = np.sum(np.abs(extracted_hu - hu_moments))
        if score < min_score:
            min_score = score
            recognized_suit = suit_name
    return recognized_suit

def process_image(image_path, rank_hu_moments, suit_hu_moments):
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

    # Step 6: Feature extraction and recognition
    rank_symbol = symbols[0]
    suit_symbol = symbols[1]

    # Display extracted symbols
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(rank_symbol, cmap='gray')
    plt.title('Extracted Rank Symbol')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(suit_symbol, cmap='gray')
    plt.title('Extracted Suit Symbol')
    plt.axis('off')

    plt.show()

    rank_features = extract_features(rank_symbol)
    suit_features = extract_features(suit_symbol)

    recognized_rank = recognize_rank(rank_features, rank_hu_moments)
    recognized_suit = recognize_suit(suit_features, suit_hu_moments)

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
    image_path = "path_to_your_image.jpg"  # Replace with your image path

    # Build reference Hu Moments
    rank_hu_moments = build_reference_hu_moments('rank_symbols')  # Folder containing rank symbol images
    suit_hu_moments = build_reference_hu_moments('suit_symbols')  # Folder containing suit symbol images

    result = process_image(image_path, rank_hu_moments, suit_hu_moments)
    if result is None:
        print("Failed to process image.")
        return

    warped, recognized_rank, recognized_suit = result

    print(f"Recognized Rank: {recognized_rank}")
    print(f"Recognized Suit: {recognized_suit}")

    visualize_recognition(warped, recognized_rank, recognized_suit)

if __name__ == "__main__":
    main()