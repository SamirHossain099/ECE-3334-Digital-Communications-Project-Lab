import cv2
import numpy as np
import os

def preprocess_and_extract_symbols(image_path):
    # Step 1: Input Image Acquisition
    img_color = cv2.imread(image_path)
    if img_color is None:
        print("Error: Image not found or unable to read.")
        return None, None, None

    # Step 2: Preprocessing
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert image if necessary
    if np.mean(img_bin) > 127:
        img_bin = 255 - img_bin

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel)
    img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel)

    # Step 3: Card Orientation Detection
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Error: No contours found.")
        return None, None, None

    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    
    angle = rect[2]
    width, height = rect[1][0], rect[1][1]
    if width < height:
        angle = angle
    else:
        angle = angle + 90

    # Rotate image to correct orientation
    (h, w) = img_gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img_rotated = cv2.warpAffine(img_color, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Step 4: Card Cropping
    img_gray_rotated = cv2.cvtColor(img_rotated, cv2.COLOR_BGR2GRAY)
    _, img_bin_rotated = cv2.threshold(img_gray_rotated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(img_bin_rotated) > 127:
        img_bin_rotated = 255 - img_bin_rotated

    contours, _ = cv2.findContours(img_bin_rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Error: No contours found in rotated image.")
        return None, None, None

    card_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(card_contour)
    card_cropped = img_rotated[y:y+h, x:x+w]

    # Step 5: Extracting Rank and Suit Symbols
    symbol_region = card_cropped[0:int(h*0.3), 0:int(w*0.17)]
    symbol_gray = cv2.cvtColor(symbol_region, cv2.COLOR_BGR2GRAY)
    _, symbol_bin = cv2.threshold(symbol_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 6: Finding Contours in Symbol Region
    contours, _ = cv2.findContours(symbol_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 50
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    if len(filtered_contours) < 2:
        print("Error: Not enough contours found to separate rank and suit symbols.")
        return None, None, None

    # Calculate center and sort contours by size and distance from center
    symbol_center_x = symbol_bin.shape[1] // 2
    symbol_center_y = symbol_bin.shape[0] // 2

    def contour_priority_score(contour, center_x, center_y):
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return float('inf')
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        distance = ((cx - center_x) ** 2 + (cy - center_y) ** 2) ** 0.5
        area = cv2.contourArea(contour)
        return distance / (area + 1e-5)

    filtered_contours.sort(key=lambda cnt: contour_priority_score(cnt, symbol_center_x, symbol_center_y))
    rank_contour = filtered_contours[0]
    suit_contour = filtered_contours[1]

    bounding_boxes = [cv2.boundingRect(c) for c in [rank_contour, suit_contour]]
    sorted_contours = sorted(zip([rank_contour, suit_contour], bounding_boxes), key=lambda b: b[1][1])

    x, y, w, h = sorted_contours[0][1]
    rank_symbol = symbol_bin[y:y+h, x:x+w]
    rank_symbol = cv2.resize(rank_symbol, (70, 100))

    x, y, w, h = sorted_contours[1][1]
    suit_symbol = symbol_bin[y:y+h, x:x+w]
    suit_symbol = cv2.resize(suit_symbol, (70, 125))

    return rank_symbol, suit_symbol, card_cropped

def load_templates(template_folder):
    rank_templates = {}
    suit_templates = {}

    ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    for rank in ranks:
        template_path = os.path.join(template_folder, 'ranks', f'{rank}.jpg')
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is not None:
            rank_templates[rank] = template
        else:
            print(f"Warning: Template for rank '{rank}' not found.")

    suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
    for suit in suits:
        template_path = os.path.join(template_folder, 'suits', f'{suit}.jpg')
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is not None:
            suit_templates[suit] = template
        else:
            print(f"Warning: Template for suit '{suit}' not found.")

    return rank_templates, suit_templates

def identify_card_with_sift(rank_symbol, suit_symbol, rank_templates, suit_templates):
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    kp_rank, desc_rank = sift.detectAndCompute(rank_symbol, None)
    kp_suit, desc_suit = sift.detectAndCompute(suit_symbol, None)

    best_rank, max_rank_matches = None, 0
    for rank_name, rank_template in rank_templates.items():
        kp_template, desc_template = sift.detectAndCompute(rank_template, None)
        matches = bf.knnMatch(desc_rank, desc_template, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good_matches) > max_rank_matches:
            max_rank_matches = len(good_matches)
            best_rank = rank_name

    best_suit, max_suit_matches = None, 0
    for suit_name, suit_template in suit_templates.items():
        kp_template, desc_template = sift.detectAndCompute(suit_template, None)
        matches = bf.knnMatch(desc_suit, desc_template, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good_matches) > max_suit_matches:
            max_suit_matches = len(good_matches)
            best_suit = suit_name

    result = f"The card is {best_rank} of {best_suit}" if best_rank and best_suit else "Unable to confidently determine the card's rank and suit."
    print(result)
    print(f"Rank match score: {max_rank_matches}, Suit match score: {max_suit_matches}")
    return result

if __name__ == "__main__":
    image_path = 'e:/Laptop/Work/Study/Uni - TTU/6) Fall 24 - Sixth Semester/Fall 2024 TTU Image Processing (ECE-4367-001) Full Term/Projects/Project 4/test_images2/heart.jpeg'
    template_folder = 'e:/Laptop/Work/Study/Uni - TTU/6) Fall 24 - Sixth Semester/Fall 2024 TTU Image Processing (ECE-4367-001) Full Term/Projects/Project 4/imgs'

    rank_templates, suit_templates = load_templates(template_folder)
    rank_symbol, suit_symbol, card_cropped = preprocess_and_extract_symbols(image_path)

    if rank_symbol is not None and suit_symbol is not None:
        result = identify_card_with_sift(rank_symbol, suit_symbol, rank_templates, suit_templates)
        cv2.putText(card_cropped, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Final Result", card_cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Symbol extraction failed.")
