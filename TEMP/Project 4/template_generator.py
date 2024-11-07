import cv2
import numpy as np
import os

def preprocess_and_extract_symbols(image_path, output_folder="e:/Laptop/Work/Study/Uni - TTU/6) Fall 24 - Sixth Semester/Fall 2024 TTU Image Processing (ECE-4367-001) Full Term/Projects/Project 4/imgs2/"):
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Step 1: Input Image Acquisition
    img_color = cv2.imread(image_path)
    if img_color is None:
        print("Error: Image not found or unable to read.")
        return None, None, None

    # Save Original Image
    cv2.imwrite(os.path.join(output_folder, "original_image.jpg"), img_color)

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

    # Save Binarized Image
    cv2.imwrite(os.path.join(output_folder, "binarized_image.jpg"), img_bin)

    # Step 3: Card Orientation Detection
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Error: No contours found.")
        return None, None

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

    # Save Rotated Image
    cv2.imwrite(os.path.join(output_folder, "rotated_image.jpg"), img_rotated)

    # Step 4: Card Cropping
    img_gray_rotated = cv2.cvtColor(img_rotated, cv2.COLOR_BGR2GRAY)
    _, img_bin_rotated = cv2.threshold(img_gray_rotated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(img_bin_rotated) > 127:
        img_bin_rotated = 255 - img_bin_rotated

    contours, _ = cv2.findContours(img_bin_rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Error: No contours found in rotated image.")
        return None, None

    card_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(card_contour)
    card_cropped = img_rotated[y:y+h, x:x+w]

    # Save Cropped Card
    cv2.imwrite(os.path.join(output_folder, "cropped_card.jpg"), card_cropped)

    # Step 5: Extracting Rank and Suit Symbols
    symbol_region = card_cropped[0:int(h*0.3), 0:int(w*0.17)]

    # Save Symbol Region
    cv2.imwrite(os.path.join(output_folder, "symbol_region.jpg"), symbol_region)

    symbol_gray = cv2.cvtColor(symbol_region, cv2.COLOR_BGR2GRAY)
    _, symbol_bin = cv2.threshold(symbol_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Save Binarized Symbol
    cv2.imwrite(os.path.join(output_folder, "binarized_symbol.jpg"), symbol_bin)

    # Step 6: Finding Contours in Symbol Region
    contours, _ = cv2.findContours(symbol_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours based on area
    min_contour_area = 50
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    if len(filtered_contours) < 2:
        print("Error: Not enough contours found to separate rank and suit symbols.")
        return None, None, None

    # Create a copy of the symbol region for visualization
    symbol_with_contours = cv2.cvtColor(symbol_bin, cv2.COLOR_GRAY2BGR)

    # Draw all filtered contours in green
    for cnt in filtered_contours:
        cv2.drawContours(symbol_with_contours, [cnt], -1, (0, 255, 0), 2)

    # Save Filtered Contours
    cv2.imwrite(os.path.join(output_folder, "filtered_contours.jpg"), symbol_with_contours)

    # Calculate the center of the symbol region
    symbol_center_x = symbol_bin.shape[1] // 2
    symbol_center_y = symbol_bin.shape[0] // 2

    # Sort contours by combined size and distance metric
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

    # Draw the selected rank and suit contours in red and blue for verification
    cv2.drawContours(symbol_with_contours, [rank_contour], -1, (0, 0, 255), 2)
    cv2.drawContours(symbol_with_contours, [suit_contour], -1, (255, 0, 0), 2)

    # Save Selected Rank and Suit Contours
    cv2.imwrite(os.path.join(output_folder, "selected_rank_suit_contours.jpg"), symbol_with_contours)

    # Sort contours by vertical position (y-coordinate)
    bounding_boxes = [cv2.boundingRect(c) for c in [rank_contour, suit_contour]]
    sorted_contours = sorted(zip([rank_contour, suit_contour], bounding_boxes), key=lambda b: b[1][1])

    # Step 7: Extracting Rank and Suit Symbols
    x, y, w, h = sorted_contours[0][1]
    rank_symbol = symbol_bin[y:y+h, x:x+w]
    rank_symbol = cv2.resize(rank_symbol, (70, 100))
    cv2.imwrite(os.path.join(output_folder, "rank_symbol.jpg"), rank_symbol)

    x, y, w, h = sorted_contours[1][1]
    suit_symbol = symbol_bin[y:y+h, x:x+w]
    suit_symbol = cv2.resize(suit_symbol, (70, 125))
    cv2.imwrite(os.path.join(output_folder, "suit_symbol.jpg"), suit_symbol)

    return rank_symbol, suit_symbol, card_cropped

def match_templates(rank_symbol, suit_symbol, rank_templates, suit_templates):
    # Template Matching for Ranks
    max_score_rank = -np.inf
    best_match_rank = None
    for rank_name, rank_template in rank_templates.items():
        res = cv2.matchTemplate(rank_symbol, rank_template, cv2.TM_CCOEFF_NORMED)
        score = res[0][0]
        print(score)
        if score > max_score_rank:
            max_score_rank = score
            best_match_rank = rank_name

    # Template Matching for Suits
    max_score_suit = -np.inf
    best_match_suit = None
    for suit_name, suit_template in suit_templates.items():
        res = cv2.matchTemplate(suit_symbol, suit_template, cv2.TM_CCOEFF_NORMED)
        score = res[0][0]
        print(score)
        if score > max_score_suit:
            max_score_suit = score
            best_match_suit = suit_name

    # Thresholds
    threshold_rank = 0  # Adjust based on experimentation
    threshold_suit = 0  # Adjust based on experimentation

    # Result Declaration
    if max_score_rank > threshold_rank and max_score_suit > threshold_suit:
        result = f"The card is {best_match_rank} of {best_match_suit}"
    else:
        result = "Unable to confidently determine the card's rank and suit."

    print(f"Rank match score: {max_score_rank}, Suit match score: {max_score_suit}")
    print(result)
    return result

# Helper function to load templates
def load_templates(template_folder):
    rank_templates = {}
    suit_templates = {}

    # Load rank templates
    ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    for rank in ranks:
        template_path = os.path.join(template_folder, 'ranks', f'{rank}.jpg')
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is not None:
            _, template_bin = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            template_resized = cv2.resize(template_bin, (70, 125))
            rank_templates[rank] = template_resized
        else:
            print(f"Warning: Template for rank '{rank}' not found.")

    # Load suit templates
    suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
    for suit in suits:
        template_path = os.path.join(template_folder, 'suits', f'{suit}.jpg')
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is not None:
            _, template_bin = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            template_resized = cv2.resize(template_bin, (70, 125))
            suit_templates[suit] = template_resized
        else:
            print(f"Warning: Template for suit '{suit}' not found.")

    return rank_templates, suit_templates

#A, 10, diamond got 9 wrong, hearts got 8 wrong, 
# Wrong - 6, 8, 9
# Main code to run the functions
if __name__ == "__main__":
    image_path = 'e:/Laptop/Work/Study/Uni - TTU/6) Fall 24 - Sixth Semester/Fall 2024 TTU Image Processing (ECE-4367-001) Full Term/Projects/Project 4/test_images2/spade.jpeg'
    template_folder = 'e:/Laptop/Work/Study/Uni - TTU/6) Fall 24 - Sixth Semester/Fall 2024 TTU Image Processing (ECE-4367-001) Full Term/Projects/Project 4/imgs'  # Folder containing 'ranks' and 'suits' subfolders

    # Load templates
    rank_templates, suit_templates = load_templates(template_folder)

    # Preprocess image and extract symbols
    rank_symbol, suit_symbol, card_cropped = preprocess_and_extract_symbols(image_path)

    if rank_symbol is not None and suit_symbol is not None:
        # Match templates and declare result
        result = match_templates(rank_symbol, suit_symbol, rank_templates, suit_templates)

        # Display the final cropped card with result
        cv2.putText(card_cropped, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Final Result", card_cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Symbol extraction failed.")
