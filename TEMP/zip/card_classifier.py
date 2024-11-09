import cv2
import numpy as np
import os
import sys
from tkinter import Tk, Button, Label
from tkinter.filedialog import askopenfilename

def preprocess_and_extract_symbols(image, use_one_in_ten=True, debug=False):
    # Step 1: Input Image Acquisition
    img_color = image.copy()
    if img_color is None:
        print("Error: Image not found or unable to read.")
        return None, None, None

    if debug:
        cv2.imshow("Original Image", img_color)
        cv2.waitKey(0)

    # Step 2: Preprocessing
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    if debug:
        cv2.imshow("Grayscale Image", img_gray)
        cv2.waitKey(0)

    _, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if debug:
        cv2.imshow("Binarized Image", img_bin)
        cv2.waitKey(0)

    # Invert image if necessary
    if np.mean(img_bin) > 127:
        img_bin = 255 - img_bin
        if debug:
            cv2.imshow("Inverted Binarized Image", img_bin)
            cv2.waitKey(0)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel)
    img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel)
    if debug:
        cv2.imshow("After Morphological Operations", img_bin)
        cv2.waitKey(0)

    # Step 3: Card Orientation Detection
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = img_color.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
    if debug:
        cv2.imshow("Contours Detected", img_contours)
        cv2.waitKey(0)

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
    (h, w) = img_gray.shape[:2]
    center = rect[0]
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img_rotated = cv2.warpAffine(img_color, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    if debug:
        cv2.imshow("Rotated Image", img_rotated)
        cv2.waitKey(0)

    # Step 4: Card Cropping
    img_gray_rotated = cv2.cvtColor(img_rotated, cv2.COLOR_BGR2GRAY)
    _, img_bin_rotated = cv2.threshold(img_gray_rotated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(img_bin_rotated) > 127:
        img_bin_rotated = 255 - img_bin_rotated

    contours, _ = cv2.findContours(img_bin_rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_rotated_contours = img_rotated.copy()
    cv2.drawContours(img_rotated_contours, contours, -1, (0, 255, 0), 2)
    if debug:
        cv2.imshow("Contours on Rotated Image", img_rotated_contours)
        cv2.waitKey(0)

    if not contours:
        print("Error: No contours found in rotated image.")
        return None, None, None

    card_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(card_contour)
    card_cropped = img_rotated[y:y+h, x:x+w]
    if debug:
        cv2.imshow("Cropped Card", card_cropped)
        cv2.waitKey(0)

    # Step 5: Extracting Rank and Suit Symbols
    # Assuming symbols are in the top-left corner
    symbol_region = card_cropped[0:int(h*0.3), 0:int(w*0.17)]
    if debug:
        cv2.imshow("Symbol Region", symbol_region)
        cv2.waitKey(0)

    symbol_gray = cv2.cvtColor(symbol_region, cv2.COLOR_BGR2GRAY)
    _, symbol_bin = cv2.threshold(symbol_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if debug:
        cv2.imshow("Binarized Symbol Region", symbol_bin)
        cv2.waitKey(0)

    # Step 6: Finding Contours in Symbol Region
    contours, _ = cv2.findContours(symbol_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours based on area
    min_contour_area = 50  # Adjust based on experimentation
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Filter out contours near the edges
    def is_contour_near_edge(contour, image_shape, margin_fraction=0.05):
        x, y, w, h = cv2.boundingRect(contour)
        img_h, img_w = image_shape[:2]
        margin_x = int(img_w * margin_fraction)
        margin_y = int(img_h * margin_fraction)
        if x < margin_x or y < margin_y or x + w > img_w - margin_x or y + h > img_h - margin_y:
            return True
        else:
            return False

    filtered_contours = []
    for cnt in filtered_contours:
        if not is_contour_near_edge(cnt, symbol_bin.shape):
            filtered_contours.append(cnt)

    if debug:
        symbol_with_contours = cv2.cvtColor(symbol_bin, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(symbol_with_contours, filtered_contours, -1, (0, 255, 0), 2)
        cv2.imshow("Filtered Contours in Symbol Region", symbol_with_contours)
        cv2.waitKey(0)

    if len(filtered_contours) == 0:
        print("Error: No valid contours found after filtering.")
        return None, None, None

    # Sort contours by vertical position (y-coordinate)
    bounding_boxes = [cv2.boundingRect(c) for c in filtered_contours]
    sorted_contours = sorted(zip(filtered_contours, bounding_boxes), key=lambda b: b[1][1])

    # Identify rank and suit symbols
    # Assuming the suit symbol is the lowest contour
    suit_contour = sorted_contours[-1][0]
    suit_bbox = sorted_contours[-1][1]

    # The rest are rank symbols (could be more than one contour, e.g., '1' and '0' in '10')
    rank_contours = [c[0] for c in sorted_contours[:-1]]
    rank_bboxes = [c[1] for c in sorted_contours[:-1]]

    # For '10', there will be two rank contours; for other cards, usually one
    # Option to choose between '1' and '0' for the rank symbol
    if len(rank_contours) == 0:
        print("Error: No rank contours found.")
        return None, None, None

    if len(rank_contours) == 1:
        # Only one rank contour
        rank_contour = rank_contours[0]
        rank_bbox = rank_bboxes[0]
    else:
        # Multiple rank contours (e.g., '1' and '0'), Choose between '1' and '0' based on horizontal position
        # '1' is on the left, '0' on the right
        if use_one_in_ten:
            # Choose the leftmost contour (assumed to be '1')
            rank_idx = np.argmin([bbox[0] for bbox in rank_bboxes])
        else:
            # Choose the rightmost contour (assumed to be '0')
            rank_idx = np.argmax([bbox[0] for bbox in rank_bboxes])
        rank_contour = rank_contours[rank_idx]
        rank_bbox = rank_bboxes[rank_idx]

    if debug:
        # Draw selected rank and suit contours for visualization
        cv2.drawContours(symbol_with_contours, [rank_contour], -1, (0, 0, 255), 2)  # Red
        cv2.drawContours(symbol_with_contours, [suit_contour], -1, (255, 0, 0), 2)  # Blue
        cv2.imshow("Selected Rank and Suit Contours", symbol_with_contours)
        cv2.waitKey(0)

    # Extract rank symbol
    x, y, w, h = rank_bbox
    rank_symbol = symbol_bin[y:y+h, x:x+w]

    # Extract suit symbol
    x, y, w, h = suit_bbox
    suit_symbol = symbol_bin[y:y+h, x:x+w]

    # Resize symbols while maintaining aspect ratio and padding
    def resize_and_pad(img, size=(70, 125)):
        h, w = img.shape
        aspect_ratio = w / h
        target_w, target_h = size

        # Compute scaling factor to fit the symbol into the target size
        scaling_factor = min(target_w / w, target_h / h)
        new_w = int(w * scaling_factor)
        new_h = int(h * scaling_factor)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create a new image with the target size and place the resized symbol at the center
        padded_img = np.zeros((target_h, target_w), dtype=np.uint8)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        padded_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img

        return padded_img

    rank_symbol_resized = resize_and_pad(rank_symbol)
    suit_symbol_resized = resize_and_pad(suit_symbol)

    if debug:
        # Display extracted symbols
        cv2.imshow("Rank Symbol", rank_symbol_resized)
        cv2.waitKey(0)
        cv2.imshow("Suit Symbol", suit_symbol_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return rank_symbol_resized, suit_symbol_resized, card_cropped

def match_templates(rank_symbol, suit_symbol, rank_templates, suit_templates):
    # Template Matching for Ranks
    max_score_rank = -np.inf
    best_match_rank = None
    for rank_name, rank_template in rank_templates.items():
        res = cv2.matchTemplate(rank_symbol, rank_template, cv2.TM_CCOEFF_NORMED)
        score = res[0][0]
        # print(f"Rank {rank_name} score: {score}")
        if score > max_score_rank:
            max_score_rank = score
            best_match_rank = rank_name

    # Template Matching for Suits
    max_score_suit = -np.inf
    best_match_suit = None
    for suit_name, suit_template in suit_templates.items():
        res = cv2.matchTemplate(suit_symbol, suit_template, cv2.TM_CCOEFF_NORMED)
        score = res[0][0]
        # print(f"Suit {suit_name} score: {score}")
        if score > max_score_suit:
            max_score_suit = score
            best_match_suit = suit_name

    # Thresholds
    threshold_rank = 0.5  # Adjust based on experimentation
    threshold_suit = 0.5  # Adjust based on experimentation

    # Result Declaration
    if max_score_rank > threshold_rank and max_score_suit > threshold_suit:
        result = f"{best_match_rank} of {best_match_suit}"
    else:
        result = "Unable to confidently determine the card's rank and suit."

    print(f"Rank match score: {max_score_rank}, Suit match score: {max_score_suit}")
    print(f"Result: {result}")
    return result

# Helper function to load templates
def load_templates():
    rank_templates = {}
    suit_templates = {}

    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_folder = os.path.join(script_dir, 'imgs')  # Folder containing 'ranks' and 'suits' subfolders

    # Load rank templates
    ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    for rank in ranks:
        template_path = os.path.join(template_folder, 'ranks', f'{rank}.png')
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
        template_path = os.path.join(template_folder, 'suits', f'{suit}.png')
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is not None:
            _, template_bin = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            template_resized = cv2.resize(template_bin, (70, 125))
            suit_templates[suit] = template_resized
        else:
            print(f"Warning: Template for suit '{suit}' not found.")

    return rank_templates, suit_templates

def main():
    # Load templates
    rank_templates, suit_templates = load_templates()

    # Create the main window
    root = Tk()
    root.title("Card Recognition")

    # Set window size and position
    window_width = 300
    window_height = 150

    # Get the screen dimension
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Find the center point
    center_x = int(screen_width/2 - window_width / 2)
    center_y = int(screen_height/2 - window_height / 2)

    # Set the position of the window to the center of the screen
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

    # Disable resizing the window
    root.resizable(False, False)

    # Create a label
    label = Label(root, text="Select input method:", font=("Arial", 14))
    label.pack(pady=10)

    # Define functions for buttons
    def use_webcam():
        root.destroy()
        # Use webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Press 's' to capture the card when it is centered in the video feed.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image from webcam.")
                break

            # Display the video feed
            cv2.imshow("Webcam Feed", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                # Process the frame
                # Enable debug mode to display images after each step
                rank_symbol, suit_symbol, card_cropped = preprocess_and_extract_symbols(frame, debug=True)
                if rank_symbol is not None and suit_symbol is not None:
                    result = match_templates(rank_symbol, suit_symbol, rank_templates, suit_templates)
                    # Display the final cropped card with result
                    cv2.putText(card_cropped, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.imshow("Final Result", card_cropped)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    print("Symbol extraction failed.")
                break
            elif key == ord('q'):
                # Exit if 'q' is pressed
                print("Exiting...")
                break

        cap.release()
        cv2.destroyAllWindows()

    def select_image_file():
        root.destroy()
        # Select image file
        root_file = Tk()
        root_file.withdraw()  # Hide the main window
        root_file.attributes('-topmost', True)  # Bring the dialog to the front
        print("Please select an image file...")
        image_path = askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
        root_file.destroy()

        if image_path:
            image = cv2.imread(image_path)
            if image is not None:
                # Optionally enable debug mode for image file processing
                rank_symbol, suit_symbol, card_cropped = preprocess_and_extract_symbols(image, debug=False)
                if rank_symbol is not None and suit_symbol is not None:
                    result = match_templates(rank_symbol, suit_symbol, rank_templates, suit_templates)
                    # Display the final cropped card with result
                    cv2.putText(card_cropped, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.imshow("Final Result", card_cropped)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    print("Symbol extraction failed.")
            else:
                print("Error: Could not read the selected image.")
        else:
            print("No file selected.")

    # Create buttons
    button_webcam = Button(root, text="Use Webcam", width=15, command=use_webcam)
    button_webcam.pack(pady=5)

    button_select_file = Button(root, text="Select Image File", width=15, command=select_image_file)
    button_select_file.pack(pady=5)

    # Start the GUI event loop
    root.mainloop()

if __name__ == "__main__":
    main()