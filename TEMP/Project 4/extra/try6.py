import cv2
import numpy as np
import os

def preprocess_image(image_path, size):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read the image. Check the path and try again.")
    
    # Resize the image to the specified size for better matching
    img = cv2.resize(img, size)

    # Apply edge detection to get the boundaries
    edges = cv2.Canny(img, 50, 150)

    # Dilate the edges to make them thicker and more connected
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Invert the image to prepare for flood filling
    filled = cv2.bitwise_not(dilated_edges)

    # Flood fill the background from a point outside the shape
    h, w = filled.shape
    mask = np.zeros((h+2, w+2), np.uint8)  # Create a mask for flood fill
    cv2.floodFill(filled, mask, (0, 0), 255)  # Flood fill from the top-left corner

    # Invert back to get the filled shape
    filled = cv2.bitwise_not(filled)

    # Display the processed image
    cv2.imshow(f"Processed Image - {os.path.basename(image_path)}", filled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return filled

def template_match(image, template_folder, template_size):
    best_match_name = None
    best_match_score = -1

    # Iterate over all template images in the specified folder
    for template_name in os.listdir(template_folder):
        template_path = os.path.join(template_folder, template_name)
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

        if template is not None:
            # Resize template to match the specified size
            template = cv2.resize(template, template_size)

            # Apply the same preprocessing steps to the template
            template_edges = cv2.Canny(template, 50, 150)
            template_dilated_edges = cv2.dilate(template_edges, np.ones((3, 3), np.uint8), iterations=1)
            template_filled = cv2.bitwise_not(template_dilated_edges)
            mask = np.zeros((template_filled.shape[0] + 2, template_filled.shape[1] + 2), np.uint8)
            cv2.floodFill(template_filled, mask, (0, 0), 255)
            template_filled = cv2.bitwise_not(template_filled)

            # Perform template matching
            res = cv2.matchTemplate(image, template_filled, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # Print matching score for each template
            print(f"Matching score for {template_name}: {max_val}")

            # Check if the current match is the best one so far
            if max_val > best_match_score:
                best_match_score = max_val
                best_match_name = os.path.splitext(template_name)[0]
        else:
            print(f"Warning: Could not read template {template_path}")

    return best_match_name, best_match_score

def identify_card(rank_image_path, suit_image_path, ranks_folder, suits_folder):
    # Preprocess the rank and suit images separately with specified sizes
    rank_image = preprocess_image(rank_image_path, (70, 125))
    suit_image = preprocess_image(suit_image_path, (70, 100))

    # Match rank and suit separately
    print("\nRank Matching Scores:")
    rank_name, rank_score = template_match(rank_image, ranks_folder, (70, 125))
    print("\nSuit Matching Scores:")
    suit_name, suit_score = template_match(suit_image, suits_folder, (70, 100))

    return {
        "Rank": rank_name,
        "Rank Score": rank_score,
        "Suit": suit_name,
        "Suit Score": suit_score
    }

# Example usage
# Provide the paths to the rank and suit images, and template folders
rank_image_path = "e:/Laptop/Work/Study/Uni - TTU/6) Fall 24 - Sixth Semester/Fall 2024 TTU Image Processing (ECE-4367-001) Full Term/Projects/Project 4/extra/I12.jpg"  # Change this to your actual rank image path
suit_image_path = "e:/Laptop/Work/Study/Uni - TTU/6) Fall 24 - Sixth Semester/Fall 2024 TTU Image Processing (ECE-4367-001) Full Term/Projects/Project 4/extra/I11.jpg"  # Change this to your actual suit image path
ranks_folder = "e:/Laptop/Work/Study/Uni - TTU/6) Fall 24 - Sixth Semester/Fall 2024 TTU Image Processing (ECE-4367-001) Full Term/Projects/Project 4/imgs/ranks"
suits_folder = "e:/Laptop/Work/Study/Uni - TTU/6) Fall 24 - Sixth Semester/Fall 2024 TTU Image Processing (ECE-4367-001) Full Term/Projects/Project 4/imgs/suits"

# Call the identify_card function
try:
    result = identify_card(rank_image_path, suit_image_path, ranks_folder, suits_folder)
    print("\nDetected Rank:", result["Rank"])
    print("Detected Suit:", result["Suit"])
    print("Rank Match Score:", result["Rank Score"])
    print("Suit Match Score:", result["Suit Score"])
except ValueError as e:
    print(e)
