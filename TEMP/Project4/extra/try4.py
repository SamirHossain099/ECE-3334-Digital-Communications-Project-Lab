import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Assuming these utility modules are available.
# If not, minimal implementations are provided within the code.

# from processing.ColorHelper import ColorHelper
# from utils.Loader import Loader
# from utils.MathHelper import MathHelper

# Minimal implementations of the utility modules

class ColorHelper:
    @staticmethod
    def gray2bin(img_gray):
        _, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        return img_bin

    @staticmethod
    def reverse(img_bin):
        return cv2.bitwise_not(img_bin)

class Loader:
    @staticmethod
    def load_ranks(path):
        ranks = []
        for filename in os.listdir(path):
            if filename.endswith('.png') or filename.endswith('.jpg'):
                rank_img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
                rank_name = os.path.splitext(filename)[0]
                rank_img = cv2.resize(rank_img, (70, 125))
                rank_img = cv2.threshold(rank_img, 127, 255, cv2.THRESH_BINARY_INV)[1]
                ranks.append(Template(rank_img, rank_name))
        return ranks

    @staticmethod
    def load_suits(path):
        suits = []
        for filename in os.listdir(path):
            if filename.endswith('.png') or filename.endswith('.jpg'):
                suit_img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
                suit_name = os.path.splitext(filename)[0]
                suit_img = cv2.resize(suit_img, (70, 100))
                suit_img = cv2.threshold(suit_img, 127, 255, cv2.THRESH_BINARY_INV)[1]
                suits.append(Template(suit_img, suit_name))
        return suits

class MathHelper:
    @staticmethod
    def length(x1, y1, x2, y2):
        return np.hypot(x2 - x1, y2 - y1)

class Template:
    def __init__(self, img, name):
        self.img = img
        self.name = name

# Functions as provided by the user

def get_thresh(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 42, 89)
    kernel = np.ones((2, 2))
    dial = cv2.dilate(canny, kernel=kernel, iterations=2)
    return dial

def find_corners_set(img, original, draw=False):
    # find the set of contours on the threshed image
    contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # sort them by highest area
    proper = sorted(contours, key=cv2.contourArea, reverse=True)

    four_corners_set = []

    for cnt in proper:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, closed=True)

        # only select those contours with a good area
        if area > 10000:
            # find out the number of corners
            approx = cv2.approxPolyDP(cnt, 0.01 * perimeter, closed=True)
            num_corners = len(approx)

            if num_corners == 4:
                # create bounding box around shape
                x, y, w, h = cv2.boundingRect(approx)

                if draw:
                    cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 3)

                # make sure the points are ordered correctly
                pts = [pt[0] for pt in approx]
                pts = order_points(pts)

                four_corners_set.append(pts)

                if draw:
                    for a in approx:
                        cv2.circle(original, (a[0][0], a[0][1]), 6, (255, 0, 0), 3)

    return four_corners_set

def order_points(pts):
    # Order points in a consistent order: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    pts = np.array(pts)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]      # Top-left
    rect[2] = pts[np.argmax(s)]      # Bottom-right
    rect[1] = pts[np.argmin(diff)]   # Top-right
    rect[3] = pts[np.argmax(diff)]   # Bottom-left

    return rect

def find_flatten_cards(img, set_of_corners, debug=False):
    width, height = 200, 300
    img_outputs = []

    for i, corners in enumerate(set_of_corners):
        top_left = corners[0]
        top_right = corners[1]
        bottom_right = corners[2]
        bottom_left = corners[3]

        # get the 4 corners of the card
        pts1 = np.float32([top_left, top_right, bottom_right, bottom_left])
        # now define which corner we are referring to
        pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

        # transformation matrix
        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        img_output = cv2.warpPerspective(img, matrix, (width, height))

        img_outputs.append(img_output)

    return img_outputs

def get_corner_snip(flattened_images: list):
    corner_images = []
    for img in flattened_images:
        # Crop the image to where the corner might be
        # vertical, horizontal
        crop_color = img[5:110, 1:38]  # Color image
        crop_gray = cv2.cvtColor(crop_color, cv2.COLOR_BGR2GRAY)

        # Resize by a factor of 4
        crop_gray = cv2.resize(crop_gray, None, fx=4, fy=4)
        crop_color = cv2.resize(crop_color, None, fx=4, fy=4)

        # Threshold the corner
        bin_img = ColorHelper.gray2bin(crop_gray)
        bilateral = cv2.bilateralFilter(bin_img, 11, 17, 17)
        canny = cv2.Canny(bilateral, 30, 200)
        kernel = np.ones((1, 1))
        result = cv2.dilate(canny, kernel=kernel, iterations=1)

        # Append the thresholded image and the original color image
        corner_images.append([result, crop_color])
    return corner_images

def split_rank_suit(img, original, debug=False) -> list:
    """
    :param debug: display opencv or not
    :param img:
    :param original: original color image
    :return: list of image, index 0: rank, index 1: suit
    """
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts_sort = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    cnts_sort = sorted(cnts_sort, key=lambda x: cv2.boundingRect(x)[1])

    if debug:
        cv2.drawContours(img, cnts_sort, -1, (0, 255, 0), 1)

    ranksuit = []

    _rank = None

    for i, cnt in enumerate(cnts_sort):
        x, y, w, h = cv2.boundingRect(cnt)
        x2, y2 = x + w, y + h

        crop = original[y:y2, x:x2]

        if i == 0:  # rank: 70, 125
            crop = cv2.resize(crop, (70, 125), interpolation=cv2.INTER_AREA)
            _rank = crop
        else:  # suit: 70, 100
            crop = cv2.resize(crop, (70, 100), interpolation=cv2.INTER_AREA)
            if debug and _rank is not None:
                r = cv2.resize(_rank, (70, 100), interpolation=cv2.INTER_AREA)
                s = cv2.resize(crop, (70, 100), interpolation=cv2.INTER_AREA)
                h_concat = np.concatenate((r, s), axis=1)
                h_concat = cv2.resize(h_concat, (250, 200), interpolation=cv2.INTER_AREA)
                cv2.imshow("crop2", h_concat)

        # Now, convert crop to grayscale
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop_bin = ColorHelper.gray2bin(crop_gray)
        crop_bin = ColorHelper.reverse(crop_bin)
        ranksuit.append(crop_bin)

    return ranksuit

def template_matching(rank, suit, train_ranks, train_suits, show_plt=False) -> tuple[str, str]:
    """Finds best rank and suit matches for the query card by comparing the extracted rank and suit images
    with the training images using template matching."""
    best_rank_match_diff = float('inf')
    best_suit_match_diff = float('inf')
    best_rank_match_name = "Unknown"
    best_suit_match_name = "Unknown"

    for train_rank in train_ranks:
        diff_img = cv2.absdiff(rank, train_rank.img)
        rank_diff = int(np.sum(diff_img) / 255)

        if rank_diff < best_rank_match_diff:
            best_rank_match_diff = rank_diff
            best_rank_match_name = train_rank.name

    for train_suit in train_suits:
        diff_img = cv2.absdiff(suit, train_suit.img)
        suit_diff = int(np.sum(diff_img) / 255)

        if suit_diff < best_suit_match_diff:
            best_suit_match_diff = suit_diff
            best_suit_match_name = train_suit.name

    # Thresholds to determine if a match is acceptable (adjust as needed)
    RANK_DIFF_MAX = 5000
    SUIT_DIFF_MAX = 3000

    if best_rank_match_diff > RANK_DIFF_MAX:
        best_rank_match_name = "Unknown"

    if best_suit_match_diff > SUIT_DIFF_MAX:
        best_suit_match_name = "Unknown"

    return best_rank_match_name, best_suit_match_name

def show_text(predictions: list, four_corners_set, img):
    for i, prediction in enumerate(predictions):
        # figure out where to place the text
        corners = np.array(four_corners_set[i])
        corners_flat = corners.reshape(-1, corners.shape[-1])
        start_x = int(corners_flat[0][0]) + 0
        half_y = int(corners_flat[0][1]) - 10

        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(img, prediction, (start_x, half_y), font, 0.8, (50, 205, 50), 2, cv2.LINE_AA)

# Main function

def main():
    # Load the image
    image_path = r'e:\Laptop\Work\Study\Uni - TTU\6) Fall 24 - Sixth Semester\Fall 2024 TTU Image Processing (ECE-4367-001) Full Term\Projects\Project 4\I1.tif'
    img = cv2.imread(image_path)  # Replace with your image path
    if img is None:
        print("Image not found.")
        return

    imgResult = img.copy()
    imgResult2 = img.copy()

    rankspath =  r'e:\Laptop\Work\Study\Uni - TTU\6) Fall 24 - Sixth Semester\Fall 2024 TTU Image Processing (ECE-4367-001) Full Term\Projects\Project 4\imgs\ranks'
    suitspath =  r'e:\Laptop\Work\Study\Uni - TTU\6) Fall 24 - Sixth Semester\Fall 2024 TTU Image Processing (ECE-4367-001) Full Term\Projects\Project 4\imgs\suits'

    # Load templates
    train_ranks = Loader.load_ranks(rankspath)  # Path to rank templates
    train_suits = Loader.load_suits(suitspath)  # Path to suit templates

    # Process the image
    thresh = get_thresh(img)
    four_corners_set = find_corners_set(thresh, imgResult, draw=False)
    flatten_card_set = find_flatten_cards(imgResult2, four_corners_set)
    cropped_images = get_corner_snip(flatten_card_set)

    ranksuit_list = []

    for i, (img_card, original_color) in enumerate(cropped_images):
        drawable = img_card.copy()
        original_copy = original_color.copy()

        # Split rank and suit
        ranksuit = split_rank_suit(drawable, original_copy, debug=False)
        if len(ranksuit) == 2:
            ranksuit_list.append(ranksuit)
        else:
            print(f"Could not extract rank and suit for card {i+1}")

    predictions = []

    for _rank, _suit in ranksuit_list:
        predict_rank, predict_suit = template_matching(_rank, _suit, train_ranks, train_suits)
        prediction = f"{predict_rank} of {predict_suit}"
        predictions.append(prediction)
        print(prediction)

    # Display results
    show_text(predictions, four_corners_set, imgResult)

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(imgResult, cv2.COLOR_BGR2RGB))
    plt.title("Recognized Cards")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
