import numpy as np
import cv2
from scipy.stats import ks_2samp

def histogram_matching_from_images(image1_path, image2_path, image3_path):
    """
    Matches the histogram of the mismatched image to the reference histograms of the other two.
    
    Parameters:
        image1_path (str): File path for the first image.
        image2_path (str): File path for the second image.
        image3_path (str): File path for the third (potentially mismatched) image.
    
    Returns:
        np.ndarray: The remapped image (mismatched image adjusted to match the reference histograms).
    """
    # Load the images
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    image3 = cv2.imread(image3_path, cv2.IMREAD_GRAYSCALE)

    # Check if images were loaded correctly
    if image1 is None or image2 is None or image3 is None:
        raise ValueError("One or more images could not be loaded. Check the file paths.")

    # Calculate histograms (256 bins for 8-bit grayscale)
    hist1, _ = np.histogram(image1.ravel(), bins=256, range=(0, 256), density=True)
    hist2, _ = np.histogram(image2.ravel(), bins=256, range=(0, 256), density=True)
    hist3, _ = np.histogram(image3.ravel(), bins=256, range=(0, 256), density=True)

    # Normalize histograms and calculate cumulative distribution functions (CDFs)
    cdf1 = np.cumsum(hist1)
    cdf2 = np.cumsum(hist2)
    cdf3 = np.cumsum(hist3)
    
    # Identify the mismatched histogram using the Kolmogorov-Smirnov test
    ks1 = ks_2samp(hist1, hist3).statistic
    ks2 = ks_2samp(hist2, hist3).statistic
    mismatched_cdf = cdf3 if ks1 > 0.05 or ks2 > 0.05 else None

    if mismatched_cdf is None:
        print("Histograms are already similar. No adjustment needed.")
        return image3

    # Create the reference CDF as the average of the two similar histograms
    reference_cdf = (cdf1 + cdf2) / 2

    # Compute the mapping function
    mapping = np.interp(mismatched_cdf, reference_cdf, np.arange(256))

    # Apply the mapping to the mismatched image
    remapped_image = cv2.LUT(image3, mapping.astype(np.uint8))

    return remapped_image

# Example usage
if __name__ == "__main__":
    remapped = histogram_matching_from_images("temp_image_processing/Worm1.png", "temp_image_processing/Worm3.png", "temp_image_processing/Worm2.png")
    cv2.imwrite("remapped_image.png", remapped)
    print("Remapped image saved as 'remapped_image.png'.")


