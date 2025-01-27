import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
import matplotlib

matplotlib.use("tkagg")
os.environ["QT_QPA_PLATFORM"] = "offscreen"


def apply_transformations(image_path):
    """
    Apply various transformations to an image using PlantCV, save the results, and display them in a grid.
    """
    # Load the image
    image, _, _ = pcv.readimage(filename=image_path)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    transformed_images = []
    titles = []

    ####################### ORIGINAL #######################
    transformed_images.append(image)
    titles.append("Original")

    ####################### BINARY THRESHOLD #######################
    gray_img = pcv.rgb2gray(rgb_img=image)
    binary_img = pcv.threshold.binary(gray_img=gray_img, threshold=128, object_type="light")
    inverted_img = pcv.invert(binary_img)
    transformed_images.append(inverted_img)
    titles.append("Binary Threshold")

    ####################### MASK #######################
    gray_img = pcv.rgb2gray(rgb_img=image)
    binary_mask = pcv.threshold.binary(gray_img=gray_img, threshold=128, object_type="light")
    masked_image = cv2.bitwise_and(image, image, mask=binary_mask)
    white_background = np.ones_like(image) * 255
    gray_threshold = 80
    background_mask = np.all(image > gray_threshold, axis=2)
    final_mask = np.logical_or(binary_mask == 255, background_mask)
    colored_mask = np.where(final_mask[:, :, None], white_background, image)
    transformed_images.append(colored_mask)
    titles.append("Mask")

    ####################### ROI #######################
    # roi_result = pcv.roi.rectangle(img=image, x=50, y=50, h=150, w=150)
    # roi_contour = roi_result.contours
    # roi_hierarchy = roi_result.hierarchy
    # kept_mask = pcv.apply_mask(img=image, mask=roi_contour, mask_color="black")
    # roi_render = image.copy()
    # if isinstance(roi_contour, list):
    #     cv2.drawContours(image=roi_render, contours=roi_contour, contourIdx=-1, color=(255, 0, 0), thickness=2)
    # else:
    #     cv2.drawContours(image=roi_render, contours=[roi_contour], contourIdx=-1, color=(255, 0, 0), thickness=2)

    # transformed_images.append(roi_render)
    # titles.append("ROI Objects")

    # ####################### CONTOURS #######################
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contour_img = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)
    # transformed_images.append(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    # titles.append("Analyze object")

    ####################### PSEUDOLANDMARKS #######################
    sift = cv2.SIFT_create()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints = sift.detect(gray_image, None)
    keypoints_img = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    transformed_images.append(cv2.cvtColor(keypoints_img, cv2.COLOR_BGR2RGB))
    titles.append("Pseudolandmarks")

    ####################### COLOR HISTOGRAM #######################
    histograms = plot_color_histogram(image)

    display_transformations(transformed_images, titles, histograms)


def plot_color_histogram(image):
    """
    Generate color histogram data for multiple color channels.
    Returns histogram data and channel names for embedding in the global figure.
    """
    # Convert image to additional color spaces
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    # Create a dictionary of channels to process
    channels = {
        "blue": image[:, :, 0],
        "green": image[:, :, 1],
        "red": image[:, :, 2],
        "hue": hsv[:, :, 0],
        "saturation": hsv[:, :, 1],
        "value": hsv[:, :, 2],
        "lightness": hls[:, :, 1],
    }

    histograms = {}
    for name, channel in channels.items():
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        histograms[name] = hist / hist.sum() * 100  # Normalize to percentage

    return histograms


def display_transformations(images, titles, histograms):
    """
    Display transformations and the color histogram in a single integrated figure.
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))  # 3x3 grid for images + histogram
    axes = axes.flatten()

    # Display transformed images
    for i, (img, title) in enumerate(zip(images, titles)):
        if len(img.shape) == 2:  # Grayscale image
            axes[i].imshow(img, cmap="gray")
        else:
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(title)
        axes[i].axis("off")

    # Display histogram in the last slot
    histogram_ax = axes[len(images)]  # Use the next available slot
    for name, hist in histograms.items():
        histogram_ax.plot(range(256), hist, label=name)
    histogram_ax.set_title("Color Histogram")
    histogram_ax.set_xlabel("Pixel Intensity (0-255)")
    histogram_ax.set_ylabel("Proportion of Pixels (%)")
    histogram_ax.legend(loc="upper right", title="Color Channels")

    # Hide any unused subplot axes
    for ax in axes[len(images) + 1:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python transformation.py -src <input_path>")
        sys.exit(1)

    input_path = sys.argv[2]

    if os.path.isfile(input_path):
        apply_transformations(input_path)
    else:
        print(f"Error: Invalid input path {input_path}")
        sys.exit(1)