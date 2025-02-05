import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
import matplotlib
from rembg import remove
from skimage.morphology import skeletonize

matplotlib.use("tkagg")
os.environ["QT_QPA_PLATFORM"] = "offscreen"

def remove_background_rembg(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = remove(image)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def create_roi_mask(image, mask):
    roi = pcv.roi.rectangle(img=mask, x=0, y=0, w=image.shape[1], h=image.shape[0])
    return pcv.roi.filter(mask=mask, roi=roi, roi_type='partial')


def overlay_roi(image, mask):
    roi_image = image.copy()
    roi_image[mask != 0] = (0, 255, 0)
    cv2.rectangle(roi_image, (0, 0), (image.shape[1], image.shape[0]), (255, 0, 0), 5)
    return roi_image


def plot_color_histogram(image):
    """
    Generate color histogram data for multiple color channels.
    Returns histogram data and channel names for embedding in the global figure.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

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
        histograms[name] = hist / hist.sum() * 100

    return histograms


def display_transformations(images, titles, histograms):
    """
    Display transformations and the color histogram in a single integrated figure.
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (img, title) in enumerate(zip(images, titles)):
        if len(img.shape) == 2:
            axes[i].imshow(img, cmap="gray", extent=[0, 255, 255, 0])
        else:
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(title)
        axes[i].set_xticks([0, 50, 100, 150, 200, 250])
        axes[i].set_yticks([0, 50, 100, 150, 200, 250])
        axes[i].spines["top"].set_color("black")
        axes[i].spines["bottom"].set_color("black")
        axes[i].spines["left"].set_color("black")
        axes[i].spines["right"].set_color("black")

    histogram_ax = axes[len(images)]
    for name, hist in histograms.items():
        histogram_ax.plot(range(256), hist, label=name)
    histogram_ax.set_title("Color Histogram")
    histogram_ax.set_xlabel("Pixel Intensity (0-255)")
    histogram_ax.set_ylabel("Proportion of Pixels (%)")
    histogram_ax.legend(loc="upper right", title="Color Channels")

    for ax in axes[len(images) + 1:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    

def apply_transformations_to_image(image_path, save_dir=None):
    image, _, _ = pcv.readimage(filename=image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    transformed_images = []
    titles = []

    transformed_images.append(image)
    titles.append("Original")

    image_no_bg = remove_background_rembg(image)
    gray_img = pcv.rgb2gray(rgb_img=image_no_bg)
    binary_img = pcv.threshold.binary(gray_img=gray_img, threshold=128, object_type="light")
    gaussian_blur = pcv.gaussian_blur(img=binary_img, ksize=(3, 3))
    transformed_images.append(gaussian_blur)
    titles.append("Gaussian Blur")

    white_background = np.ones_like(image) * 255
    fg_mask = image_no_bg[:, :, 0] > 0
    image_on_white = np.where(fg_mask[:, :, None], image_no_bg, white_background)

    binary_mask = pcv.threshold.binary(gray_img=gray_img, threshold=128, object_type="light")
    masked_image = cv2.bitwise_and(image_on_white, image_on_white, mask=binary_mask)

    gray_threshold = 80
    background_mask = np.all(image_on_white > gray_threshold, axis=2)
    final_mask = np.logical_or(binary_mask == 255, background_mask)

    colored_mask = np.where(final_mask[:, :, None], white_background, image_on_white)
    transformed_images.append(colored_mask)
    titles.append("Mask")

    filtered_mask = create_roi_mask(image, binary_mask)
    roi_image = overlay_roi(image, filtered_mask)

    transformed_images.append(roi_image)
    titles.append("ROI Objects")

    analysis_image = pcv.analyze.size(img=image, labeled_mask=binary_mask)
    transformed_images.append(analysis_image)
    titles.append("Analyze objects")

    sift = cv2.SIFT_create()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints = sift.detect(gray_image, None)
    keypoints_img = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    transformed_images.append(cv2.cvtColor(keypoints_img, cv2.COLOR_BGR2RGB))
    titles.append("Pseudolandmarks")

    histograms = plot_color_histogram(image)

    if save_dir:
        for img, title in zip(transformed_images, titles):
            save_path = os.path.join(save_dir, f"{base_name}_{title.replace(' ', '_')}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 else img)
    else:
        display_transformations(transformed_images, titles, histograms)


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] == "-h":
        print("Usage: python transformation.py -src <input_path> [-dest <output_directory>]")
        print("Arguments:")
        print("  -src <input_path>      Path to an image or a directory containing images")
        print("  -dest <output_dir>     (Optional) Directory to save image transformations")
        sys.exit(0)

    input_path = sys.argv[2]
    output_dir = None

    if "-dest" in sys.argv:
        output_dir = sys.argv[sys.argv.index("-dest") + 1]
        os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(input_path):
        apply_transformations_to_image(input_path, output_dir)
    elif os.path.isdir(input_path):
        for file_name in os.listdir(input_path):
            file_path = os.path.join(input_path, file_name)
            if os.path.isfile(file_path) and file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                apply_transformations_to_image(file_path, output_dir)
    else:
        print(f"Error: Invalid input path {input_path}")
        sys.exit(1)