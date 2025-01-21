import os
import sys
import shutil
import cv2
import numpy as np

def augment_image(image_path, output_directory):
    """
    Apply various augmentations to a single image and save the results.
    :param image_path: Path to the input image.
    :param output_directory: Directory to save the augmented images.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image {image_path}")
        return

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    flipped = cv2.flip(image, 1)
    cv2.imwrite(os.path.join(output_directory, f"{base_name}_Flip.jpg"), flipped)

    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(os.path.join(output_directory, f"{base_name}_Rotate.jpg"), rotated)

    height, width = image.shape[:2]
    crop = image[height // 4:3 * height // 4, width // 4:3 * width // 4]
    cv2.imwrite(os.path.join(output_directory, f"{base_name}_Crop.jpg"), crop)

    pts1 = np.float32([[0, 0], [width, 0], [0, height]])
    pts2 = np.float32([[0, 0], [width * 0.8, height * 0.2], [width * 0.2, height]])
    matrix = cv2.getAffineTransform(pts1, pts2)
    skewed = cv2.warpAffine(image, matrix, (width, height))
    cv2.imwrite(os.path.join(output_directory, f"{base_name}_Skew.jpg"), skewed)

    M = np.array([[1, 0.2, 0], [0.2, 1, 0]], dtype=float)
    shear = cv2.warpAffine(image, M, (width, height))
    cv2.imwrite(os.path.join(output_directory, f"{base_name}_Shear.jpg"), shear)

    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    cv2.imwrite(os.path.join(output_directory, f"{base_name}_Blur.jpg"), blurred)

def augment_dataset(input_directory, augmented_root):
    """
    Apply augmentations to all images in a directory and replicate the directory structure in augmented_root.
    :param input_directory: Root directory containing the original dataset.
    :param augmented_root: Root directory to store the augmented dataset.
    """
    if not os.path.exists(input_directory):
        raise FileNotFoundError(f"Error: The input directory '{input_directory}' does not exist.")

    for root, dirs, files in os.walk(input_directory):
        relative_path = os.path.relpath(root, input_directory)
        augmented_directory = os.path.join(augmented_root, relative_path)
        if not os.path.exists(augmented_directory):
            os.makedirs(augmented_directory)

        for filename in files:
            image_path = os.path.join(root, filename)
            if os.path.isfile(image_path):
                augment_image(image_path, root)

                for img_file in os.listdir(root):
                    src = os.path.join(root, img_file)
                    dst = os.path.join(augmented_directory, img_file)
                    if os.path.isfile(src) and not os.path.exists(dst):
                        shutil.copy(src, dst)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Error: You must provide the input directory and the root of the augmented directory.")
        print("Usage: python augmentation.py <input_directory> <augmented_directory>")
        sys.exit(1)

    input_dir = sys.argv[1]
    augmented_root = sys.argv[2]

    try:
        augment_dataset(input_dir, augmented_root)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)