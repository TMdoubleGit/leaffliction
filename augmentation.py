import os
import sys
import shutil
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


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


def augment_image(image_path):
    """
    Apply various augmentations to a single image, save the results, and display them using PIL.
    :param image_path: Path to the input image.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image {image_path}")
        return

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    dir_name = os.path.dirname(image_path)

    augmented_files = []

    ####################### FLIP #######################
    flipped = cv2.flip(image, 1)
    flip_path = os.path.join(dir_name, f"{base_name}_Flip.jpg")
    cv2.imwrite(flip_path, flipped)
    augmented_files.append(flip_path)
    ####################################################

    ####################### ROTATE #######################
    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    rotate_path = os.path.join(dir_name, f"{base_name}_Rotate.jpg")
    cv2.imwrite(rotate_path, rotated)
    augmented_files.append(rotate_path)
    ######################################################

    ####################### CROP #######################
    height, width = image.shape[:2]
    crop = image[height // 4:3 * height // 4, width // 4:3 * width // 4]
    crop_path = os.path.join(dir_name, f"{base_name}_Crop.jpg")
    cv2.imwrite(crop_path, crop)
    augmented_files.append(crop_path)
    ####################################################

    ####################### SKEW #######################
    pts1 = np.float32([[0, 0], [width, 0], [0, height]])
    pts2 = np.float32([[0, 0], [width * 0.8, height * 0.2], [width * 0.2, height]])
    matrix = cv2.getAffineTransform(pts1, pts2)
    skewed = cv2.warpAffine(image, matrix, (width, height))
    skew_path = os.path.join(dir_name, f"{base_name}_Skew.jpg")
    cv2.imwrite(skew_path, skewed)
    augmented_files.append(skew_path)
    ####################################################

    ####################### SHEAR #######################
    M = np.array([[1, 0.2, 0], [0.2, 1, 0]], dtype=float)
    shear = cv2.warpAffine(image, M, (width, height))
    shear_path = os.path.join(dir_name, f"{base_name}_Shear.jpg")
    cv2.imwrite(shear_path, shear)
    augmented_files.append(shear_path)
    #####################################################

    ####################### BLUR #######################
    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    blur_path = os.path.join(dir_name, f"{base_name}_Blur.jpg")
    cv2.imwrite(blur_path, blurred)
    augmented_files.append(blur_path)
    ####################################################

    ##mute pour creer le augmented_directory
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i, file_path in enumerate(augmented_files):
        pil_image = Image.open(file_path)
        axes[i].imshow(pil_image)
        axes[i].set_title(os.path.basename(file_path))
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: You must provide the input image.")
        print("Usage: python augmentation.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        augment_image(image_path) ## ligne pour la correction
        # augment_dataset(input_dir, augmented_root) ## ligne pour creer le augmented_directory
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)