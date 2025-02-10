import os
import sys
import shutil
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def count_images_in_folders(input_directory, valid_extensions=None):
    if valid_extensions is None:
        valid_extensions = {'.jpg', '.jpeg', '.png'}

    folder_counts = {}

    for root, _, files in os.walk(input_directory):
        relative_path = os.path.relpath(root, input_directory)
        parts = relative_path.split(os.sep)

        if len(parts) > 1:
            image_count = sum(
                1 for f in files
                if os.path.splitext(f)[1].lower() in valid_extensions
            )
            if image_count > 0:
                folder_counts[root] = image_count

    return folder_counts


def balance_dataset(input_directory):
    folder_counts = count_images_in_folders(input_directory)
    print(folder_counts)
    max_images = max(folder_counts.values())

    for folder, count in folder_counts.items():
        if count < max_images:
            images = [os.path.join(folder, f) for f in os.listdir(folder)]
            num_to_add = max_images - count

            print(
                f"📢 Augmenting {folder}: Adding {num_to_add} images "
                f"to reach {max_images}"
            )

            while num_to_add > 0:
                for img_path in images:
                    augment_image(img_path)
                    num_to_add -= 6
                    if num_to_add <= 0:
                        break

    print("✅ Dataset équilibré avec succès !")


def augment_dataset(
    input_directory="./dataset", augmented_root="augmented_dir"
):
    if not os.path.exists(input_directory):
        raise FileNotFoundError(
            f"Error: The input directory '{input_directory}' does not exist."
        )

    balance_dataset(input_directory)

    for root, _, files in os.walk(input_directory):
        relative_path = os.path.relpath(root, input_directory)
        parts = relative_path.split(os.sep)

        if len(parts) > 1:
            relative_path = os.path.join(*parts[1:])
            augmented_directory = os.path.join(augmented_root, relative_path)

            os.makedirs(augmented_directory, exist_ok=True)

            for img_file in files:
                src = os.path.join(root, img_file)
                dst = os.path.join(augmented_directory, img_file)
                if os.path.isfile(src) and not os.path.exists(dst):
                    shutil.copy(src, dst)


def augment_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image {image_path}")
        return

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    dir_name = os.path.dirname(image_path)

    augmented_files = []

    flipped = cv2.flip(image, 1)
    flip_path = os.path.join(dir_name, f"{base_name}_Flip.jpg")
    cv2.imwrite(flip_path, flipped)
    augmented_files.append(flip_path)

    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    rotate_path = os.path.join(dir_name, f"{base_name}_Rotate.jpg")
    cv2.imwrite(rotate_path, rotated)
    augmented_files.append(rotate_path)

    contrast = cv2.convertScaleAbs(image, alpha=2, beta=0)
    contrast_path = os.path.join(dir_name, f"{base_name}_Contrast.jpg")
    cv2.imwrite(contrast_path, contrast)
    augmented_files.append(contrast_path)

    height, width = image.shape[:2]

    pts1 = np.float32([[0, 0], [width, 0], [0, height]])
    pts2 = np.float32(
        [[0, 0], [width * 0.8, height * 0.2], [width * 0.2, height]]
    )
    matrix = cv2.getAffineTransform(pts1, pts2)
    skewed = cv2.warpAffine(image, matrix, (width, height))
    skew_path = os.path.join(dir_name, f"{base_name}_Skew.jpg")
    cv2.imwrite(skew_path, skewed)
    augmented_files.append(skew_path)

    M = np.array([[1, 0.2, 0], [0.2, 1, 0]], dtype=float)
    shear = cv2.warpAffine(image, M, (width, height))
    shear_path = os.path.join(dir_name, f"{base_name}_Shear.jpg")
    cv2.imwrite(shear_path, shear)
    augmented_files.append(shear_path)

    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    blur_path = os.path.join(dir_name, f"{base_name}_Blur.jpg")
    cv2.imwrite(blur_path, blurred)
    augmented_files.append(blur_path)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i, file_path in enumerate(augmented_files):
        pil_image = Image.open(file_path)
        axes[i].imshow(pil_image)
        axes[i].set_title(os.path.basename(file_path))
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: You must provide the input image.")
        print("Usage: python augmentation.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        augment_image(image_path)
        # augment_dataset()  # Ligne pour créer le augmented_directory
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
