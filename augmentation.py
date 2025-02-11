import os
import sys
import cv2
import numpy as np
import random
import shutil
import matplotlib.pyplot as plt


def count_images_in_folders(input_directory, valid_extensions=None):
    """
    Count the number of images in each subdirectory of the input directory.
    Returns a dictionary { folder_path : image_count }.
    """
    if valid_extensions is None:
        valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

    folder_counts = {}

    for root, _, files in os.walk(input_directory):
        image_count = sum(
            1 for f in files if os.path.splitext(f)[1].strip().lower() in valid_extensions
        )
        if image_count > 0:
            folder_counts[root] = image_count

    return folder_counts


def copy_original_images(input_directory, output_directory):
    """
    Copies all original images from the input directory to the output directory,
    preserving the folder structure.
    """
    for root, _, files in os.walk(input_directory):
        relative_path = os.path.relpath(root, input_directory)
        output_subdir = os.path.join(output_directory, relative_path)
        os.makedirs(output_subdir, exist_ok=True)

        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src = os.path.join(root, file)
                dst = os.path.join(output_subdir, f"{os.path.splitext(file)[0]}_Original.jpg")
                shutil.copy(src, dst)


def augment(image_path, save_dir, max_images):
    """
    Augments an image and saves it to `save_dir`, ensuring that the folder
    does not exceed `max_images`.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image {image_path}")
        return 0

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    augmentations = [
        ("Flip", cv2.flip(image, 1)),
        ("Rotate", cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)),
        ("Contrast", cv2.convertScaleAbs(image, alpha=2, beta=0)),
        ("Skew", cv2.warpAffine(image, cv2.getAffineTransform(
            np.float32([[0, 0], [image.shape[1], 0], [0, image.shape[0]]]),
            np.float32([[0, 0], [image.shape[1] * 0.8, image.shape[0] * 0.2], [image.shape[1] * 0.2, image.shape[0]]])
        ), (image.shape[1], image.shape[0]))),
        ("Shear", cv2.warpAffine(image, np.array([[1, 0.2, 0], [0.2, 1, 0]], dtype=float), (image.shape[1], image.shape[0]))),
        ("Blur", cv2.GaussianBlur(image, (7, 7), 0))
    ]

    existing_files = set(os.listdir(save_dir))
    num_generated = 0

    for suffix, augmented_img in augmentations:
        if len(existing_files) >= max_images:
            break  # Stop if max limit is reached

        file_name = f"{base_name}_{suffix}.jpg"
        save_path = os.path.join(save_dir, file_name)

        if file_name not in existing_files:
            cv2.imwrite(save_path, augmented_img)
            existing_files.add(file_name)
            num_generated += 1

    return num_generated


def augment_dataset(input_directory, output_directory):
    """
    Augments an entire dataset and ensures all subdirectories reach the same number of images.
    """
    if not os.path.exists(input_directory):
        raise FileNotFoundError(f"Error: The input directory '{input_directory}' does not exist.")

    print("📂 Copying original images to augmented dataset...")
    copy_original_images(input_directory, output_directory)

    folder_counts = count_images_in_folders(input_directory)

    if not folder_counts:
        print("🚨 No folder with images found! Please check your dataset.")
        return

    max_images = max(folder_counts.values())
    print(f"📊 Target size: {max_images} images per folder.")

    for folder, count in folder_counts.items():
        relative_path = os.path.relpath(folder, input_directory)
        output_subdir = os.path.join(output_directory, relative_path)
        os.makedirs(output_subdir, exist_ok=True)

        if count >= max_images:
            print(f"✅ {folder} already has {count} images. No augmentation needed.")
            continue

        num_to_add = max_images - count
        print(f"📢 Augmenting {folder}: Adding {num_to_add} images to reach {max_images}")

        images = [os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        generated_images = 0
        while generated_images < num_to_add:
            img_path = random.choice(images)
            generated_images += augment(img_path, save_dir=output_subdir, max_images=max_images)

    print("✅ Dataset successfully balanced!")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Error: You must provide both an input and output directory.")
        print("Usage: python augmentation.py <input_directory> <output_directory>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]

    if os.path.isdir(input_path):
        augment_dataset(input_path, output_dir)
    else:
        print(f"Error: Invalid input path {input_path}")
        sys.exit(1)
