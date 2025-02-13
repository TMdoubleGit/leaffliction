import os
import sys
import cv2
import numpy as np
import random
import shutil
import argparse
import matplotlib.pyplot as plt


def count_images_in_folders(input_directory, valid_extensions=None):
    """
    Count the number of images in each disease subdirectory of the dataset.
    Returns a dictionary { folder_name : image_count }.
    """
    if valid_extensions is None:
        valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

    folder_counts = {}

    for root, _, files in os.walk(input_directory):
        category = os.path.basename(root)
        if root != input_directory and len(files) > 0:
            image_count = sum(
                1 for f in files
                if os.path.splitext(f)[1].strip().lower() in valid_extensions)
            if image_count > 0:
                folder_counts[category] = image_count

    return folder_counts


def copy_original_images(input_directory, output_directory):
    """
    Copies all original images into a flat structure,
    keeping only the disease category subdirectories.
    """
    for root, _, files in os.walk(input_directory):
        category = os.path.basename(root)
        relative_path = os.path.relpath(root, input_directory)
        if os.path.dirname(relative_path) == "":
            continue
        if root == input_directory:
            continue

        output_subdir = os.path.join(output_directory, category)
        os.makedirs(output_subdir, exist_ok=True)

        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src = os.path.join(root, file)
                dst = os.path.join(output_subdir,
                                   f"{os.path.splitext(file)[0]}_Original.jpg")
                shutil.copy(src, dst)


def augment(image_path, max_images, save_dir, num_to_generate):
    """
    Augments an image and saves it to save_dir, ensuring that the folder
    does not exceed max_images.
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
            np.float32([[0, 0], [image.shape[1] * 0.8,
                        image.shape[0] * 0.2],
                        [image.shape[1] * 0.2, image.shape[0]]])
        ), (image.shape[1], image.shape[0]))),
        ("Shear", cv2.warpAffine(image, np.array([[1, 0.2, 0], [0.2, 1, 0]],
                                                 dtype=float),
                                                (image.shape[1],
                                                 image.shape[0]))),
        ("Blur", cv2.GaussianBlur(image, (7, 7), 0))
    ]

    num_generated = 0

    if save_dir is None:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()

        for ax, (title, augmented_img) in zip(axes, augmentations):
            ax.imshow(cv2.cvtColor(augmented_img, cv2.COLOR_BGR2RGB))
            ax.set_title(title)
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    else:
        existing_files = set(os.listdir(save_dir))
        for suffix, augmented_img in augmentations:
            if len(existing_files) >= max_images:
                num_generated += 1
                break
            file_name = f"{base_name}_{suffix}.jpg"
            save_path = os.path.join(save_dir, file_name)
            if file_name not in existing_files:
                cv2.imwrite(save_path, augmented_img)
                existing_files.add(file_name)
                num_generated += 1

    return num_generated


def augment_dataset(input_directory, output_directory=None):
    """
    Augments an entire dataset and ensures all
    disease subdirectories reach the same number of images.
    Handles re-augmentation if needed.
    """
    if not os.path.exists(input_directory):
        raise FileNotFoundError(
                f"Error: The input directory\
                 '{input_directory}' does not exist.")

    print("📂 Copying original images to augmented dataset...")
    copy_original_images(input_directory, output_directory)

    folder_counts = count_images_in_folders(input_directory)

    if not folder_counts:
        print("🚨 No folder with images found! Please check your dataset.")
        return

    max_images = max(folder_counts.values())
    print(f"📊 Target size: {max_images} images per folder.")

    for root, _, files in os.walk(input_directory):
        category = os.path.basename(root)
        if root == input_directory or category not in folder_counts:
            continue

        output_subdir = os.path.join(output_directory, category)
        os.makedirs(output_subdir, exist_ok=True)

        if folder_counts[category] >= max_images:
            print(f"✅ {category} already has\
             {folder_counts[category]} images. No augmentation needed.")
            continue

        to_add = max_images - folder_counts[category]
        print(f"📢 Augmenting {category}" +
              f"Adding {to_add} images to reach {max_images}")

        images = [os.path.join(output_subdir, f) for f
                  in os.listdir(output_subdir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if to_add > 6 * len(images):
            print(f"🔄 Re-augmenting in {category} \
                   because it needs {to_add} images" +
                  f" but has only {len(images)} originals.")

        generated_images = 0
        while generated_images < to_add:
            img_path = random.choice(images)
            if to_add > 6 * len(images):
                images = [os.path.join(output_subdir, f)
                          for f in os.listdir(output_subdir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            generated_images += augment(
                                        img_path, max_images,
                                        save_dir=output_subdir,
                                        num_to_generate=to_add-generated_images
                                        )

    print("✅ Dataset successfully balanced!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply image augmentations.")
    parser.add_argument(
        "-src", required=True, help="Path to an image or directory.")
    parser.add_argument(
        "-dest", help="Optional directory to save images.")

    args = parser.parse_args()
    input_path = args.src
    if os.path.isdir(input_path) and args.dest: 
        output_dir = args.dest
    else:
        output_dir = None

    os.makedirs(output_dir, exist_ok=True) if output_dir else None

    if os.path.isdir(input_path):
        if not output_dir:
            print("Error: Specify a destination directory with -dest.")
            sys.exit(1)
        augment_dataset(input_path, output_dir)
    elif os.path.isfile(input_path):
        augment(input_path, max_images=6, save_dir=None, num_to_generate=6)

    else:
        print(f"Error: Invalid input path {input_path}")
        sys.exit(1)
