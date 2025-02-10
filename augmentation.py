import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def count_images_in_folders(input_directory, valid_extensions=None):
    if valid_extensions is None:
        valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

    folder_counts = {}

    for root, _, files in os.walk(input_directory):
        relative_path = os.path.relpath(root, input_directory)
        parts = relative_path.split(os.sep)

        if len(parts) >= 1:
            image_count = sum(
                1 for f in files
                if os.path.splitext(f)[1].strip().lower() in valid_extensions
            )
            if image_count > 0:
                folder_counts[root] = image_count

    return folder_counts


def balance_dataset(input_directory):
    folder_counts = count_images_in_folders(input_directory)
    if not folder_counts:
        print(
            "🚨 Aucun dossier avec des images trouvé ! Vérifiez votre dataset.")
        return
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
                    augment(img_path, save_dir=folder)
                    num_to_add -= 6
                    if num_to_add <= 0:
                        break

    print("✅ Dataset équilibré avec succès !")


def augment(image_path, save_dir=None):
    """
    Augmente une image.
    - Si `save_dir` est None, affiche les augmentations.
    - Sinon, sauvegarde les augmentations dans `save_dir`.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image {image_path}")
        return

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    augmented_images = []
    augmented_titles = []

    def process_and_save(img, suffix):
        file_name = f"{base_name}_{suffix}.jpg"
        if save_dir:
            save_path = os.path.join(save_dir, file_name)
            cv2.imwrite(save_path, img)
        augmented_images.append(img)
        augmented_titles.append(suffix)

    if save_dir is not None:
        process_and_save(image, "Original")
    process_and_save(cv2.flip(image, 1), "Flip")
    process_and_save(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), "Rotate")
    process_and_save(cv2.convertScaleAbs(image, alpha=2, beta=0), "Contrast")

    height, width = image.shape[:2]

    pts1 = np.float32([[0, 0], [width, 0], [0, height]])
    pts2 = np.float32(
        [[0, 0], [width * 0.8, height * 0.2], [width * 0.2, height]])
    matrix = cv2.getAffineTransform(pts1, pts2)
    process_and_save(cv2.warpAffine(image, matrix, (width, height)), "Skew")

    M = np.array([[1, 0.2, 0], [0.2, 1, 0]], dtype=float)
    process_and_save(cv2.warpAffine(image, M, (width, height)), "Shear")

    process_and_save(cv2.GaussianBlur(image, (7, 7), 0), "Blur")

    if save_dir is None:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        for i, (img, title) in enumerate(zip(
                augmented_images, augmented_titles)):
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[i].set_title(title)
            axes[i].axis("off")
        plt.tight_layout()
        plt.show()


def augment_dataset(input_directory, output_directory):
    """
    Augmente un dataset complet et enregistre
    les nouvelles images dans `output_directory`.
    """
    if not os.path.exists(input_directory):
        raise FileNotFoundError(
            f"Error: The input directory '{input_directory}' does not exist.")

    balance_dataset(input_directory)

    for root, _, files in os.walk(input_directory):
        for img_file in files:
            img_path = os.path.join(root, img_file)
            if img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                relative_path = os.path.relpath(root, input_directory)
                output_subdir = os.path.join(output_directory, relative_path)
                os.makedirs(output_subdir, exist_ok=True)
                augment(img_path, save_dir=output_subdir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: You must provide an input image or dataset.")
        print("Usage: python augmentation.py <input_path> [output_directory]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    if os.path.isfile(input_path):
        augment(input_path, output_dir)
    elif os.path.isdir(input_path):
        if not output_dir:
            print(
                "Error: You must specify an output\
                directory when processing a dataset.")
            sys.exit(1)
        augment_dataset(input_path, output_dir)
    else:
        print(f"Error: Invalid input path {input_path}")
        sys.exit(1)
