import os
import shutil
import random


def create_small_dataset(original_dataset: str,
                         small_dataset: str,
                         total_images: int = 100,
                         seed: int = None):
    if seed is not None:
        random.seed(seed)

    if not os.path.exists(small_dataset):
        os.makedirs(small_dataset, exist_ok=True)

    all_images = []
    for root, _, files in os.walk(original_dataset):
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                all_images.append(os.path.join(root, file))

    selected_images = random.sample(all_images,
                                    min(total_images, len(all_images)))

    for img_path in selected_images:
        relative_path = os.path.relpath(img_path, original_dataset)
        target_path = os.path.join(small_dataset, relative_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copy(img_path, target_path)

    print("Small dataset generated following this path" +
          f"{small_dataset} containing {len(selected_images)} images.")


dataset_original = "./dataset"
dataset_reduit = "./small_dataset"
create_small_dataset(dataset_original, dataset_reduit, seed=22)
