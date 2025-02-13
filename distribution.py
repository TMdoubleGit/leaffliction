import os
import sys
import matplotlib.pyplot as plt
import numpy as np


def analyze_plant_dataset(dataset_path):
    """
    Generate a global pie and bar chart for all images of a specified plant.
    :param dataset_path: Full path to the plant directory
    (e.g. ./dataset/apple).
    """
    if not os.path.isdir(dataset_path):
        print(
            f"Error: The specified path '{dataset_path}' "
            f"does not exist or is not a directory."
        )
        sys.exit(1)

    directory_name = os.path.basename(dataset_path)

    category_counts = {}
    for disease in os.listdir(dataset_path):
        disease_path = os.path.join(dataset_path, disease)
        if os.path.isdir(disease_path):
            count = len(
                [
                    img for img in os.listdir(disease_path)
                    if os.path.isfile(os.path.join(disease_path, img))
                ]
            )
            category_counts[disease] = count

    if not category_counts:
        print(
            f"Error: No categories found in '{dataset_path}'. "
            f"Make sure it contains subdirectories with images."
        )
        sys.exit(1)

    sorted_categories = sorted(
        category_counts.items(), key=lambda x: x[1], reverse=True
    )

    labels = [item[0] for item in sorted_categories]
    counts = [item[1] for item in sorted_categories]

    total_images = sum(counts)
    if not total_images:
        print("No images found in dataset subdirectories")
        sys.exit(1)
    sizes = [count / total_images * 100 for count in counts]

    color_map = plt.cm.tab20(np.linspace(0, 1, len(labels)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f"{directory_name.capitalize()} Dataset Distribution")

    axes[0].pie(
        sizes, labels=labels, autopct="%1.1f%%",
        startangle=140, colors=color_map
    )
    axes[0].set_title("Pie Chart")

    axes[1].bar(labels, counts, color=color_map)
    axes[1].set_title("Bar Chart")
    axes[1].set_xlabel("Categories")
    axes[1].set_ylabel("Number of Images")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: You must provide the dataset path as an argument.")
        print("Usage: python distribution.py <dataset_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    analyze_plant_dataset(dataset_path)
