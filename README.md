# Leaf Affliction Detection

This project is designed to detect and classify plant leaf diseases using deep learning models. It includes data augmentation, transformation, and visualization tools to enhance model training and prediction accuracy.

## Features

- **Data Augmentation**: Enhance dataset with various image transformations.
- **Model Training**: Train a convolutional neural network to classify leaf diseases.
- **Prediction**: Predict the disease class of a given leaf image.
- **Visualization**: Plot learning curves and dataset distribution.

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/jguigli/leaffliction.git
   cd leaffliction
   ```

2. **Download Dataset**:
   ```bash
   ./dl_dataset.sh
   ```

3. **Set Up Virtual Environment**:
   ```bash
   make install
   ```

## Usage

- **Train the Model**:
  ```bash
  make train arg="<dataset_path> <modified_dataset_path>"
  ```

- **Augment Image**:
  ```bash
  make augmentation arg="-src <source_file>"
  ```

- **Augment Dataset**:
  ```bash
  make augmentation arg="-src <source_directory> -dest <destination_directory>"
  ```

- **Transform Image**:
  ```bash
  make transformation arg="-src <source_file>"
  ```

- **Transform Dataset**:
  ```bash
  make transformation arg="-src <source_directory> -dest <destination_directory>"
  ```

- **Predict Image**:
  ```bash
  make predict arg="<image_path>"
  ```

- **Analyze Dataset Distribution**:
  ```bash
  make distribution arg="<dataset_path>"
  ```

## Clean Up

- **Remove Virtual Environment**:
  ```bash
  make fclean
  ```

## Notes

- Ensure all dependencies are listed in `requirements.txt`.
- Adjust paths and arguments as needed for your specific setup.