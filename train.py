from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Rescaling
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

import sys

def train(dataset_path):

    # declaration d'un modele simple (5 couches)
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(8, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    # generer la data de training
    train_data = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        labels='inferred',  # label par sous dossiers (8)
        label_mode='categorical',  # encodage one-hot
        image_size=(128, 128),  # a checker
        batch_size=32,
        validation_split=0.2,
        subset='training',
    )

    # generer la data de validation
    validation_data = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        labels='inferred',
        label_mode='categorical',
        image_size=(128, 128),
        batch_size=32,
        validation_split=0.2,
        subset='validation',
    )

    # Normalise chaque image
    normalization_layer = Rescaling(1./255)
    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

    model.fit(
        train_data,
        validation_data=validation_data,
        epochs=100)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: You must provide the dataset path as an argument.")
        print("Usage: python train.py <dataset_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    train(dataset_path)