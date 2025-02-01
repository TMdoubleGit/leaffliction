import sys
import os
import tensorflow as tf
import matplotlib as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Rescaling

from .augmentation import augment_dataset
# from .transformation import transform_dataset

def plot_learning_curves(self, name, curves_train, curves_validation):
    plt.plot(range(len(curves_train)), curves_train, curves_validation)
    plt.xlabel('Epochs')
    plt.ylabel(name)
    plt.show()
    plt.show()

def train(dataset_path):

    # declaration d'un modele simple (5 couches)
    model = Sequential([
        Rescaling(1./255),
        Conv2D(32, (3, 3), activation='relu'),
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
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        labels='inferred',  # label par sous dossiers (8)
        label_mode='categorical',
        batch_size=32,
        validation_split=0.2,
        subset='training',
    )

    # generer la data de validation
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        labels='inferred',
        label_mode='categorical',
        batch_size=32,
        validation_split=0.2,
        subset='validation',
    )

    training_metrics = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=10)
    
    print(f"========== Training metrics ==========\n", +
            f"loss: {training_metrics.history['loss']}\n" +
            f"accuracy: {training_metrics.history['accuracy']}\n" +
            f"\n========== Validation metrics ==========\n", +
            f"val_loss: {training_metrics.history['val_loss']}\n" +
            f"val_accuracy: {training_metrics.history['val_accuracy']}\n"
    )

    plot_learning_curves('Loss', training_metrics.history['loss'], training_metrics.history['val_loss'])
    plot_learning_curves('Accuracy', training_metrics.history['accuracy'], training_metrics.history['val_accuracy'])

    # PATH a modifier avant EVALUATION
    if not os.path.exists('./saved_model'):
        os.makedirs('./saved_model')
    model.save('./saved_model/leafflication')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: You must provide the dataset path as an argument.")
        print("Usage: python train.py <dataset_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]

    augment_dataset(dataset_path, "dataset_training")
    # transform_dataset("dataset_training")

    train("dataset_training")