import sys
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Rescaling, Dropout

# from .augmentation import augment_dataset
# from .transformation import transform_dataset

def plot_learning_curves(name, curves_train, curves_validation):
    plt.plot(range(len(curves_train)), curves_train, curves_validation)
    plt.xlabel('Epochs')
    plt.ylabel(name)
    plt.show()
    plt.show()

def train(dataset_path):

    # generer la data de training
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        labels='inferred',  # label par sous dossiers (8)
        label_mode='categorical',
        batch_size=32,
        validation_split=0.2,
        subset='training',
        seed=42,
    )

    # generer la data de validation
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        labels='inferred',
        label_mode='categorical',
        batch_size=32,
        validation_split=0.2,
        subset='validation',
        seed=42,
    )

    train_class_names = train_dataset.class_names
    classes_number = len(train_class_names)

    model = Sequential([
        Rescaling(1./255),
        Conv2D(filters=16, kernel_size=4, activation='relu'),
        MaxPooling2D(),
        Conv2D(filters=32, kernel_size=4, activation='relu'),
        MaxPooling2D(),
        Dropout(0.1),
        Conv2D(filters=64, kernel_size=4, activation='relu'),
        MaxPooling2D(),
        Dropout(0.1),
        Conv2D(filters=128, kernel_size=4, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(units=128, activation='relu'),
        Dense(units=classes_number, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])


    training_metrics = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=3)
    
    print(f"========== Training metrics ==========\n" +
            f"loss: {training_metrics.history['loss'][-1]}\n" +
            f"accuracy: {training_metrics.history['accuracy'][-1]}\n" +
            f"\n========== Validation metrics ==========\n" +
            f"val_loss: {training_metrics.history['val_loss'][-1]}\n" +
            f"val_accuracy: {training_metrics.history['val_accuracy'][-1]}\n"
    )

    plot_learning_curves('Loss', training_metrics.history['loss'], training_metrics.history['val_loss'])
    plot_learning_curves('Accuracy', training_metrics.history['accuracy'], training_metrics.history['val_accuracy'])


    if not os.path.exists('./saved_model'):
        os.makedirs('./saved_model')
    model.save('./saved_model/leafflication.keras')

    with open("./saved_model/classes_names.pkl", "wb") as fichier:
        pickle.dump(train_class_names, fichier)
        # manage ERROR


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: You must provide the dataset path as an argument.")
        print("Usage: python train.py <dataset_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]

    # augment_dataset(dataset_path, "dataset_training")
    # transform_dataset("dataset_training")

    # partie ZIP

    try:
        train(dataset_path)
    except Exception as e:
        print(e)
        sys.exit(1)