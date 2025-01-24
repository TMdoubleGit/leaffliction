import sys
import os
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Rescaling

from .augmentation import augment_dataset
# from .transformation import transform_dataset

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
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        labels='inferred',  # label par sous dossiers (8)
        label_mode='categorical',
        image_size=(128, 128),  # 128 OU 256 ?????????????????
        batch_size=32,
        validation_split=0.2,
        subset='training',
    )

    # generer la data de validation
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
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

    training_metrics = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=100)
    
    print(f"========== Training metrics ==========\n", +
            f"loss: {training_metrics.history['loss']}\n" +
            f"accuracy: {training_metrics.history['accuracy']}\n" +
            f"\n========== Validation metrics ==========\n", +
            f"val_loss: {training_metrics.history['val_loss']}\n" +
            f"val_accuracy: {training_metrics.history['val_accuracy']}\n"
    )

    # PATH a modifier avant EVALUATION
    if not os.path.exists('./augmented_directory/saved_model'):
        os.makedirs('./augmented_directory/saved_model')
    model.save('./augmented_directory/saved_model/leafflication')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: You must provide the dataset path as an argument.")
        print("Usage: python train.py <dataset_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]

    # LIGNES a SUPPRIMER avant EVALUATION
    augment_dataset(dataset_path, "augmented_directory")
    # transform_dataset(dataset_path, "augmented_directory")
    # ZIP le doss avec le shasum une fois fini

    train(dataset_path)