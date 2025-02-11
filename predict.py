import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle

from transformation import apply_transformations_to_image

from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont


def display_prediction(image_path, predicted_class):
    img_original = Image.open(image_path)
    img_transformed = Image.open(image_path) # REMPLACER PAR L'IMAGE TRANSFORMEE

    img_transformed = img_transformed.resize(img_original.size)

    concatenated_img = Image.new('RGB', (img_original.width + img_transformed.width, img_original.height + 150))
    concatenated_img.paste(img_original, (0, 0))
    concatenated_img.paste(img_transformed, (img_original.width, 0))

    draw = ImageDraw.Draw(concatenated_img)
    text = f"Class predicted: {predicted_class}"
    font = ImageFont.load_default()

    image_width = img_original.width + img_transformed.width
    text_x = (image_width - 150) // 2
    text_y = img_original.height + 50
    text_color = (255, 255, 255)
    draw.text((text_x, text_y), text, fill=text_color, font=font)

    concatenated_img.show()


# def load_and_preprocess_image(image_path):
#     img = Image.open(image_path).convert('RGB')
#     # img_array = np.array(img) / 255.0
#     img_array = np.expand_dims(img, axis=0)

#     ########################3 transformation de l'image ###########################

#     return img_array


def predict_image(image_path):
    model = load_model('./saved_model/leafflication.keras')
    # manage ERROR

    # img_array = load_and_preprocess_image(image_path)

    transformed_images = apply_transformations_to_image(image_path, save_dir=None, transformations={"blur", "mask", "roi", "analyze", "pseudolandmarks"})
    
    print(len(transformed_images))
    batch = np.vstack(transformed_images)
    batch = np.expand_dims(batch, axis=0)
    
    # predictions = model.predict(img_array)
    predictions = model.predict(batch)

    avg_prediction = np.mean(predictions, axis=0)  # Moyenne des scores de confiance
    final_class = np.argmax(avg_prediction)  # Classe avec la plus haute confiance

    print(predictions)
    print(avg_prediction)
    print(final_class)

    # class_index = np.argmax(predictions)
    # confidence = predictions[0][class_index]
    
    with open("./saved_model/classes_names.pkl", "rb") as fichier:
        classes_names = pickle.load(fichier)
        # manage ERROR
    
    predicted_class = classes_names[final_class]
    
    return predicted_class


def predict(image_path):
    predicted_class = predict_image(image_path)
    # display_prediction(image_path, predicted_class)
    

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: You must provide the image path as an argument.")
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        predict(image_path)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)