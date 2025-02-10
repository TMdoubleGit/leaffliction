import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont


def display_prediction(image_path, predicted_class):
    img_original = Image.open(image_path)
    img_transformed = Image.open('path/to/image2.jpg') # REMPLACER PAR L'IMAGE TRANSFORMEE

    img_transformed = img_transformed.resize(img_original.size)

    concatenated_img = Image.new('RGB', (img_original.width + img_transformed.width, img_original.height))
    concatenated_img.paste(img_original, (0, 0))
    concatenated_img.paste(img_transformed, (img_original.width, 0))

    draw = ImageDraw.Draw(concatenated_img)
    text = f"Class predicted: {predicted_class}"
    font = ImageFont.truetype("arial.ttf", size=20)
    text_position = (10, img_transformed.height + 10)
    text_color = (255, 255, 255)
    draw.text(text_position, text, fill=text_color, font=font)

    concatenated_img.show()


def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    # img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    ########################3 transformation de l'image ###########################

    return img_array


def predict_image(image_path):
    model = load_model('./saved_model/leafflication.keras')

    img_array = load_and_preprocess_image(image_path)
    # blurred = preprocess_image(gaussian_blur(image_path))
    # masked = preprocess_image(apply_mask(image_path))
    # roi = preprocess_image(extract_roi(image_path))
    
    # batch = np.vstack([original, blurred, masked, roi])
    
    predictions = model.predict(img_array)
    # predictions = model.predict(batch)

    # avg_prediction = np.mean(predictions, axis=0)  # Moyenne des scores de confiance
    # final_class = np.argmax(avg_prediction)  # Classe avec la plus haute confiance

    class_index = np.argmax(predictions)
    # confidence = predictions[0][class_index]
    
    class_names = ['apple_black_rot',
                   'apple_healthy',
                   'apple_rust',
                   'apple_scab',
                   'grape_black_rot',
                   'grape_esca',
                   'grape_healthy',
                   'grape_spot']
    
    predicted_class = class_names[class_index]
    
    return predicted_class


def predict(image_path):
    predicted_class = predict_image(image_path)
    display_prediction(image_path, predicted_class)
    

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