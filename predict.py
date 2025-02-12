import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle

from transformation import apply_transformations_to_image

from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont


def display_prediction(transformed_images, predicted_class):
    img_original = Image.fromarray(transformed_images[0])
    img_transformed = Image.fromarray(transformed_images[2])

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


def predict_image(image_path):
    predictions = []

    model = load_model('./saved_model/leafflication.keras')
    with open("./saved_model/classes_names.pkl", "rb") as fichier:
        classes_names = pickle.load(fichier)

    transformed_images = apply_transformations_to_image(image_path,
                                                        save_dir=None,
                                                        transformations={"blur", "mask", "roi", "analyze", "pseudolandmarks"}
                                                        )

    processed_images = []
    for img in transformed_images:
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        processed_images.append(img)

    for img in processed_images:
        prediction = model.predict(np.expand_dims(img, axis=0))
        predictions.append(prediction)

    avg_prediction = np.mean(predictions, axis=0)
    final_class = np.argmax(avg_prediction)

    print(f'predictions : {predictions}')
    
    predicted_class = classes_names[final_class]
    print(predicted_class)
    
    return predicted_class, transformed_images


def predict(image_path):
    predicted_class, transformed_images = predict_image(image_path)
    # display_prediction(transformed_images, predicted_class)
    

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: You must provide the image path as an argument.")
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
        
    try:
        predict(image_path)
    except Exception as e:
        print(e)
        sys.exit(1)