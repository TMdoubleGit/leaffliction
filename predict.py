import sys
import numpy as np
import pickle
import os
import logging
import absl.logging

from transformation import apply_transformations_to_image

from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

absl.logging.set_verbosity(absl.logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

logging.getLogger("absl").setLevel(logging.ERROR)


def display_prediction(transformed_images, predicted_class):
    img = Image.fromarray(transformed_images[0])
    img_t = Image.fromarray(transformed_images[2])

    concatenated_img = Image.new('RGB',
                                 (img.width + img_t.width, img.height + 150)
                                 )
    concatenated_img.paste(img, (0, 0))
    concatenated_img.paste(img_t, (img.width, 0))

    draw = ImageDraw.Draw(concatenated_img)
    text = f"Class predicted: {predicted_class}"
    font = ImageFont.load_default()

    image_width = img.width + img_t.width
    text_x = (image_width - 150) // 2
    text_y = img.height + 50
    text_color = (255, 255, 255)
    draw.text((text_x, text_y), text, fill=text_color, font=font)

    concatenated_img.show()


def predict_image(image_path):
    predictions = []

    model = load_model('./saved_model/leafflication.keras')

    with open("./saved_model/classes_names.pkl", "rb") as fichier:
        classes_names = pickle.load(fichier)

    imgs_t = apply_transformations_to_image(image_path,
                                            save_dir=None,
                                            transformations={"blur",
                                                             "mask",
                                                             "roi",
                                                             "analyze",
                                                             "pseudolandmarks"
                                                             }
                                            )

    processed_images = []
    for img in imgs_t:
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        processed_images.append(img)

    for img in processed_images:
        prediction = model.predict(np.expand_dims(img, axis=0))
        predictions.append(prediction)

    avg_prediction = np.mean(predictions, axis=0)
    final_class = np.argmax(avg_prediction)

    predicted_class = classes_names[final_class]

    return predicted_class, imgs_t


def predict(image_path):
    predicted_class, transformed_images = predict_image(image_path)
    display_prediction(transformed_images, predicted_class)


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
