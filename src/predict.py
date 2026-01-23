import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


def main():
    parser = argparse.ArgumentParser(description="Predict malaria infection from a single cell image.")
    parser.add_argument("--model", default="models/malaria_cnn.keras", help="Path to saved model (.keras or .h5)")
    parser.add_argument("--image", required=True, help="Path to input image file")
    parser.add_argument("--img_size", type=int, default=128, help="Image width/height (square). Default=128")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)

    img = image.load_img(args.image, target_size=(args.img_size, args.img_size))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    prob_uninfected = float(model.predict(arr, verbose=0)[0][0])

    if prob_uninfected >= 0.5:
        pred = "Uninfected"
        conf = prob_uninfected
    else:
        pred = "Parasitized"
        conf = 1.0 - prob_uninfected

    print(f"Prediction: {pred} (confidence: {conf:.3f})")
    print(f"Raw sigmoid output (P(Uninfected)): {prob_uninfected:.3f}")


if __name__ == "__main__":
    main()
