import argparse
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping


def build_model(input_shape=(128, 128, 3)) -> tf.keras.Model:
    model = Sequential([
        Conv2D(16, (3, 3), activation="relu", input_shape=input_shape),
        MaxPool2D(2, 2),
        Dropout(0.2),

        Conv2D(32, (3, 3), activation="relu"),
        MaxPool2D(2, 2),
        Dropout(0.3),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPool2D(2, 2),
        Dropout(0.3),

        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def save_curves(history, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Accuracy curve
    plt.figure()
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_curve.png"), dpi=200)
    plt.close()

    # Loss curve
    plt.figure()
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train a CNN for malaria cell classification.")
    parser.add_argument("--data_dir", required=True,
                        help="Path to dataset folder containing two subfolders: Parasitized/ and Uninfected/")
    parser.add_argument("--img_size", type=int, default=128, help="Image width/height (square). Default=128")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size. Default=16")
    parser.add_argument("--epochs", type=int, default=20, help="Max epochs. Default=20")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split. Default=0.2")
    parser.add_argument("--model_out", default="models/malaria_cnn.keras", help="Model output path.")
    parser.add_argument("--results_dir", default="results", help="Directory to save plots.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=args.val_split)

    train_gen = datagen.flow_from_directory(
        directory=args.data_dir,
        target_size=(args.img_size, args.img_size),
        class_mode="binary",
        batch_size=args.batch_size,
        subset="training",
        shuffle=True,
    )

    val_gen = datagen.flow_from_directory(
        directory=args.data_dir,
        target_size=(args.img_size, args.img_size),
        class_mode="binary",
        batch_size=args.batch_size,
        subset="validation",
        shuffle=False,
    )

    model = build_model(input_shape=(args.img_size, args.img_size, 3))
    model.summary()

    early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=[early_stop],
    )

    model.save(args.model_out)
    save_curves(history, args.results_dir)

    print("\n✅ Training complete.")
    print(f"Saved model to: {args.model_out}")
    print(f"Saved curves to: {args.results_dir}/")


if __name__ == "__main__":
    main()