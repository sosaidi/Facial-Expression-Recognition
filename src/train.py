import sys
import os
import urllib.request
import ssl
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from keras_tuner import HyperModel, Hyperband
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from load_data import load_images_and_labels, split_data

# Function to download pre-trained weights
def download_weights(url, filename):
    """
    Downloads pre-trained weights for MobileNetV2.

    Uses urllib to fetch the weights from the given URL and saves them to the specified filename.

    Source: Standard practice for downloading files with urllib in Python.
    Reference: https://stackoverflow.com/questions/7243750/download-file-from-web-in-python-3
    """
    context = ssl._create_unverified_context()
    with urllib.request.urlopen(url, context=context) as response, open(filename, 'wb') as out_file:
        data = response.read()
        out_file.write(data)
    print(f"{filename} downloaded successfully.")

# Define a HyperModel for tuning
class MobileNetV2HyperModel(HyperModel):
    def build(self, hp):
        """
        Builds a MobileNetV2 model with hyperparameter tuning capabilities.

        Utilizes Keras Tuner for hyperparameter optimization.

        Source: Common structure for defining HyperModels with Keras Tuner.
        Reference: https://keras.io/keras_tuner/
        """
        base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
        base_model.trainable = False

        model = Sequential([
            tf.keras.layers.Resizing(224, 224),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            Dropout(hp.Float('dropout_rate', 0.2, 0.5, step=0.1)),
            Dense(hp.Int('units', 64, 256, step=64), activation='relu'),
            Dropout(hp.Float('dropout_rate2', 0.2, 0.5, step=0.1)),
            Dense(7, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-4, 5e-4, 1e-3])),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

# Main training function
def main():
    """
    Main function to train the MobileNetV2 model for emotion recognition.

    Includes data loading, preprocessing, model building, and hyperparameter tuning.

    Source: General approach for deep learning model training with Keras and TensorFlow.
    Reference: https://stackoverflow.com/questions/45462244/how-to-use-dataaugmentation-in-keras
    """
    image_folder = './data/images/train'
    audio_folder = './data/audio'
    X, y = load_images_and_labels(image_folder)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Repeat channels to match MobileNetV2 input
    X_train = np.repeat(X_train, 3, axis=-1)
    X_test = np.repeat(X_test, 3, axis=-1)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        shear_range=0.2,
        fill_mode='nearest'
    )
    datagen.fit(X_train)

    # GPU Configuration
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU is available and will be used for training.")
    else:
        print("GPU is not available. Training will proceed on CPU.")

    # Hyperparameter tuning with Keras Tuner
    tuner = Hyperband(
        MobileNetV2HyperModel(),
        objective='val_accuracy',
        max_epochs=10,
        directory='hyperband',
        project_name='emotion_recognition'
    )

    # Search for the best hyperparameters
    tuner.search(datagen.flow(X_train, y_train, batch_size=64), epochs=10, validation_data=(X_test, y_test))

    # Retrieve the best model
    best_model = tuner.get_best_models(num_models=1)[0]

    # Fine-tuning the base model
    base_model = best_model.layers[1]
    base_model.trainable = True
    best_model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
    best_model.fit(datagen.flow(X_train, y_train, batch_size=64), epochs=10, validation_data=(X_test, y_test))

    loss, accuracy = best_model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    best_model.save('emotion_recognition_model.keras')

if __name__ == "__main__":
    main()