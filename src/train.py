import urllib.request
import ssl
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, Input, concatenate, GlobalAveragePooling2D, Resizing
from tensorflow.keras.optimizers import Adam
from keras_tuner import HyperModel, Hyperband
from load_data import load_images_and_labels, split_data, load_audio_and_labels


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

        image_input = Input(shape=(224, 224, 3), name='image_input')
        audio_input = Input(shape=(40,), name='audio_input')

        x = Resizing(224, 224)(image_input)
        x = base_model(x)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(hp.Float('dropout_rate', 0.2, 0.5, step=0.1))(x)
        x = Dense(hp.Int('units', 64, 256, step=64), activation='relu')(x)
        x = Dropout(hp.Float('dropout_rate2', 0.2, 0.5, step=0.1))(x)

        y = Dense(hp.Int('units', 64, 256, step=64), activation='relu')(audio_input)
        y = Dropout(hp.Float('dropout_rate2', 0.2, 0.5, step=0.1))(y)

        combined = concatenate([x, y])
        output = Dense(7, activation='softmax')(combined)

        model = tf.keras.models.Model(inputs=[image_input, audio_input], outputs=output)

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

    x_images, y_images = load_images_and_labels(image_folder)
    x_audio, y_audio = load_audio_and_labels(audio_folder)

    # Ensure both datasets have the same number of samples
    min_samples = min(len(x_images), len(x_audio))
    x_images = x_images[:min_samples]
    y_images = y_images[:min_samples]
    x_audio = x_audio[:min_samples]
    y_audio = y_audio[:min_samples]

    x_train_img, x_test_img, y_train_img, y_test_img = split_data(x_images, y_images)
    x_train_audio, x_test_audio, y_train_audio, y_test_audio = split_data(x_audio, y_audio)

    # Repeat channels to match MobileNetV2 input
    x_train_img = np.repeat(x_train_img, 3, axis=-1)
    x_test_img = np.repeat(x_test_img, 3, axis=-1)

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
    datagen.fit(x_train_img)

    # GPU Configuration
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU is available and will be used for training.")
    else:
        print("GPU is not available. Training will proceed on CPU.")

    # Hyperparameter tuning with Keras Tuner
    max_epochs = 5  # Reduce the max epochs
    factor = 3

    tuner = Hyperband(
        MobileNetV2HyperModel(),
        objective='val_accuracy',
        max_epochs=max_epochs,
        factor=factor,
        directory='hyperband',
        project_name='emotion_recognition'
    )

    # Calculate the total number of trials
    max_trials = sum(max_epochs // (factor ** i) for i in range(int(np.log(max_epochs) / np.log(factor)) + 1))
    print(f"Total number of trials: {max_trials}")
    print(f"Max epochs per trial: {max_epochs}")

    # Search for the best hyperparameters
    tuner.search([x_train_img, x_train_audio], y_train_img, batch_size=32, epochs=2,
                 validation_data=([x_test_img, x_test_audio], y_test_img))  # Batch size reduziert auf 32

    # Retrieve the best model
    best_model = tuner.get_best_models(num_models=1)[0]

    # Fine-tuning the base model
    base_model = best_model.layers[2]
    base_model.trainable = True
    best_model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
    best_model.fit([x_train_img, x_train_audio], y_train_img, batch_size=32, epochs=2,
                   validation_data=([x_test_img, x_test_audio], y_test_img))  # Batch size reduziert auf 32

    loss, accuracy = best_model.evaluate([x_test_img, x_test_audio], y_test_img)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    best_model.save('emotion_recognition_model.keras')

    # Print the total number of trials
    print(f"Total trials: {len(tuner.oracle.trials)}")


if __name__ == "__main__":
    main()
