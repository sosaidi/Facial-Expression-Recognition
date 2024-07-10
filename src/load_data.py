import os
import numpy as np
import pandas as pd
from keras.src.utils import load_img, img_to_array, to_categorical


def load_images_and_labels(image_folder):
    """
    Loads images and their corresponding labels from a specified directory.

    This function scans the given folder for subdirectories (each representing a class label),
    loads images from each subdirectory, and processes them into arrays suitable for model input.

    Source: Adapted from a common pattern in loading image datasets in Python.
    Reference: https://stackoverflow.com/questions/53649470/how-to-load-all-images-using-imagedatagenerator-flow-from-directory
    """
    images = []
    labels = []

    for root, dirs, files in os.walk(image_folder):
        print(f"Verzeichnis: {root}")
        print(f"Unterverzeichnisse: {dirs}")
        print(f"Dateien: {files}")

    for label in os.listdir(image_folder):
        label_folder = os.path.join(image_folder, label)
        if not os.path.isdir(label_folder):
            continue

        for image_name in os.listdir(label_folder):
            image_path = os.path.join(label_folder, image_name)
            if not image_path.endswith(('.png', '.jpg', '.jpeg')):
                continue
            # Convert image to array
            image = load_img(image_path, color_mode='grayscale', target_size=(48, 48))
            image = img_to_array(image)
            images.append(image)
            labels.append(label)

    # Normalize images and convert labels to categorical format
    images = np.array(images, dtype='float32') / 255.0
    labels = pd.Categorical(labels).codes
    labels = to_categorical(labels, num_classes=len(np.unique(labels)))

    print(f"Anzahl geladener Bilder: {len(images)}")
    print(f"Anzahl geladener Labels: {len(labels)}")

    if len(images) == 0:
        raise ValueError("Keine Bilder gefunden. Überprüfe den Pfad und die Verzeichnisstruktur.")

    return images, labels


def split_data(X, y):
    """
    Splits data into training and test sets.

    This function uses sklearn's train_test_split to divide the dataset into training and test sets.

    Source: Standard usage of train_test_split from scikit-learn.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=0.2, random_state=42)


def extract_audio_features(audio_file):
    """
    Extracts audio features (MFCC) from an audio file.

    This function uses librosa to load an audio file and extract its MFCC features, which are then averaged.

    Source: Common practice in audio processing to extract MFCC features.
    Reference: https://stackoverflow.com/questions/49343636/audio-feature-extraction-using-librosa
    """
    import librosa
    y, sr = librosa.load(audio_file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc