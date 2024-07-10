import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import librosa

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical


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
    for label in os.listdir(image_folder):
        label_folder = os.path.join(image_folder, label)
        if os.path.isdir(label_folder):
            for image_name in os.listdir(label_folder):
                image_path = os.path.join(label_folder, image_name)
                if image_path.endswith(('.png', '.jpg', '.jpeg')):
                    image = load_img(image_path, color_mode='grayscale', target_size=(48, 48))
                    image = img_to_array(image)
                    images.append(image)
                    labels.append(label)
    if not images:
        raise ValueError("Keine Bilder gefunden. Überprüfe den Pfad und die Verzeichnisstruktur.")
    images = np.array(images, dtype='float32') / 255.0
    labels = pd.Categorical(labels).codes
    labels = to_categorical(labels, num_classes=len(np.unique(labels)))
    return images, labels


def split_data(x, y):
    """
        Splits data into training and test sets.

        This function uses sklearn's train_test_split to divide the dataset into training and test sets.

        Source: Standard usage of train_test_split from scikit-learn.
        Reference: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
        """
    return train_test_split(x, y, test_size=0.2, random_state=42)


def extract_audio_features(audio_file):
    """
    Extracts audio features (MFCC) from an audio file.

    This function uses librosa to load an audio file and extract its MFCC features, which are then averaged.

    Source: Common practice in audio processing to extract MFCC features.
    Reference: https://stackoverflow.com/questions/49343636/audio-feature-extraction-using-librosa
    """
    y, sr = librosa.load(audio_file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc


def load_audio_and_labels(audio_folder):
    """
    Loads audio files and their corresponding labels from a specified directory.

    This function scans the given folder for subdirectories (each representing a class label),
    loads audio files from each subdirectory, and processes them into arrays suitable for model input.

    Source: Adapted from a common pattern in loading audio datasets in Python.
    Reference: https://stackoverflow.com/questions/49343636/audio-feature-extraction-using-librosa
    """
    audio_features = []
    labels = []
    for label in os.listdir(audio_folder):
        label_folder = os.path.join(audio_folder, label)
        if os.path.isdir(label_folder):
            for audio_name in os.listdir(label_folder):
                audio_path = os.path.join(label_folder, audio_name)
                if audio_path.endswith('.wav'):
                    features = extract_audio_features(audio_path)
                    audio_features.append(features)
                    labels.append(label)
    if not audio_features:
        raise ValueError("Keine Audiodateien gefunden. Überprüfe den Pfad und die Verzeichnisstruktur.")
    audio_features = np.array(audio_features, dtype='float32')
    labels = pd.Categorical(labels).codes
    labels = to_categorical(labels, num_classes=len(np.unique(labels)))
    return audio_features, labels
    
