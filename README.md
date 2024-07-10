# Facial Expression and Audio Emotion Recognition

This project aims to recognize facial expressions and audio-based emotions using deep learning techniques. The MobileNetV2 model is employed for visual feature extraction, and a custom classifier is trained to recognize seven different emotions from images and audio.

## Project Structure

- **Facial-Expression-Recognition/**
  - **.ipynb_checkpoints/**: Jupyter notebook checkpoints.
  - **.venv/**: Virtual Python environment.
  - **data/**
    - **images/**: Training and validation images organized by emotion.
      - **train/**: Training images.
      - **validation/**: Validation images.
    - **audio/**: Audio recordings organized by emotion.
      - **train/**: Training audio files.
      - **validation/**: Validation audio files.
  - **docs/**: Documentation files.
    - `project_documentation.md`: Detailed project documentation.
  - **notebook/**: Jupyter notebooks for data analysis.
    - `exploratory_data_analysis.ipynb`: Notebook for exploratory data analysis.
  - **results/**: Directory for storing model results.
    - `model_output/`: Outputs from the model.
    - `test_results.txt`: Test results.
  - **src/**: Source code.
    - `__init__.py`: Initialization file.
    - `load_data.py`: Script for loading and preprocessing data.
    - `mobilenet_v2_weights.h5`: Pretrained MobileNetV2 weights.
    - `model.py`: Model architecture and training functions.
    - `predict.py`: Script for making predictions.
    - `train.py`: Script for training the model.
  - **tf_venv/**: TensorFlow virtual environment.
  - `emotion_recognition_model.h5`: Trained model (H5 format).
  - `emotion_recognition_model.keras`: Trained model (Keras format).
  - `get-pip.py`: Script for installing Pip.
  - `mobilenet_v2_weights.h5`: Pretrained MobileNetV2 weights.
  - `README.md`: Project overview and instructions.
  - `requirements.txt`: Project dependencies.

## Key Components

### Data Loading and Preprocessing

- **`load_data.py`**: Contains functions to load images and audio, and to split data into training and test sets.
  - `load_images_and_labels(image_folder)`: Loads images and labels from the specified folder.
  - `load_audio_and_labels(audio_folder)`: Loads audio data and labels from the specified folder.
  - `split_data(X, y)`: Splits data into training and test sets.

### Model Training

- **`train.py`**: Main script for training the model.
  - `download_weights(url, filename)`: Downloads pretrained weights for MobileNetV2.
  - `build_model(input_shape, num_classes, weights_path)`: Builds and returns the model architecture.
  - `main()`: Loads data, augments data, builds the model, and trains it.

### Model Prediction

- **`predict.py`**: Script for making predictions on images, audio, or webcam feed.
  - `preprocess_image(image_path)`: Preprocesses an image for prediction.
  - `preprocess_audio(audio_path)`: Preprocesses an audio file for prediction.
  - `predict_emotion(image_path, audio_path)`: Predicts the emotion from a given image or audio file.
  - `capture_image_from_webcam()`: Captures images from the webcam and predicts emotions in real-time.

## Installation and Execution

### Prerequisites

- Python 3.12
- TensorFlow 2.16.1
- Librosa
- Other dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd Facial-Expression-Recognition
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv .venv
    source .venv/bin/activate  # For Unix/MacOS
    .venv\Scripts\activate  # For Windows
    ```

3. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

### Training the Model

1. Ensure the image and audio data are available in the `data/images/train` and `data/audio` directories respectively.
2. Run the training script:
    ```sh
    python src/train.py
    ```

### Predicting Emotions

1. To predict emotions from an image or audio file, edit `src/predict.py` and provide the paths to the files:
    ```sh
    python src/predict.py
    ```

2. To enable emotion recognition through the webcam, uncomment the `capture_image_from_webcam()` function in `src/predict.py` and run the script:
    ```sh
    python src/predict.py
    ```
---