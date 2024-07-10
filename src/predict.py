import cv2
import numpy as np
from keras.src.saving.saving_lib import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt

# Emotion label dictionary
emotion_dict = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}

def preprocess_image(image_path):
    """
    Preprocesses an image for emotion prediction.

    Loads an image from the specified path, converts it to grayscale, resizes it,
    and normalizes pixel values.

    Source: Standard image preprocessing techniques for neural networks.
    Reference: https://stackoverflow.com/questions/42573954/how-to-normalize-images-in-tensorflow-and-keras
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    image = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)  # Convert to 3 channels
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_emotion(image_path, model):
    """
    Predicts the emotion displayed in a given image.

    Uses the provided model to predict the emotion of a preprocessed image.

    Source: Common method for making predictions with Keras models.
    Reference: https://stackoverflow.com/questions/48394420/how-to-predict-new-samples-using-a-keras-model
    """
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    emotion = np.argmax(prediction)
    emotion_label = emotion_dict[emotion]
    print(f"Predicted emotion: {emotion_label}")

def capture_image_from_webcam(model):
    """
    Captures frames from the webcam and predicts the emotion in real-time.

    Displays the predicted emotion label on the webcam feed.

    Source: Common approach for real-time webcam processing with OpenCV and Keras.
    Reference: https://stackoverflow.com/questions/44752240/how-to-use-opencv-with-tensorflow
    """
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        image = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        normalized = image.astype('float32') / 255.0
        reshaped = np.expand_dims(normalized, axis=0)

        prediction = model.predict(reshaped)
        emotion = np.argmax(prediction)
        emotion_label = emotion_dict[emotion]

        cv2.putText(frame, emotion_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def get_gradcam_heatmap(model, image, last_conv_layer_name):
    """
    Generates a Grad-CAM heatmap for a given image and model.

    Source: Implementation of Grad-CAM (Gradient-weighted Class Activation Mapping).
    Reference: https://github.com/keras-team/keras-io/blob/master/examples/vision/grad_cam.py
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, np.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

def display_gradcam(image_path, model, last_conv_layer_name):
    """
    Displays a Grad-CAM heatmap over the original image.

    Superimposes the heatmap on the image to visualize the areas that
    contribute most to the predicted class.

    Source: Common practice for visualizing model attention using Grad-CAM.
    Reference: https://github.com/keras-team/keras-io/blob/master/examples/vision/grad_cam.py
    """
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    heatmap = get_gradcam_heatmap(model, img_array, last_conv_layer_name)
    heatmap = np.uint8(255 * heatmap)
    jet = plt.get_cmap("jet")
    jet_heatmap = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_heatmap[heatmap]

    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.size[0], img.size[1]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * 0.4 + img_array[0]
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Load the trained model
    model = load_model('emotion_recognition_model.keras', compile=False)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Prediction example
    image_path = './data/images/validation/fear/101.jpg'
    predict_emotion(image_path, model)

    # Grad-CAM example
    display_gradcam(image_path, model, 'Conv_1_bn')

    # Webcam (comment out if not needed)
    # capture_image_from_webcam(model)