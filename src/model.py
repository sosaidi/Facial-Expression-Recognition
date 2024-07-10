import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Layer
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D

from src.train import download_weights


class AttentionLayer(Layer):
    """
        Custom Attention Layer.

        This layer applies a simple attention mechanism over the input. It calculates weights
        for each element in the input sequence and computes a weighted sum to produce the output.

        Source: Inspired by common patterns in implementing attention mechanisms in neural networks.
        Reference: https://stackoverflow.com/questions/52976178/how-to-implement-an-attention-layer-in-keras
        """
    def __init__(self):
        super(AttentionLayer, self).__init__()
        self.W = None

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(shape=(input_shape[-1],), initializer='zeros', trainable=True)
        self.u = self.add_weight(shape=(input_shape[-1],), initializer='glorot_uniform', trainable=True)

    def call(self, x):
        v = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        vu = tf.tensordot(v, self.u, axes=1)
        alphas = tf.nn.softmax(vu)
        output = tf.reduce_sum(x * tf.expand_dims(alphas, -1), 1)
        return output

def build_model(input_shape=(48, 48, 3), num_classes=7, weights_path='mobilenet_v2_weights.h5'):
    """
        Builds and returns the MobileNetV2-based model with an attention layer.

        This function constructs a neural network model using MobileNetV2 as the base model and adds
        a custom attention layer on top. It includes dropout layers for regularization and dense layers
        for classification.

        Source: Standard practice of model building in TensorFlow/Keras.
        Reference for MobileNetV2 setup:
        https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2
        Reference for downloading weights:
        https://stackoverflow.com/questions/49250523/how-to-download-a-file-over-http-using-python
        """
    if not os.path.exists(weights_path):
        download_weights('https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5', weights_path)
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
    base_model.load_weights(weights_path)
    base_model.trainable = False
    model = Sequential([
        tf.keras.layers.Resizing(224, 224),
        base_model,
        AttentionLayer(),
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    
