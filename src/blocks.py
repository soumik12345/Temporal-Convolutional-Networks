import inspect
from .utils import *
import tensorflow as tf
from tensorflow.keras import layers


class LayerNorm(tf.keras.layers.Layer):

    def __init__(self, hidden_size):
        """
        Layer Normalisation Operation

        :param hidden_size: Hidden Size.

        :returns: None.
        """
        super(LayerNorm, self).__init__()
        self.hidden_size = hidden_size

    def build(self, input_shape):
        self.scale = self.add_weight(
            'layer_norm_scale', [self.hidden_size],
            dtype=tf.float64, initializer=tf.ones_initializer()
        )
        self.bias = self.add_weight(
            'layer_norm_bias', [self.hidden_size],
            dtype=tf.float64, initializer=tf.zeros_initializer
        )
        self.built = True

    def call(self, x, epsilon=1e-6):
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(
            tf.square(x - mean),
            axis=[-1], keepdims=True
        )
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        return norm_x * self.scale + self.bias