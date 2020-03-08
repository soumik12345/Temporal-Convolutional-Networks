from src.blocks import *
import tensorflow as tf


inp = tf.keras.Input(shape=(20, 1))
x = TCNLayer(return_sequences=False)(inp)
x = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs=inp, outputs=x)

model.summary()