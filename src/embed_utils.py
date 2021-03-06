import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import regularizers


def conv2d_block(X, num_filters, kernel_size, padding='SAME'):
    out = tf.keras.layers.Conv2D(filters=num_filters,
                                 kernel_size=kernel_size,
                                 strides=1,
                                 padding=padding,
                                 activation=None,
                                 use_bias=False)(X)
    out = tf.keras.layers.LayerNormalization(epsilon=1e-6)(out)
    return out


def res_block(X, num_filters, kernel_size, padding='SAME'):
    out = conv2d_block(X, num_filters,
                       kernel_size, padding=padding)
    out = tf.keras.layers.ReLU()(out)
    out = conv2d_block(out, num_filters,
                       kernel_size, padding=padding)
    out = tf.keras.layers.Add()([out, X])
    out = tf.keras.layers.ReLU()(out)
    return out
