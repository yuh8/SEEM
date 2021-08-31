import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras import layers


def conv2d_block(X, num_filters, kernel_size, padding='SAME'):
    out = layers.Conv2D(filters=num_filters,
                        kernel_size=kernel_size,
                        strides=1,
                        padding=padding,
                        activation=None,
                        use_bias=False)(X)
    out = layers.BatchNormalization()(out)
    return out


def res_block(X, num_filters, kernel_size, padding='SAME'):
    out = conv2d_block(X, num_filters,
                       kernel_size, padding=padding)
    out = layers.ReLU()(out)
    out = conv2d_block(X, num_filters,
                       kernel_size, padding=padding)
    out = X + out
    out = layers.ReLU()(out)
    return out
