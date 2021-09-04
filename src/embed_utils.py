import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras import layers
from tensorflow.keras import regularizers


def conv2d_block(X, num_filters, kernel_size, padding='SAME'):
    out = layers.Conv2D(filters=num_filters,
                        kernel_size=kernel_size,
                        kernel_regularizer=regularizers.L2(1e-4),
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
    out = layers.Add()([out, X])
    out = layers.ReLU()(out)
    return out
