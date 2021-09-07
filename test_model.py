import numpy as np
import tensorflow as tf
from data_gen import data_iterator
from train_generator import loss_func, get_metrics, get_optimizer, SeedGenerator
from src.misc_utils import load_json_model


if __name__ == "__main__":
    test_path = 'D:/seed_data/generator/test_data/test_batch/'
    with tf.device('/cpu:0'):
        model = load_json_model("generator_model/generator_model.json", SeedGenerator, "SeedGenerator")
        model.compile(optimizer=get_optimizer(),
                      loss_fn=loss_func,
                      metric_fn=get_metrics)
        model.load_weights("./checkpoints/generator/")

    for X_in, y in data_iterator(test_path):
        with tf.device('/cpu:0'):
            y_pred = model.predict(X_in)
        print(np.vstack((y_pred.argmax(-1), y.argmax(-1))))
        mask = np.where(X_in[1][0] < 1)
        input = X_in[:10, :10, :-1].sum(-1)
        breakpoint()
