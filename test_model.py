import numpy as np
from data_gen import data_iterator_test, data_iterator
from train_generator import loss_func, get_metrics, get_optimizer, SeedGenerator
from src.misc_utils import load_json_model


if __name__ == "__main__":
    test_path = 'D:/seed_data/generator/test_data/test_batch/'
    train_path = 'D:/seed_data/generator/train_data/train_batch/'
    model = load_json_model("generator_model/generator_model.json", SeedGenerator, "SeedGenerator")
    model.compile(optimizer=get_optimizer(),
                  loss_fn=loss_func,
                  metric_fn=get_metrics)
    model.load_weights("./checkpoints/generator/")
    for X_in, y in data_iterator(train_path):
        y_pred = model.predict(X_in)
        mask = np.where(X_in[1][0] < 1)
        input = X_in[0][:10, :10, :-1].sum(-1)
        breakpoint()
