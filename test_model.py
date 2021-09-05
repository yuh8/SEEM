import pickle
import glob
import numpy as np
from train_generator import loss_func, get_metrics, get_optimizer, SeedGenerator
from src.misc_utils import load_json_model


if __name__ == "__main__":
    test_path = 'D:/seed_data/generator/test_data/test_batch/'
    model = load_json_model("generator_model/generator_model.json", SeedGenerator, "SeedGenerator")
    model.compile(optimizer=get_optimizer(),
                  loss_fn=loss_func,
                  metric_fn=get_metrics)
    model.load_weights("./checkpoints/generator/")

    for f_name in glob.glob(test_path + 'Xy_*.pkl'):
        with open(f_name, 'rb') as handle:
            Xy = pickle.load(handle)

        X_in = Xy[0][0].todense()
        y = Xy[1].todense()
        y_pred = model.predict(X_in)
        mask = np.where(X_in[1][0] < 1)
        input = X_in[:10, :10, :-1].sum(-1)
        breakpoint()
