from datetime import date
from tensorflow import keras
from src.misc_utils import load_json_model
from train_generator import loss_func, get_metrics, get_optimizer, SeedGenerator
from data_gen import data_iterator_test
today = str(date.today())
test_path = 'D:/seed_data/generator/test_data/test_batch/'

model_new = load_json_model("generator_model_2021-12-16/generator_model.json")
model_new.compile(optimizer='adam',
                  loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=[keras.metrics.CategoricalAccuracy()])
model_new.load_weights("./checkpoints/generator_2021-12-16/")
res = model_new.evaluate(data_iterator_test(test_path),
                         return_dict=True)
