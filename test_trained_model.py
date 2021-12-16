from datetime import date
from src.misc_utils import load_json_model
from train_generator import loss_func, get_metrics, get_optimizer, SeedGenerator
from data_gen import data_iterator_test
today = str(date.today())
test_path = 'D:/seed_data/generator/test_data/test_batch/'

model_new = load_json_model("generator_model_2021-12-16/generator_model.json",
                            SeedGenerator, "SeedGenerator")
model_new.compile(optimizer=get_optimizer(),
                  loss_fn=loss_func,
                  metric_fn=get_metrics)
model_new.load_weights("./checkpoints/generator_2021-12-16/")
res = model_new.evaluate(data_iterator_test(test_path),
                         return_dict=True)
