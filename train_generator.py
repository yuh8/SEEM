import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from multiprocessing import freeze_support
from data_gen import data_iterator, data_iterator_test
from src.embed_utils import conv2d_block, res_block
from src.misc_utils import create_folder, save_model_to_json, load_json_model
from src.CONSTS import (ATOM_LIST, CHARGES, BOND_NAMES,
                        MAX_NUM_ATOMS, FEATURE_DEPTH,
                        NUM_FILTERS, FILTER_SIZE, NUM_RES_BLOCKS)


def core_model():
    # [BATCH, MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH + 1]
    X = layers.Input(shape=(MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH + 1))
    num_act_charge_actions = len(ATOM_LIST) * len(CHARGES)
    num_loc_bond_actions = (MAX_NUM_ATOMS - 1) * len(BOND_NAMES)
    X_mask = layers.Input(shape=(num_act_charge_actions + num_loc_bond_actions))

    # [BATCH, MAX_NUM_ATOMS, MAX_NUM_ATOMS, NUM_FILTERS]
    out = conv2d_block(X, NUM_FILTERS, FILTER_SIZE)

    # [BATCH, MAX_NUM_ATOMS/16, MAX_NUM_ATOMS/16, NUM_FILTERS]
    major_block_size = NUM_RES_BLOCKS // 4
    for _ in range(major_block_size):
        out = res_block(out, NUM_FILTERS, FILTER_SIZE)
    out = layers.MaxPool2D(2, 2)(out)

    # out = conv2d_block(out, NUM_FILTERS, FILTER_SIZE)
    for _ in range(major_block_size):
        out = res_block(out, NUM_FILTERS, FILTER_SIZE)
    out = layers.MaxPool2D(2, 2)(out)

    # out = conv2d_block(out, NUM_FILTERS * 2, FILTER_SIZE)
    for _ in range(major_block_size):
        out = res_block(out, NUM_FILTERS, FILTER_SIZE)
    out = layers.MaxPool2D(2, 2)(out)

    # out = conv2d_block(out, NUM_FILTERS, FILTER_SIZE)
    for _ in range(major_block_size):
        out = res_block(out, NUM_FILTERS, FILTER_SIZE)
    out = layers.MaxPool2D(2, 2)(out)

    # out = conv2d_block(out, NUM_FILTERS * 2, FILTER_SIZE)
    for _ in range(major_block_size):
        out = res_block(out, NUM_FILTERS, FILTER_SIZE)
    out = layers.MaxPool2D(2, 2)(out)

    # # [BATCH, MAX_NUM_ATOMS/32, MAX_NUM_ATOMS/32, ACTION_SIZE]
    # out = layers.Conv2D(NUM_FILTERS,
    #                     kernel_size=1,
    #                     strides=1,
    #                     padding='SAME',
    #                     activation=None,
    #                     use_bias=False)(out)

    out = layers.GlobalAveragePooling2D()(out)
    action_logits = layers.Dense(num_act_charge_actions + num_loc_bond_actions,
                                 activation=None,
                                 use_bias=False)(out)
    mask = tf.cast(X_mask, action_logits.dtype)
    action_logits += (mask * -1e9)
    return X, X_mask, action_logits


def get_metrics():
    train_act_acc = tf.keras.metrics.CategoricalAccuracy(name="train_act_acc")
    val_act_acc = tf.keras.metrics.CategoricalAccuracy(name="val_act_acc")
    return train_act_acc, val_act_acc


def loss_func(y_true, y_pred):
    loss_obj = keras.losses.CategoricalCrossentropy(from_logits=True)
    loss = loss_obj(y_true, y_pred)
    return loss


def get_optimizer(finetune=False):
    lr = 0.001
    if finetune:
        lr = 0.00001
    lr_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [200000, 400000, 600000], [lr, lr / 10, lr / 50, lr / 100],
        name=None
    )
    opt_op = tf.keras.optimizers.Adam(learning_rate=lr_fn)
    return opt_op


class SeedGenerator(keras.Model):
    def compile(self, optimizer, loss_fn, metric_fn):
        super(SeedGenerator, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_act_acc, self.val_act_acc = metric_fn()

    def train_step(self, train_data):
        X, y = train_data

        # capture the scope of gradient
        with tf.GradientTape() as tape:
            logits = self(X, training=True)
            loss = self.loss_fn(y, logits)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # compute metrics keeping an moving average
        self.train_act_acc.update_state(y, logits)
        return {"train_act_acc": self.train_act_acc.result()}

    def test_step(self, val_data):
        X, y = val_data

        # predict
        logits = self(X, training=False)
        # compute metrics stateless
        self.val_act_acc.update_state(y, logits)
        return {"val_act_acc": self.val_act_acc.result()}

    @property
    def metrics(self):
        # clear metrics after every epoch
        return [self.train_act_acc, self.val_act_acc]


if __name__ == "__main__":
    freeze_support()
    ckpt_path = 'checkpoints/generator/'
    create_folder(ckpt_path)
    create_folder("generator_model")
    train_path = 'D:/seed_data/generator/train_data/train_batch/'
    val_path = 'D:/seed_data/generator/test_data/val_batch/'
    test_path = 'D:/seed_data/generator/test_data/test_batch/'
    callbacks = [tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                                    save_freq=1000,
                                                    save_weights_only=True,
                                                    monitor='train_act_acc',
                                                    mode='max',
                                                    save_best_only=True)]
    steps_per_epoch = len(glob.glob(train_path + 'Xy_*.pkl'))
    val_steps = len(glob.glob(val_path + 'Xy_*.pkl'))
    # train
    X, X_mask, action = core_model()
    model = SeedGenerator([X, X_mask], action)
    model.compile(optimizer=get_optimizer(),
                  loss_fn=loss_func,
                  metric_fn=get_metrics)
    save_model_to_json(model, "generator_model/generator_model.json")

    model.summary()
    model.fit(data_iterator(train_path),
              epochs=1,
              validation_data=data_iterator_test(val_path),
              validation_steps=10000,
              callbacks=callbacks,
              steps_per_epoch=20000)
    res = model.evaluate(data_iterator_test(test_path),
                         return_dict=True)

    # save trained model in two ways
    model.save("generator_full_model/", include_optimizer=False)
    model.save_weights("./generator_weights/generator")

    model_new = load_json_model("generator_model/generator_model.json",
                                SeedGenerator, "SeedGenerator")
    model.compile(optimizer=get_optimizer(),
                  loss_fn=loss_func,
                  metric_fn=get_metrics)
    model_new.load_weights("./generator_weights/generator")
    res = model_new.evaluate(data_iterator_test(test_path),
                             return_dict=True)
