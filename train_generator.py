import glob
import numpy as np
import tensorflow as tf
from datetime import date
from tensorflow import keras
from tensorflow.keras import layers, models
from multiprocessing import freeze_support
from src.embed_utils import conv2d_block, res_block
from src.misc_utils import create_folder, save_model_to_json
from src.CONSTS import (ATOM_LIST, CHARGES, BOND_NAMES,
                        MAX_NUM_ATOMS, FEATURE_DEPTH, BATCH_SIZE, VAL_BATCH_SIZE,
                        NUM_FILTERS, FILTER_SIZE, NUM_RES_BLOCKS)

today = str(date.today())


def core_model():
    # [BATCH, MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH + 1]
    X = layers.Input(shape=(MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH + 1))
    num_act_charge_actions = len(ATOM_LIST) * len(CHARGES)
    num_loc_bond_actions = (MAX_NUM_ATOMS - 1) * len(BOND_NAMES)
    # X_mask = layers.Input(shape=(num_act_charge_actions + num_loc_bond_actions))

    # [BATCH, MAX_NUM_ATOMS, MAX_NUM_ATOMS, NUM_FILTERS]
    out = conv2d_block(X, NUM_FILTERS, FILTER_SIZE)

    # [BATCH, MAX_NUM_ATOMS/16, MAX_NUM_ATOMS/16, NUM_FILTERS]
    major_block_size = NUM_RES_BLOCKS // 4
    for _ in range(major_block_size):
        out = res_block(out, NUM_FILTERS, FILTER_SIZE)
    out = layers.MaxPool2D(2, 2)(out)

    for _ in range(major_block_size):
        out = res_block(out, NUM_FILTERS, FILTER_SIZE * 2)
    out = layers.MaxPool2D(2, 2)(out)

    for _ in range(major_block_size):
        out = res_block(out, NUM_FILTERS, FILTER_SIZE * 2)
    out = layers.MaxPool2D(2, 2)(out)

    for _ in range(major_block_size):
        out = res_block(out, NUM_FILTERS, FILTER_SIZE * 4)
    out = layers.MaxPool2D(2, 2)(out)

    for _ in range(major_block_size):
        out = res_block(out, NUM_FILTERS, FILTER_SIZE * 4)
    out = layers.MaxPool2D(2, 2)(out)

    out = layers.GlobalMaxPooling2D()(out)
    action_logits = layers.Dense(num_act_charge_actions + num_loc_bond_actions,
                                 activation=None,
                                 use_bias=False)(out)
    return X, action_logits


def get_metrics():
    train_act_acc = tf.keras.metrics.CategoricalAccuracy(name="train_act_acc")
    val_act_acc = tf.keras.metrics.CategoricalAccuracy(name="val_act_acc")
    train_loss = tf.keras.metrics.CategoricalCrossentropy(name='train_loss',
                                                          from_logits=True)
    val_loss = tf.keras.metrics.CategoricalCrossentropy(name='val_loss',
                                                        from_logits=True)
    return train_act_acc, val_act_acc, train_loss, val_loss


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
        self.train_act_acc, self.val_act_acc, \
            self.train_loss, self.val_loss = metric_fn()

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
        self.train_loss.update_state(y, logits)
        return {"train_act_acc": self.train_act_acc.result(),
                "train_loss": self.train_loss.result()}

    def test_step(self, val_data):
        X, y = val_data

        # predict
        logits = self(X, training=False)
        # compute metrics stateless
        self.val_act_acc.update_state(y, logits)
        self.val_loss.update_state(y, logits)
        return {"val_act_acc": self.val_act_acc.result(),
                "val_loss": self.val_loss.result()}

    @property
    def metrics(self):
        # clear metrics after every epoch
        return [self.train_act_acc, self.val_act_acc,
                self.train_loss, self.val_loss]


def data_iterator_train():
    num_files = len(glob.glob(train_path + 'SA_*.npz'))
    batch_nums = np.arange(num_files)
    while True:
        np.random.shuffle(batch_nums)
        for batch in batch_nums:
            f_name = train_path + f'SA{batch}.npz'
            SA = np.load(f_name)

            X = SA['S']
            X[..., -1] /= 8
            y = SA['A']
            yield X, y


def data_iterator_val():
    num_files = len(glob.glob(val_path + 'SA_*.npz'))
    batch_nums = np.arange(num_files)
    while True:
        np.random.shuffle(batch_nums)
        for batch in batch_nums:
            f_name = train_path + f'SA{batch}.npz'
            SA = np.load(f_name)

            X = SA['S']
            X[..., -1] /= 8
            y = SA['A']
            yield X, y


def data_iterator_test():
    num_files = len(glob.glob(test_path + 'SA_*.npz'))
    batch_nums = np.arange(num_files)
    for batch in batch_nums:
        f_name = train_path + f'SA{batch}.npz'
        SA = np.load(f_name)

        X = SA['S']
        X[..., -1] /= 8
        y = SA['A']
        yield X, y


def _fixup_shape(x, y):
    x.set_shape([None, MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH + 1])
    y.set_shape([None, len(ATOM_LIST) * len(CHARGES) + (MAX_NUM_ATOMS - 1) * len(BOND_NAMES)])
    return x, y


if __name__ == "__main__":
    freeze_support()
    ckpt_path = f'checkpoints/generator_{today}/'
    create_folder(ckpt_path)
    create_folder(f"generator_model_chembl_{today}")
    train_path = '/mnt/seed_data/generator/train_data/train_batch_zinc/'
    val_path = '/mnt/seed_data/generator/test_data/val_batch_zinc/'
    test_path = '/mnt/seed_data/generator/test_data/test_batch_zinc/'
    callbacks = [tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                                    save_freq=1000,
                                                    save_weights_only=True,
                                                    monitor='categorical_accuracy',
                                                    mode='max',
                                                    save_best_only=True)]
    steps_per_epoch = len(glob.glob(train_path + 'SA_*.npz')) // BATCH_SIZE
    val_steps = len(glob.glob(val_path + 'SA_*.npz')) // VAL_BATCH_SIZE
    # train
    X, action_logits = core_model()
    model = keras.Model(inputs=X, outputs=action_logits)
    model.compile(optimizer=get_optimizer(),
                  loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=[keras.metrics.CategoricalAccuracy()])
    save_model_to_json(model, f"generator_model_chembl_{today}/generator_model_chembl.json")

    model.summary()

    train_dataset = tf.data.Dataset.from_generator(
        data_iterator_train,
        output_types=(tf.float32, tf.float32),
        output_shapes=((MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH + 1),
                       None, len(ATOM_LIST) * len(CHARGES) + (MAX_NUM_ATOMS - 1) * len(BOND_NAMES)))

    train_dataset = train_dataset.shuffle(buffer_size=1000, seed=0,
                                          reshuffle_each_iteration=True)
    train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True).map(_fixup_shape)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_generator(
        data_iterator_val,
        output_types=(tf.float32, tf.float32),
        output_shapes=((MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH),
                       None, len(ATOM_LIST) * len(CHARGES) + (MAX_NUM_ATOMS - 1) * len(BOND_NAMES)))
    val_dataset = val_dataset.batch(VAL_BATCH_SIZE, drop_remainder=True).map(_fixup_shape)
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_generator(
        data_iterator_test,
        output_types=(tf.float32, tf.float32),
        output_shapes=((MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH),
                       None, len(ATOM_LIST) * len(CHARGES) + (MAX_NUM_ATOMS - 1) * len(BOND_NAMES)))
    test_dataset = test_dataset.batch(VAL_BATCH_SIZE, drop_remainder=True).map(_fixup_shape)
    test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    model.fit(train_dataset,
              epochs=4,
              validation_data=val_dataset,
              validation_steps=val_steps,
              callbacks=callbacks,
              steps_per_epoch=steps_per_epoch)
    res = model.evaluate(test_dataset,
                         return_dict=True)

    # save trained model in two ways
    model.save(f"generator_full_model_{today}/")
    model_new = models.load_model(f"generator_full_model_chembl_{today}/")
    res = model_new.evaluate(test_dataset,
                             return_dict=True)
