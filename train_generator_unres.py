import glob
import tensorflow as tf
from datetime import date
from tensorflow import keras
from tensorflow.keras import layers, models
from multiprocessing import freeze_support
from data_gen import data_iterator, data_iterator_test
from src.embed_utils import conv2d_block, res_block
from src.misc_utils import create_folder, save_model_to_json
from src.CONSTS import (ATOM_LIST, CHARGES, BOND_NAMES,
                        MAX_NUM_ATOMS, FEATURE_DEPTH,
                        NUM_FILTERS, FILTER_SIZE, NUM_RES_BLOCKS)

today = str(date.today())


def core_model():
    # [BATCH, MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH + 1]
    X = layers.Input(shape=(MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH + 1))
    num_act_charge_actions = len(ATOM_LIST) * len(CHARGES)
    num_loc_bond_actions = MAX_NUM_ATOMS * MAX_NUM_ATOMS * len(BOND_NAMES)
    # X_mask = layers.Input(shape=(num_act_charge_actions + num_loc_bond_actions))

    # [BATCH, MAX_NUM_ATOMS, MAX_NUM_ATOMS, NUM_FILTERS]
    out = conv2d_block(X, NUM_FILTERS, FILTER_SIZE)

    # [BATCH, MAX_NUM_ATOMS/16, MAX_NUM_ATOMS/16, NUM_FILTERS]
    major_block_size = NUM_RES_BLOCKS // 4
    for _ in range(major_block_size):
        out = res_block(out, NUM_FILTERS, FILTER_SIZE)
    out = layers.MaxPool2D(2, 2)(out)

    for _ in range(major_block_size):
        out = res_block(out, NUM_FILTERS, FILTER_SIZE)
    out = layers.MaxPool2D(2, 2)(out)

    for _ in range(major_block_size):
        out = res_block(out, NUM_FILTERS, FILTER_SIZE)
    out = layers.MaxPool2D(2, 2)(out)

    for _ in range(major_block_size):
        out = res_block(out, NUM_FILTERS, FILTER_SIZE)
    out = layers.MaxPool2D(2, 2)(out)

    for _ in range(major_block_size):
        out = res_block(out, NUM_FILTERS, FILTER_SIZE)
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


if __name__ == "__main__":
    freeze_support()
    ckpt_path = 'checkpoints/generator_{}/'.format(today)
    create_folder(ckpt_path)
    create_folder("generator_model_random_mol_unres_{}".format(today))
    train_path = 'D:/seed_data/generator/train_data/train_batch/'
    val_path = 'D:/seed_data/generator/test_data/val_batch/'
    test_path = 'D:/seed_data/generator/test_data/test_batch/'
    log_dir = "logs/random_mol_unres_{}/".format(today)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks = [tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                                    save_freq=1000,
                                                    save_weights_only=True,
                                                    monitor='categorical_accuracy',
                                                    mode='max',
                                                    save_best_only=True),
                 tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=100)]
    steps_per_epoch = len(glob.glob(train_path + 'Xy_*.pkl'))
    val_steps = len(glob.glob(val_path + 'Xy_*.pkl'))
    # train
    X, action_logits = core_model()
    model = keras.Model(inputs=X, outputs=action_logits)
    model.compile(optimizer=get_optimizer(),
                  loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=[keras.metrics.CategoricalAccuracy()])
    save_model_to_json(model, "generator_model_random_mol_unres_{}/generator_model_random_mol_unres.json".format(today))

    model.summary()
    model.fit(data_iterator(train_path),
              epochs=4,
              validation_data=data_iterator(val_path),
              validation_steps=val_steps,
              callbacks=callbacks,
              steps_per_epoch=steps_per_epoch)
    res = model.evaluate(data_iterator_test(test_path),
                         return_dict=True)

    # save trained model in two ways
    model.save("generator_full_model_random_mol_unres_{}/".format(today))
    model_new = models.load_model("generator_full_model_random_mol_unres_{}/".format(today))
    res = model_new.evaluate(data_iterator_test(test_path),
                             return_dict=True)
