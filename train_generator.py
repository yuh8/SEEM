import glob
import tensorflow as tf
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

    # [BATCH, MAX_NUM_ATOMS/32, MAX_NUM_ATOMS/32, ACTION_SIZE]
    out = layers.Conv2D(num_act_charge_actions + num_loc_bond_actions,
                        kernel_size=1,
                        strides=1,
                        padding='SAME',
                        activation=None,
                        use_bias=False)(out)

    action_logits = tf.reduce_max(out, axis=(1, 2))
    mask = tf.cast(X_mask, action_logits.dtype)
    action_logits -= mask * 1e-9
    action = tf.nn.softmax(action_logits, axis=-1)
    return X, X_mask, action


def get_metrics():
    train_auc = tf.keras.metrics.AUC(name="train_auc")
    val_auc = tf.keras.metrics.AUC(name="val_auc")
    return train_auc, val_auc


def loss_func(y_true, y_pred):
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
    loss = tf.reduce_mean(loss)
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
                                                    monitor='loss',
                                                    mode='min',
                                                    save_best_only=True)]
    steps_per_epoch = len(glob.glob(train_path + 'Xy_*.pkl'))
    val_steps = len(glob.glob(val_path + 'Xy_*.pkl'))
    # train
    X, X_mask, action = core_model()
    model = tf.keras.Model([X, X_mask], action)
    model.compile(optimizer=get_optimizer(),
                  loss=loss_func)

    model.summary()
    model.fit(data_iterator(train_path),
              epochs=1,
              validation_data=data_iterator_test(val_path),
              validation_steps=20,
              callbacks=callbacks,
              steps_per_epoch=300)
    res = model.evaluate(data_iterator_test(test_path),
                         return_dict=True)

    # save trained model in two ways
    model.save("generator_full_model/", include_optimizer=False)
    model.save_weights("./generator_weights/generator")

    save_model_to_json(model, "generator_model/generator_model.json")
    model_new = load_json_model("generator_model/generator_model.json")
    model_new.compile(optimizer=get_optimizer(),
                      loss=loss_func)
    model_new.load_weights("./generator_weights/generator")
    res = model_new.evaluate(data_iterator_test(test_path),
                             return_dict=True)
