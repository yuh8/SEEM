import pickle
import glob
import numpy as np
import pandas as pd
import sparse as sp
from sklearn.model_selection import train_test_split
from src.data_process_utils import smiles_to_graph, decompose_smi_graph
from src.misc_utils import create_folder
from src.CONSTS import BATCH_SIZE


def get_train_val_test_data():
    df_data = pd.read_csv("D:/seed_data/small_mol_db.csv", sep=';', low_memory=False)
    create_folder('D:/seed_data/generator/train_data/train_batch/')
    create_folder('D:/seed_data/generator/test_data/test_batch/')
    create_folder('D:/seed_data/generator/test_data/val_batch/')

    # train, val, test split
    df_train, df_test \
        = train_test_split(df_data, test_size=0.1, random_state=43)

    df_train, df_val \
        = train_test_split(df_train, test_size=0.05, random_state=43)

    df_train.to_csv('D:/seed_data/generator/train_data/df_train.csv', index=False)
    df_test.to_csv('D:/seed_data/generator/test_data/df_test.csv', index=False)
    df_val.to_csv('D:/seed_data/generator/test_data/df_val.csv', index=False)


def save_data_batch(raw_data_path, dest_data_path):
    df_val = pd.read_csv(raw_data_path)
    x = []
    y = []
    x_mask = []
    batch = 0
    for _, row in df_val.iterrows():
        try:
            smi_graph, _ = smiles_to_graph(row.Smiles)
        except:
            continue
        actions, masks, states = decompose_smi_graph(smi_graph)
        x.extend(states)
        x_mask.extend(masks)
        y.extend(actions)
        if len(x) >= BATCH_SIZE:
            _X = sp.COO(np.stack(x[:BATCH_SIZE]))
            _X_mask = sp.COO(np.vstack(x_mask[:BATCH_SIZE]))
            _y = sp.COO(np.vstack(y[:BATCH_SIZE]))
            _data = ((_X, _X_mask), _y)
            with open(dest_data_path + 'Xy_{}.pkl'.format(batch), 'wb') as f:
                pickle.dump(_data, f)
            x = x[BATCH_SIZE:]
            x_mask = x_mask[BATCH_SIZE:]
            y = y[BATCH_SIZE:]
            batch += 1

    if x:
        _X = sp.COO(np.stack(x))
        _X_mask = sp.COO(np.vstack(x_mask))
        _y = sp.COO(np.vstack(y))
        _data = ((_X, _X_mask), _y)
        with open(dest_data_path + 'Xy_{}.pkl'.format(batch), 'wb') as f:
            pickle.dump(_data, f)


def data_iterator(data_path):
    num_files = len(glob.glob(data_path + 'Xy_*.pkl'))
    batch_nums = np.arange(num_files)
    while True:
        np.random.shuffle(batch_nums)
        for batch in batch_nums:
            f_name = data_path + 'Xy_{}.pkl'.format(batch)
            with open(f_name, 'rb') as handle:
                Xy = pickle.load(handle)

            X = (Xy[0][0].todense(), Xy[0][1].todense())
            y = Xy[1].todense()
            sample_nums = np.arange(y.shape[0])
            np.random.shuffle(sample_nums)
            X = (X[0][sample_nums, ...], X[1][sample_nums, ...])
            y = y[sample_nums, :]
            yield X, y


if __name__ == "__main__":
    get_train_val_test_data()
    save_data_batch('D:/seed_data/generator/train_data/df_train.csv',
                    'D:/seed_data/generator/train_data/train_batch/')
    save_data_batch('D:/seed_data/generator/test_data/df_val.csv',
                    'D:/seed_data/generator/test_data/val_batch/')
    save_data_batch('D:/seed_data/generator/test_data/df_test.csv',
                    'D:/seed_data/generator/test_data/test_batch/')
