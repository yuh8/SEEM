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
    df_data = pd.read_csv("D:/seed_data/Zinc_250k.smi", sep=" ", header=None)
    df_data.columns = ['Smiles']
    create_folder('D:/seed_data/generator/train_data/train_batch_zinc/')
    create_folder('D:/seed_data/generator/test_data/test_batch_zinc/')
    create_folder('D:/seed_data/generator/test_data/val_batch_zinc/')

    # train, val, test split
    df_train, df_test \
        = train_test_split(df_data, test_size=int(1e4), random_state=43)

    df_train, df_val \
        = train_test_split(df_train, test_size=int(1e4), random_state=43)

    df_train.to_csv('D:/seed_data/generator/train_data/df_train_zinc.csv', index=False)
    df_test.to_csv('D:/seed_data/generator/test_data/df_test_zinc.csv', index=False)
    df_val.to_csv('D:/seed_data/generator/test_data/df_val_zinc.csv', index=False)


def save_data_batch(raw_data_path, dest_data_path):
    df = pd.read_csv(raw_data_path)
    df = df.sample(frac=1).reset_index(drop=True)
    x = []
    y = []
    batch = 0
    for idx, row in df.iterrows():
        try:
            smi_graph = smiles_to_graph(row.Smiles)
            if smi_graph is None:
                continue
        except:
            continue

        actions, states = decompose_smi_graph(smi_graph)
        x.extend(states)
        y.extend(actions)
        while len(x) > BATCH_SIZE:
            _X = sp.COO(np.stack(x[:BATCH_SIZE]))
            _y = sp.COO(np.vstack(y[:BATCH_SIZE]))
            _data = (_X, _y)
            with open(dest_data_path + 'Xy_{}.pkl'.format(batch), 'wb') as f:
                pickle.dump(_data, f)
            x = x[BATCH_SIZE:]
            y = y[BATCH_SIZE:]
            batch += 1
            if batch >= 1000000:
                break
        if batch >= 1000000:
            break
        print('{}/{} molecules done'.format(idx, df.shape[0]))

    if x:
        _X = sp.COO(np.stack(x))
        _y = sp.COO(np.vstack(y))
        _data = (_X, _y)
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

            X = Xy[0].todense()
            y = Xy[1].todense()
            sample_nums = np.arange(y.shape[0])
            np.random.shuffle(sample_nums)
            yield X[sample_nums, ...], y[sample_nums, :]


def data_iterator_test(test_path):
    for f_name in glob.glob(test_path + 'Xy_*.pkl'):
        with open(f_name, 'rb') as handle:
            Xy = pickle.load(handle)
        yield Xy[0].todense(), Xy[1].todense()


if __name__ == "__main__":
    get_train_val_test_data()
    save_data_batch('D:/seed_data/generator/train_data/df_train_zinc.csv',
                    'D:/seed_data/generator/train_data/train_batch_zinc/')
    save_data_batch('D:/seed_data/generator/test_data/df_val_zinc.csv',
                    'D:/seed_data/generator/test_data/val_batch_zinc/')
    save_data_batch('D:/seed_data/generator/test_data/df_test_zinc.csv',
                    'D:/seed_data/generator/test_data/test_batch_zinc/')
