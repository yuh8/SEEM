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
    df_data = pd.read_csv("/mnt/small_mol_db.csv", sep=';', low_memory=False)
    create_folder('/mnt/seed_data/generator/train_data/train_batch/')
    create_folder('/mnt/seed_data/generator/test_data/test_batch/')
    create_folder('/mnt/seed_data/generator/test_data/val_batch/')

    # train, val, test split
    df_train, df_test \
        = train_test_split(df_data, test_size=0.01, random_state=43)

    df_train, df_val \
        = train_test_split(df_train, test_size=0.01, random_state=43)

    df_train.to_csv('/mnt/seed_data/generator/train_data/df_train.csv', index=False)
    df_test.to_csv('/mnt/seed_data/generator/test_data/df_test.csv', index=False)
    df_val.to_csv('/mnt/seed_data/generator/test_data/df_val.csv', index=False)


def save_data_batch(raw_data_path, dest_data_path):
    df = pd.read_csv(raw_data_path)
    df = df.sample(frac=1).reset_index(drop=True)
    x = []
    y = []
    batch = 0
    for idx, row in df.iterrows():
        try:
            smi_graph, valences = smiles_to_graph(row.Smiles)
            if smi_graph is None:
                continue
        except:
            continue

        actions, states = decompose_smi_graph(smi_graph, valences)
        for idx, state in enumerate(states):
            np.savez_compressed(dest_data_path + f'SA_{batch}', S=state, A=actions[idx])
            batch += 1
            if batch == 100000000:
                break
        else:
            continue
        break


if __name__ == "__main__":
    get_train_val_test_data()
    save_data_batch('/mnt/seed_data/generator/train_data/df_train.csv',
                    '/mnt/seed_data/generator/train_data/train_batch/')
    save_data_batch('/mnt/seed_data/generator/test_data/df_val.csv',
                    '/mnt/seed_data/generator/test_data/val_batch/')
    save_data_batch('/mnt/seed_data/generator/test_data/df_test.csv',
                    '/mnt/seed_data/generator/test_data/test_batch/')
