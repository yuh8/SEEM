import numpy as np
import pandas as pd
import sparse as sp
from sklearn.model_selection import train_test_split
from src.misc_utils import create_folder


df = pd.read_csv("D:/seed_data/small_mol_db.csv", sep=';', low_memory=False)


def get_train_val_test_data():
    df_data = pd.read_csv("D:/seed_data/small_mol_db.csv", sep=';', low_memory=False)
    create_folder('D:/seed_data/generator/train_data/')
    create_folder('D:/seed_data/generator/train_data/')

    # train, val, test split
    df_train, df_test \
        = train_test_split(df_data, test_size=0.1, random_state=43)

    df_train, df_val \
        = train_test_split(df_train, test_size=0.05, random_state=43)

    df_train.to_csv('D:/seed_data/generator/train_data/df_train.csv', index=False)
    df_test.to_csv('D:/seed_data/generator/test_data/df_test.csv', index=False)
    df_val.to_csv('D:/seed_data/generator/test_data/df_val.csv', index=False)


# def get_val_data():
#     df_val = pd.read_csv('D:/seed_data/generator/test_data/df_val.csv')
#     x = []
#     y = []
#     for _, row in df_val.iterrows():
#         x.append(get_encoded_smi(row.X))
#         y.append(get_encoded_smi(row.Y))

#     _data = (np.vstack(x), np.vstack(y))
#     with open('D:/seed_data/generator/test_data/' + 'Xy_val.pkl', 'wb') as f:
#         pickle.dump(_data, f)
