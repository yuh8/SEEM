import pandas as pd
from rdkit import Chem
from src.data_process_utils import standardize_smiles_error_handle


df_base = pd.read_csv("D:/seed_data/generator/train_data/df_train.csv", low_memory=False)[:1000000]
gen_samples_df = []
count = 0
for _, row in df_base.iterrows():
    gen_sample = {}
    try:
        smi = standardize_smiles_error_handle(row.Smiles)
        mol = Chem.MolFromSmiles(smi)
        elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
        gen_sample["Smiles"] = smi
        gen_sample["NumAtoms"] = len(elements)
    except:
        continue
    gen_samples_df.append(gen_sample)
    count += 1
    print("{} / {} done".format(count, df_base.shape[0]))


gen_samples_df = pd.DataFrame(gen_samples_df)
gen_samples_df.to_csv('molecules_chembl.csv', index=False)
