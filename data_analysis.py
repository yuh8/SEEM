import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from src.data_process_utils import is_smile_valid, standardize_smiles
from src.CONSTS import (ATOM_LIST, BOND_NAMES)
RDLogger.DisableLog('rdApp.*')


def get_bond_sum_for_each_atom(smi):
    if not is_smile_valid(smi):
        return None
    smi = standardize_smiles(smi)
    mol = Chem.MolFromSmiles(smi)
    try:
        Chem.Kekulize(mol)
    except:
        return None
    elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
    explicit_valence = [atom.GetExplicitValence() for atom in mol.GetAtoms()]
    bond_summary_matrix = np.zeros((len(ATOM_LIST), len(BOND_NAMES) + 1))

    for ii, at in enumerate(ATOM_LIST):
        if at in elements:
            for jj, ej in enumerate(elements):
                if ej == at:
                    bond_summary_matrix[ii, -1] += explicit_valence[jj]
                    for kk, _ in enumerate(elements):
                        if jj == kk:
                            continue
                        if mol.GetBondBetweenAtoms(jj, kk) is not None:
                            bond_name = mol.GetBondBetweenAtoms(jj, kk).GetBondType()

                            bond_idx = BOND_NAMES.index(bond_name)
                            bond_summary_matrix[ii, bond_idx] += 1

    return bond_summary_matrix


if __name__ == "__main__":
    df_data = pd.read_csv("D:/seed_data/small_mol_db.csv", sep=';', low_memory=False)
    bond_summary_matrix = np.zeros((len(ATOM_LIST), len(BOND_NAMES) + 1))
    count = 0
    for _, row in df_data.iterrows():
        matrix = get_bond_sum_for_each_atom(row.Smiles)
        if matrix is None:
            continue

        bond_summary_matrix += matrix
        count += 1
        if count % 10000 == 0:
            print(bond_summary_matrix.sum(0))
