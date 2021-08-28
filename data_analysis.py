import numpy as np
import pandas as pd
from collections import Counter
from rdkit import RDLogger
from src.data_process_utils import is_smile_valid, standardize_smiles, standardize_smiles_to_mol
from src.CONSTS import (ATOM_LIST, BOND_NAMES)
RDLogger.DisableLog('rdApp.*')


def get_bond_sum_for_each_atom(smi):
    if not is_smile_valid(smi):
        return None
    smi = standardize_smiles(smi)
    mol = standardize_smiles_to_mol(smi)
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


def get_unique_elements_stats():
    df_data = pd.read_csv("D:/seed_data/small_mol_db.csv", sep=';', low_memory=False)
    elements = []
    charges = []
    for idx, row in df_data.iterrows():
        smi = row.Smiles
        try:
            smi = standardize_smiles(smi)
            mol = standardize_smiles_to_mol(smi)
            _elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
            _charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
        except:
            continue
        elements.extend(_elements)
        charges.extend(_charges)
        print(idx)
    print(Counter(elements))
    print(Counter(charges))
    return Counter(elements)


if __name__ == "__main__":
    get_unique_elements_stats()
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
