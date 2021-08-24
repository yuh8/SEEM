import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from .CONSTS import (FEATURE_DEPTH, MAX_NUM_ATOMS,
                     FEATURE_DEPTH, ATOM_LIST,
                     BOND_NAMES, CHARGES)
RDLogger.DisableLog('rdApp.*')


def is_smile_valid(smi):
    if Chem.MolFromSmiles(smi) is None:
        return False
    return True


def is_mol_valid(mol):
    try:
        Chem.MolToSmiles(mol)
    except:
        return False
    return True


def standardize_smiles(smi):
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False)
    return smi


def smiles_to_graph(smi):
    if not is_smile_valid(smi):
        return None
    smi = standardize_smiles(smi)
    smi_graph = np.zeros((MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH))
    mol = Chem.MolFromSmiles(smi)
    elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
    charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    for ii, ei in enumerate(elements):
        for jj, ej in enumerate(elements):
            feature_vec = np.zeros(FEATURE_DEPTH)
            if ii == jj:
                charge_idx = CHARGES.index(charges[ii]) - len(CHARGES)
                feature_vec[charge_idx] = 1

            if ii > jj:
                continue

            atom_idx_ii = ATOM_LIST.index(ei)
            atom_idx_jj = ATOM_LIST.index(ej)
            feature_vec[atom_idx_ii] += 1
            feature_vec[atom_idx_jj] += 1
            if mol.GetBondBetweenAtoms(ii, jj) is not None:
                bond_name = mol.GetBondBetweenAtoms(ii, jj).GetBondType()
                bond_idx = BOND_NAMES.index(bond_name)
                bond_feature_idx = len(ATOM_LIST) + bond_idx
                feature_vec[bond_feature_idx] = 1
            smi_graph[ii, jj, :] = feature_vec
            smi_graph[jj, ii, :] = feature_vec

    return smi_graph, charges


def update_atom_property(mol, charges):
    for key in charges:
        mol.GetAtomWithIdx(key).SetFormalCharge(int(charges[key]))
    return mol


def graph_to_smiles(smi_graph, charges):
    con_graph = np.sum(smi_graph, axis=-1)
    graph_dim = con_graph.shape[0]
    mol = Chem.RWMol()
    atoms = {}
    charges = {}
    for ii in range(graph_dim):
        if con_graph[ii, ii] == 3:
            atom_feature_vec = smi_graph[ii, ii, :len(ATOM_LIST)]
            charge_feature_vec = smi_graph[ii, ii, -len(CHARGES):]
            atom = np.array(ATOM_LIST)[atom_feature_vec == 2][0]
            atom = Chem.Atom(atom)
            atom_idx = mol.AddAtom(atom)
            atoms[ii] = atom_idx
            charges[atom_idx] = np.array(CHARGES)[charge_feature_vec == 1][0]

    for ii in range(graph_dim):
        for jj in range(graph_dim):
            if ii >= jj:
                continue

            if (con_graph[ii, jj] == 3) and \
                    (ii in atoms.keys()) and (jj in atoms.keys()):
                bond_feature_vec = smi_graph[ii, jj, len(ATOM_LIST):-len(CHARGES)].astype(int)
                bond_type = BOND_NAMES[np.where(bond_feature_vec)[0][0]]
                mol.AddBond(atoms[ii], atoms[jj], bond_type)

    mol = update_atom_property(mol, charges)
    smi = Chem.MolToSmiles(mol, isomericSmiles=False)
    return smi, mol


if __name__ == "__main__":
    smi = 'CCC(C(O)C)CN'
    smiles_to_graph(smi)
