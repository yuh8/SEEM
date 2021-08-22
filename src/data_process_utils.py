import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from .CONSTS import FEATURE_DEPTH, MAX_NUM_ATOMS, FEATURE_DEPTH, ATOM_LIST, BOND_NAMES
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
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    smi = Chem.CanonSmiles(smi)
    return smi


def smiles_to_graph(smi):
    if not is_smile_valid(smi):
        return None
    smi = standardize_smiles(smi)
    smi_graph = np.zeros((MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH))
    mol = Chem.MolFromSmiles(smi)
    elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
    for ii, ei in enumerate(elements):
        for jj, ej in enumerate(elements):
            feature_vec = np.zeros(FEATURE_DEPTH)
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

    return smi_graph


def graph_to_smiles(smi_graph):
    connection_graph = np.sum(smi_graph, axis=-1)
    graph_dim = connection_graph.shape[0]
    mol = Chem.RWMol()
    atoms = {}
    for ii in range(graph_dim):
        if connection_graph[ii, ii] == 2:
            diag_feature_vec = smi_graph[ii, ii, :len(ATOM_LIST)]
            atom = np.array(ATOM_LIST)[diag_feature_vec == 2][0]
            atoms[ii] = mol.AddAtom(Chem.Atom(atom))

    for ii in range(graph_dim):
        for jj in range(graph_dim):
            if ii >= jj:
                continue

            if connection_graph[ii, jj] == 3:
                bond_feature_vec = smi_graph[ii, jj, len(ATOM_LIST):].astype(int)
                bond_type = BOND_NAMES[np.where(bond_feature_vec)[0][0]]
                mol.AddBond(atoms[ii], atoms[jj], bond_type)

    mol = mol.GetMol()
    smi = Chem.MolToSmiles(mol)
    return smi


if __name__ == "__main__":
    smi = 'CCC(C(O)C)CN'
    smiles_to_graph(smi)
