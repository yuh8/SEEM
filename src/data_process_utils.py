import numpy as np
from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger
from .CONSTS import (FEATURE_DEPTH, MAX_NUM_ATOMS,
                     FEATURE_DEPTH, ATOM_LIST,
                     BOND_NAMES, CHARGES, MAX_NUM_ATOMS)
RDLogger.DisableLog('rdApp.*')


def has_valid_elements(mol):
    has_unknown_element = [atom.GetSymbol() not in ATOM_LIST for atom in mol.GetAtoms()]
    if sum(has_unknown_element) > 0:
        return False

    has_unknown_charge = [atom.GetFormalCharge() not in CHARGES for atom in mol.GetAtoms()]
    if sum(has_unknown_charge) > 0:
        return False

    has_radical = [atom.GetNumRadicalElectrons() > 0 for atom in mol.GetAtoms()]
    if sum(has_radical) > 0:
        return False

    return True


def is_smile_valid(smi):
    try:
        if Chem.MolFromSmiles(smi) is None:
            return False
    except:
        return False

    mol = Chem.MolFromSmiles(smi)
    if not has_valid_elements(mol):
        return False

    return True


def is_mol_valid(mol):
    try:
        Chem.MolToSmiles(mol)
    except:
        return False

    if not has_valid_elements(mol):
        return False

    return True


def standardize_smiles(smi):
    '''
    convert smiles to Kekulized form
    to convert aromatic bond to single/double/triple bond
    '''
    mol = Chem.MolFromSmiles(smi)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    smi = Chem.MolToSmiles(mol, isomericSmiles=False)
    return smi


def standardize_smiles_to_mol(smi):
    '''
    remove aromatic bonds in mol object
    '''
    smi = standardize_smiles(smi)
    mol = Chem.MolFromSmiles(smi)
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except:
        return mol
    return mol


def draw_smiles(smi, file_name):
    mol = Chem.MolFromSmiles(smi)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    Draw.MolToFile(mol, '{}.png'.format(file_name))


def smiles_to_graph(smi):
    if not is_smile_valid(smi):
        return None
    mol = standardize_smiles_to_mol(smi)
    smi_graph = np.zeros((MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH))
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
            atom = np.array(ATOM_LIST)[atom_feature_vec.argmax()]
            atom = Chem.Atom(atom)
            atom_idx = mol.AddAtom(atom)
            atoms[ii] = atom_idx
            charges[atom_idx] = np.array(CHARGES)[charge_feature_vec.argmax()]

    for ii in range(graph_dim):
        for jj in range(graph_dim):
            if ii >= jj:
                continue

            if (con_graph[ii, jj] == 3) and \
                    (ii in atoms.keys()) and (jj in atoms.keys()):
                bond_feature_vec = smi_graph[ii, jj, len(ATOM_LIST):-len(CHARGES)].astype(int)
                bond_type = BOND_NAMES[bond_feature_vec.argmax()]
                mol.AddBond(atoms[ii], atoms[jj], bond_type)

    mol = update_atom_property(mol, charges)
    smi = Chem.MolToSmiles(mol, isomericSmiles=False)
    return smi, mol


def get_initial_act_vec():
    num_atom_actions = len(ATOM_LIST)
    num_charge_actions = len(CHARGES)
    num_act_charge_actions = num_atom_actions * num_charge_actions
    # number of location to place atoms x num of bond types
    num_loc_bond_actions = (MAX_NUM_ATOMS - 1) * len(BOND_NAMES)
    action_vec = np.zeros(num_act_charge_actions + num_loc_bond_actions)
    return action_vec


def act_idx_to_vect(action_idx):
    action_vec = get_initial_act_vec()
    atom_act_idx, charge_act_idx = action_idx[0]
    loc_act_idx, bond_act_idx = action_idx[1]
    if atom_act_idx is not None and charge_act_idx is not None:
        dest_idx = atom_act_idx * len(CHARGES) + charge_act_idx
        action_vec[dest_idx] = 1

    if loc_act_idx is not None and bond_act_idx is not None:
        start_idx = len(ATOM_LIST) * len(CHARGES)
        dest_idx = start_idx + loc_act_idx * len(BOND_NAMES) + bond_act_idx
        action_vec[dest_idx] = 1
    return action_vec


def get_action_mask_from_state(state):
    pass


def decompose_smi_graph(smi_graph):
    con_graph = np.sum(smi_graph, axis=-1)
    gragh_dim = con_graph.shape[0]
    state = np.zeros((MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH))
    states = [deepcopy(state)]
    actions = []
    for jj in range(gragh_dim):
        # terminate
        if sum(smi_graph[jj, jj, :]) == 0:
            return actions, states[:-1]
        # adding atom and charge
        atom_act_idx = smi_graph[jj, jj, :len(ATOM_LIST)].argmax()
        charge_act_idx = smi_graph[jj, jj, -len(CHARGES):].argmax()
        loc_act_idx = None
        bond_act_idx = None
        action_idx = ((atom_act_idx, charge_act_idx), (loc_act_idx, bond_act_idx))
        actions.append(act_idx_to_vect(action_idx))
        state[jj, jj, :] = smi_graph[jj, jj, :]
        states.append(deepcopy(state))
        for ii in range(jj):
            charge_act_idx = None
            atom_act_idx = None
            if sum(smi_graph[ii, jj, :]) == 3:
                # adding connection bond
                loc_act_idx = ii
                bond_act_idx = smi_graph[ii, jj, len(ATOM_LIST):-len(CHARGES)].argmax()
                action_idx = ((atom_act_idx, charge_act_idx), (loc_act_idx, bond_act_idx))
                actions.append(act_idx_to_vect(action_idx))
                state[ii, jj, :] = smi_graph[ii, jj, :]
                # ensure symmetry
                state[jj, ii, :] = smi_graph[ii, jj, :]
                states.append(deepcopy(state))

    return actions, states[:-1]


if __name__ == "__main__":
    smi = 'CCC(C(O)C)CN'
    smiles_to_graph(smi)
