from rdkit import Chem


MAX_NUM_ATOMS = 100
ATOM_LIST = [
    'H', 'B', 'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I',
    'Se', 'Na', 'Si'
]

BOND_DICT = Chem.rdchem.BondType.values
BOND_NAMES = list(BOND_DICT.values())
CHARGES = [-2, -1, 0, 1, 2]

FEATURE_DEPTH = len(ATOM_LIST) + len(BOND_NAMES) + len(CHARGES)
