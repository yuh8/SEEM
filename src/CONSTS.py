from rdkit import Chem


MAX_NUM_ATOMS = 160
ATOM_LIST = [
    'H', 'B', 'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I',
    'Na', 'K', 'Ca', 'Li', 'Rb', 'Se', 'Si', 'Te', 'Mg', 'Al',
    'Ba', 'Be', 'As', 'Sr', 'Zn', 'Ag', 'Bi'
]

BOND_DICT = Chem.rdchem.BondType.values
BOND_NAMES = list(BOND_DICT.values())[:4]
CHARGES = [-4, -3, -2, -1, 0, 1, 2, 3, 4]

FEATURE_DEPTH = len(ATOM_LIST) + len(BOND_NAMES) + len(CHARGES)
BATCH_SIZE = 128
