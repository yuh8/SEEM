from rdkit import Chem


MAX_NUM_ATOMS = 48
ATOM_LIST = [
    'H', 'B', 'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I',
    '[Se]', '[Na+]', '[Si]'
]

BOND_DICT = Chem.rdchem.BondType.values
BOND_NAMES = list(BOND_DICT.values())

FEATURE_DEPTH = len(ATOM_LIST) + len(BOND_NAMES)
