from rdkit import Chem


MAX_NUM_ATOMS = 160

ATOM_FREQ_DICT = {'C': 35467421, 'N': 5330638, 'O': 5173387, 'F': 709638, 'S': 665573,
                  'Cl': 443265, 'Br': 87847, 'P': 27831, 'I': 13015,
                  'Na': 7286, 'B': 3144, 'Si': 2140, 'Se': 1867, 'K': 1113,
                  'Li': 493, 'As': 206, 'Te': 105, 'Ca': 103, 'Zn': 73,
                  'Mg': 72, 'Al': 45, 'Ag': 20, 'Sr': 12, 'Ba': 11,
                  'Cs': 5, 'Rb': 4, 'H': 4, 'Bi': 4,
                  'Ra': 4, 'Xe': 4,
                  'Be': 2, 'Kr': 2, 'He': 1}

CHARGE_FREQ_DICT = {0: 47681162, 1: 134725, -1: 118658, 3: 532, 2: 252, -2: 3, -3: 2, -4: 1}

ATOM_LIST = ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'P', 'I', 'Na', 'B', 'Si', 'Se', 'K']

# https://en.wikipedia.org/wiki/Valence_(chemistry)
ATOM_MAX_VALENCE = [1, 4, 5, 2, 1, 6, 1, 7, 5, 7, 1, 3, 4, 6, 1]

BOND_DICT = Chem.rdchem.BondType.values
BOND_NAMES = list(BOND_DICT.values())[:4]
CHARGES = [-1, 0, 1, 2, 3]

FEATURE_DEPTH = len(ATOM_LIST) + len(BOND_NAMES) + len(CHARGES)
BATCH_SIZE = 128
