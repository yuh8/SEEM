import numpy as np
import pandas as pd
from copy import deepcopy
from multiprocessing import Pool, freeze_support
from tensorflow import keras
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors
from scipy.special import softmax
from src.data_process_utils import (get_action_mask_from_state,
                                    standardize_smiles_error_handle,
                                    get_last_col_with_atom, draw_smiles,
                                    get_initial_act_vec, graph_to_smiles)
from src.misc_utils import create_folder, load_json_model
from src.CONSTS import (BOND_NAMES, MAX_NUM_ATOMS,
                        MIN_NUM_ATOMS, FEATURE_DEPTH,
                        ATOM_MAX_VALENCE,
                        ATOM_LIST, CHARGES)


def check_validity(mol):
    try:
        Chem.SanitizeMol(mol,
                         sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True
    except ValueError:
        return False


def sample_action(action_logits, state, T=1):
    action_mask = get_action_mask_from_state(state)
    action_logits = np.where(action_mask, -1e9, action_logits)
    action_probs = softmax(action_logits / T)
    act_vec = get_initial_act_vec()
    action_size = act_vec.shape[0]
    action_idx = np.random.choice(action_size, p=action_probs)
    return action_idx


def update_state_with_action(action_logits, state, num_atoms):
    is_terminate = False
    num_act_charge_actions = len(ATOM_LIST) * len(CHARGES)
    max_remaining_valence = state[:, :, -1].sum(-1).max()
    col = get_last_col_with_atom(state)
    col_has_atom = (state[:, col, :-1].sum(-1) > 0).any()
    if (max_remaining_valence < 2) and (col > 0):
        is_terminate = True
        return state, is_terminate

    feature_vec = np.zeros(FEATURE_DEPTH)
    action_idx = sample_action(action_logits, state)

    if action_idx <= num_act_charge_actions:
        if col >= num_atoms:
            is_terminate = True
            return state, is_terminate

        atom_idx = action_idx // len(CHARGES)
        charge_idx = action_idx % len(CHARGES) - len(CHARGES)
        feature_vec[atom_idx] = 2
        feature_vec[charge_idx] = 1

        if col_has_atom:
            state[col + 1, col + 1, :-1] = feature_vec
            # once an atom is added, initialize with full valence
            state[col + 1, col + 1, -1] = ATOM_MAX_VALENCE[atom_idx]
        else:
            state[col, col, :-1] = feature_vec
            state[col, col, -1] = ATOM_MAX_VALENCE[atom_idx]
    else:
        row = (action_idx - num_act_charge_actions) // len(BOND_NAMES)
        bond_idx = (action_idx - num_act_charge_actions) % len(BOND_NAMES)
        bond_feature_idx = len(ATOM_LIST) + bond_idx
        atom_idx_row = state[row, row, :len(ATOM_LIST)].argmax()
        atom_idx_col = state[col, col, :len(ATOM_LIST)].argmax()
        feature_vec[bond_feature_idx] = 1
        feature_vec[atom_idx_row] += 1
        feature_vec[atom_idx_col] += 1
        state[row, col, :-1] = feature_vec
        state[col, row, :-1] = feature_vec
        state[row, row, -1] -= bond_idx
        state[col, col, -1] -= bond_idx

    return state, is_terminate


def update_state_with_action_validity_check(action_logits, state, num_atoms):
    is_terminate = False
    num_act_charge_actions = len(ATOM_LIST) * len(CHARGES)
    max_remaining_valence = state[:, :, -1].sum(-1).max()
    col = get_last_col_with_atom(state)
    col_has_atom = (state[:, col, :-1].sum(-1) > 0).any()
    if (max_remaining_valence < 2) and (col > 0):
        is_terminate = True
        return state, is_terminate

    valid = False
    while not valid:
        feature_vec = np.zeros(FEATURE_DEPTH)
        try:
            action_idx = sample_action(action_logits, state)
        except:
            is_terminate = True
            return state, is_terminate
        state_new = deepcopy(state)
        if action_idx <= num_act_charge_actions:
            if col >= num_atoms:
                is_terminate = True
                return state, is_terminate

            atom_idx = action_idx // len(CHARGES)
            charge_idx = action_idx % len(CHARGES) - len(CHARGES)
            feature_vec[atom_idx] = 2
            feature_vec[charge_idx] = 1

            if col_has_atom:
                state_new[col + 1, col + 1, :-1] = feature_vec
                # once an atom is added, initialize with full valence
                state_new[col + 1, col + 1, -1] = ATOM_MAX_VALENCE[atom_idx]
            else:
                state_new[col, col, :-1] = feature_vec
                state_new[col, col, -1] = ATOM_MAX_VALENCE[atom_idx]
            return state_new, is_terminate
        else:
            row = (action_idx - num_act_charge_actions) // len(BOND_NAMES)
            bond_idx = (action_idx - num_act_charge_actions) % len(BOND_NAMES)
            bond_feature_idx = len(ATOM_LIST) + bond_idx
            atom_idx_row = state_new[row, row, :len(ATOM_LIST)].argmax()
            atom_idx_col = state_new[col, col, :len(ATOM_LIST)].argmax()
            feature_vec[bond_feature_idx] = 1
            feature_vec[atom_idx_row] += 1
            feature_vec[atom_idx_col] += 1
            state_new[row, col, :-1] = feature_vec
            state_new[col, row, :-1] = feature_vec
            state_new[row, row, -1] -= bond_idx
            state_new[col, col, -1] -= bond_idx
            mol = graph_to_smiles(state_new[:, :, :-1], return_mol=True)
            valid = check_validity(mol)
            if not valid:
                action_logits[action_idx] = -1e9

    return state_new, is_terminate


def generate_smiles(model, gen_idx):
    num_atoms = np.random.randint(MIN_NUM_ATOMS, MAX_NUM_ATOMS)
    state = np.zeros((MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH + 1))
    is_terminate = False

    while not is_terminate:
        X_in = deepcopy(state[np.newaxis, ...])
        X_in[..., -1] /= 8
        action_logits = model(X_in, training=False).numpy()[0]
        state, is_terminate = update_state_with_action_validity_check(action_logits, state, num_atoms)
        # state, is_terminate = update_state_with_action(action_logits, state, num_atoms)

    smi_graph = state[..., :-1]
    smi = graph_to_smiles(smi_graph)
    smi = _canonicalize_smiles(smi)
    draw_smiles(smi, "gen_samples/gen_sample_{}".format(gen_idx))
    return smi, num_atoms


def _canonicalize_smiles(smi):
    try:
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    except:
        return np.nan
    return smi


def compute_unique_score():
    gen_samples_df = pd.read_csv("generated_molecules.csv")
    gen_samples_df.loc[:, 'CanSmiles'] = gen_samples_df.Smiles.map(_canonicalize_smiles)
    gen_samples_df = gen_samples_df[~gen_samples_df.CanSmiles.isnull()]
    num_uniques = gen_samples_df.CanSmiles.unique().shape[0]
    unique_score = np.round(num_uniques / gen_samples_df.shape[0], 3)
    print("Unique score = {}".format(unique_score))
    return unique_score


def compute_novelty_score():
    gen_samples_df = pd.read_csv("generated_molecules.csv")
    train_samples_df = pd.read_csv('D:/seed_data/generator/train_data/df_train.csv')
    gen_samples_df.loc[:, 'CanSmiles'] = gen_samples_df.Smiles.map(_canonicalize_smiles)
    train_samples_df.loc[:, 'CanSmiles'] = train_samples_df.Smiles.map(_canonicalize_smiles)
    gen_samples_df = gen_samples_df[~gen_samples_df.CanSmiles.isnull()]
    train_samples_df = train_samples_df[~train_samples_df.CanSmiles.isnull()]
    gen_smi_unique = list(gen_samples_df.CanSmiles.unique())
    train_smi_unique = list(train_samples_df.CanSmiles.unique())
    intersection_samples = list(set(gen_smi_unique) & set(train_smi_unique))
    novelty_score = np.round(1 - len(intersection_samples) / len(gen_smi_unique), 3)
    print("Novelty score = {}".format(novelty_score))
    return novelty_score


def _parallel_get_fps(smi):
    smi = _canonicalize_smiles(smi)
    if isinstance(smi, str):
        mol = Chem.MolFromSmiles(smi)
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 6, 2048)


def _parallel_get_int_dists(fp_new, fps):
    dist = DataStructs.BulkTanimotoSimilarity(fp_new, fps)
    return np.sum(dist)


def _parallel_get_ext_dists(fp_new, fps):
    sm = DataStructs.BulkTanimotoSimilarity(fp_new, fps)
    return np.max(sm)


def compute_external_diversity(file_A, file_B):
    smiles_A = pd.read_csv(file_A).Smiles.map(standardize_smiles_error_handle)
    smiles_B = pd.read_csv(file_B).iloc[:1000000].Smiles
    smiles_B = smiles_B.map(standardize_smiles_error_handle)
    smiles_A = smiles_A[~smiles_A.isnull()]
    smiles_B = smiles_B[~smiles_B.isnull()]

    with Pool() as pool:
        fps_A = pool.map(_parallel_get_fps, smiles_A)

    with Pool() as pool:
        fps_B = pool.map(_parallel_get_fps, smiles_B)

    fps_A = list(filter(None, fps_A))
    fps_B = list(filter(None, fps_B))

    sims = []
    for fp in fps_A:
        simi = _parallel_get_ext_dists(fp, fps_B)
        sims.append(simi)

    return 1 - np.sum(sims) / len(fps_A)


def compute_internal_diversity(file_A):
    smiles_A = pd.read_csv(file_A).Smiles.map(standardize_smiles_error_handle)
    smiles_A = smiles_A[~smiles_A.isnull()]

    with Pool() as pool:
        fps_A = pool.map(_parallel_get_fps, smiles_A)

    fps_A = list(filter(None, fps_A))

    sims = []
    for fp in fps_A:
        simi = _parallel_get_int_dists(fp, fps_A)
        sims.append(simi)

    return 1 - np.sum(sims) / (len(fps_A) ** 2)


if __name__ == "__main__":
    freeze_support()
    create_folder('gen_samples/')
    model = load_json_model("generator_model_2021-12-17/generator_model.json")
    model.compile(optimizer='adam',
                  loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=[keras.metrics.CategoricalAccuracy()])
    model.load_weights("./checkpoints/generator_2021-12-17/")
    gen_samples_df = []
    count = 0
    for idx in range(10000):
        gen_sample = {}
        try:
            smi, num_atoms = generate_smiles(model, idx)
        except Exception as e:
            print(e)
            continue

        gen_sample["Smiles"] = smi
        gen_sample["NumAtoms"] = num_atoms
        gen_samples_df.append(gen_sample)
        count += 1
        print("validation rate = {}".format(np.round(count / (idx + 1), 3)))

    gen_samples_df = pd.DataFrame(gen_samples_df)
    gen_samples_df.to_csv('generated_molecules.csv', index=False)
    compute_unique_score()
    compute_novelty_score()
    ext_div = compute_external_diversity('generated_molecules.csv', 'molecules_chembl.csv')
    int_div = compute_internal_diversity('generated_molecules.csv')
    print('external diversity = {0} and internal diversity = {1}'.format(ext_div, int_div))
    breakpoint()
