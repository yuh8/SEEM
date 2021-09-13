import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Descriptors import qed
from scipy.special import softmax
from train_generator import loss_func, get_metrics, get_optimizer, SeedGenerator
from src.data_process_utils import (get_action_mask_from_state,
                                    get_last_col_with_atom, draw_smiles,
                                    get_initial_act_vec, graph_to_smiles)
from src.misc_utils import create_folder, load_json_model
from src.CONSTS import (BOND_NAMES, MAX_NUM_ATOMS,
                        MIN_NUM_ATOMS, FEATURE_DEPTH,
                        ATOM_MAX_VALENCE,
                        ATOM_LIST, CHARGES)


def sample_action(action_logits, state, T=1):
    action_mask = get_action_mask_from_state(state)
    action_probs = softmax(action_logits / T)
    action_probs = action_probs * (1 - action_mask)
    action_probs = action_probs / np.sum(action_probs)
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


def generate_smiles(model, gen_idx):
    num_atoms = np.random.randint(MIN_NUM_ATOMS, MAX_NUM_ATOMS)
    state = np.zeros((MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH + 1))
    is_terminate = False

    while not is_terminate:
        X_in = state[np.newaxis, ...]
        action_logits = model(X_in, training=False).numpy()[0]
        state, is_terminate = update_state_with_action(action_logits, state, num_atoms)

    smi_graph = state[..., :-1]
    smi = graph_to_smiles(smi_graph,
                          draw_mol=True,
                          file_name="gen_samples/gen_sample_{}".format(gen_idx))
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


if __name__ == "__main__":
    create_folder('gen_samples/')
    model = load_json_model("generator_model/generator_model.json", SeedGenerator, "SeedGenerator")
    model.compile(optimizer=get_optimizer(),
                  loss_fn=loss_func,
                  metric_fn=get_metrics)
    model.load_weights("./checkpoints/generator/")
    gen_samples_df = []
    count = 0
    for idx in range(10000):
        gen_sample = {}
        try:
            smi, num_atoms = generate_smiles(model, idx)
        except:
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
    breakpoint()
