import numpy as np
import pandas as pd
from rdkit.Chem.Descriptors import qed
from scipy.special import softmax
from train_generator import loss_func, get_metrics, get_optimizer, SeedGenerator
from src.data_process_utils import (get_action_mask_from_state,
                                    get_last_col_with_atom, draw_smiles,
                                    get_initial_act_vec, graph_to_smiles)
from src.misc_utils import create_folder, load_json_model
from src.CONSTS import (BOND_NAMES, MAX_NUM_ATOMS,
                        FEATURE_DEPTH,
                        ATOM_MAX_VALENCE,
                        ATOM_LIST, CHARGES)


def sample_action(action_logits, state, T=0.75):
    action_mask = get_action_mask_from_state(state)
    action_logits -= action_mask * 1e9
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


def generate_smiles(model, max_num_atoms, gen_idx):
    num_atoms = np.random.randint(9, max_num_atoms + 1)
    state = np.zeros((MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH + 1))
    is_terminate = False

    while not is_terminate:
        X_in = state[np.newaxis, ...]
        action_logits = model(X_in, training=False).numpy()[0]
        state, is_terminate = update_state_with_action(action_logits, state, num_atoms)

    smi_graph = state[..., :-1]
    smi, mol = graph_to_smiles(smi_graph)
    draw_smiles(smi, "gen_samples/gen_sample_{}".format(gen_idx))
    print('Smiles: {} with QED {}'.format(smi, qed(mol)))
    qed_score = qed(mol)
    return smi, qed_score


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
            smi, qed_score = generate_smiles(model, 36, idx)
        except:
            continue

        gen_sample["Smiles"] = smi
        gen_sample["QED"] = qed_score
        gen_samples_df.append(gen_sample)
        count += 1
        print("validation rate = {}".format(np.round(count / (idx + 1), 3)))

    gen_samples_df = pd.DataFrame(gen_samples_df)
    gen_samples_df.to_csv('generated_molecules.csv', index=False)
