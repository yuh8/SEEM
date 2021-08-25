import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from copy import deepcopy
from src.data_process_utils import (is_mol_valid, is_smile_valid,
                                    standardize_smiles, decompose_smi_graph,
                                    smiles_to_graph, graph_to_smiles)
RDLogger.DisableLog('rdApp.*')
np.set_printoptions(threshold=np.inf)


def unit_test_mol(mol_path="data/data_train.smi"):
    with Chem.SmilesMolSupplier(mol_path, nameColumn=-1, titleLine=False) as suppl:
        count_pass = 0
        count_all = 0
        for mol in suppl:
            if not is_mol_valid(mol):
                continue
            # charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
            # if any(np.abs(charges) > 2):
            #     continue
            mol_original = deepcopy(mol)
            smi_original = Chem.MolToSmiles(mol)
            try:
                smi_original = standardize_smiles(smi_original)
            except:
                continue
            smi_graph, charges = smiles_to_graph(smi_original)
            smi_reconstructed, mol = graph_to_smiles(smi_graph, charges)
            count_all += 1
            if (smi_original != smi_reconstructed) and ("[nH]" not in smi_original):
                # print("original ={0}, reconstructed ={1}".format(smi_original, smi_reconstructed))
                # breakpoint()
                continue
            count_pass += 1
            print(count_pass / count_all)


def unit_test_smiles(smi_path):
    count_pass = 0
    count_all = 0
    df_train = pd.read_csv(smi_path)
    for smi_original in df_train.SMILES.values:
        if not is_smile_valid(smi_original):
            continue

        try:
            smi_original = standardize_smiles(smi_original)
        except:
            continue

        smi_graph, charges = smiles_to_graph(smi_original)
        smi_reconstructed, _ = graph_to_smiles(smi_graph, charges)
        count_all += 1
        if (smi_original != smi_reconstructed) and ("[nH]" not in smi_original):
            # print("original ={0}, reconstructed ={1}".format(smi_original, smi_reconstructed))
            # breakpoint()
            continue
        count_pass += 1
        print(count_pass / count_all)


def test_single(smi_original):
    smi_original = standardize_smiles(smi_original)
    smi_graph, charges = smiles_to_graph(smi_original)
    smi_reconstructed, _ = graph_to_smiles(smi_graph, charges)
    breakpoint()
    return smi_original == smi_reconstructed


def test_decompose_smi_graph(smi_original):
    smi_graph, charges = smiles_to_graph(smi_original)
    decompose_smi_graph(smi_graph)


if __name__ == "__main__":
    test_single("COC(=O)CCCCC(CCSS/C(CCO)=C(\C)N(C=O)Cc1cnc(C)nc1N)SC(C)=O")
    breakpoint()
    test_decompose_smi_graph("CN1CCCC1")
    unit_test_mol()
    unit_test_smiles("data/df_train.csv")
