import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from copy import deepcopy
from src.data_process_utils import (is_mol_valid, is_smile_valid, draw_smiles,
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
            smi_original = Chem.MolToSmiles(mol)
            try:
                smi_original = standardize_smiles(smi_original)
            except:
                continue
            smi_graph = smiles_to_graph(smi_original)
            smi_reconstructed = graph_to_smiles(smi_graph)
            count_all += 1
            if (smi_original != smi_reconstructed) and ("[nH]" not in smi_original):
                print("original: {0}, reconstructed: {1}, pass_rate: {2}".format(smi_original, smi_reconstructed, round(count_pass / count_all, 4)))
                # breakpoint()
                continue
            count_pass += 1
    print('final pass rate = {}'.format(round(count_pass / count_all, 4)))


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

        smi_graph = smiles_to_graph(smi_original)
        smi_reconstructed = graph_to_smiles(smi_graph)
        count_all += 1
        if (smi_original != smi_reconstructed) and ("[nH]" not in smi_original):
            print("original: {0}, reconstructed: {1}, pass_rate: {2}".format(smi_original, smi_reconstructed, round(count_pass / count_all, 4)))
            continue
        count_pass += 1
    print('final pass rate = {}'.format(round(count_pass / count_all, 4)))


def test_single(smi_original):
    smi_original = standardize_smiles(smi_original)
    smi_graph = smiles_to_graph(smi_original)
    smi_reconstructed = graph_to_smiles(smi_graph)
    return smi_original == smi_reconstructed


def test_decompose_smi_graph(smi_original):
    smi_graph = smiles_to_graph(smi_original)
    decompose_smi_graph(smi_graph)


def unit_test_chembl(draw=False):
    df_data = pd.read_csv("D:/seed_data/small_mol_db.csv", sep=';', low_memory=False)
    count_all = 0
    count_pass = 0
    for _, row in df_data.iterrows():
        smi_original = deepcopy(row.Smiles)
        if not is_smile_valid(smi_original):
            continue

        try:
            smi_original = standardize_smiles(smi_original)
        except:
            continue

        smi_graph = smiles_to_graph(smi_original)
        smi_reconstructed = graph_to_smiles(smi_graph)
        count_all += 1
        if (smi_original != smi_reconstructed) and ("[nH]" not in smi_original):
            if draw:
                draw_smiles(row.Smiles, 'mol_original')
                draw_smiles(smi_reconstructed, 'mol_reconstructed')
            print("original: {0}, reconstructed: {1}, pass_rate: {2}".format(smi_original, smi_reconstructed, round(count_pass / count_all, 4)))
            continue
        count_pass += 1
    print('final pass rate = {}'.format(round(count_pass / count_all, 4)))


if __name__ == "__main__":
    # print(test_single("COC(=O)CCCCC(CCSS/C(CCO)=C(\C)N(C=O)Cc1cnc(C)nc1N)SC(C)=O"))
    # test_decompose_smi_graph("NCC(O)CON")
    # unit_test_mol()
    # unit_test_smiles("data/df_train.csv")
    unit_test_chembl()
