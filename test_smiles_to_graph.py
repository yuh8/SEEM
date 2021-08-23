import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from copy import deepcopy
from src.data_process_utils import is_mol_valid, standardize_smiles, smiles_to_graph, graph_to_smiles
RDLogger.DisableLog('rdApp.*')
np.set_printoptions(threshold=np.inf)


def unit_test(smi_path="data/data_train.smi"):
    with Chem.SmilesMolSupplier(smi_path, nameColumn=-1, titleLine=False) as suppl:
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
                continue
            count_pass += 1
            print(count_pass / count_all)


if __name__ == "__main__":
    unit_test()
