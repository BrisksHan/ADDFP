import pickle
import numpy as np
import scipy.io
import pandas as pd
import random
import utils
import csv
from rdkit import Chem
from rdkit.Chem import AllChem
import lzma



def read_drug_smiles(file_path = 'data/data_cvs1/drug_SMILES_750.csv'):
    '''
    read the drug smiles from the csv file
    '''
    with open(file_path, 'r') as file:
        data = csv.reader(file)
        drug_smiles_dict = {}
        drug_ids = []
        drug_smiles = []
        for row in data:
            if len(row) >= 2:  # Ensure row has drug_id and SMILES
                drug_id = row[0]
                smiles = row[1]
                drug_ids.append(drug_id)
                drug_smiles.append(smiles)
                drug_smiles_dict[drug_id] = smiles
        '''
        print(drug_smiles_dict)
        count = 0
        for drug_id, smiles in drug_smiles_dict.items():
            count += 1
            print(f"Count:{count}, Drug ID: {drug_id}, SMILES: {smiles}")
        print(f"Total drugs: {count}")
        '''
        return drug_smiles_dict, drug_ids, drug_smiles
    
def read_frequency(file_path = 'data/drug_side_frequencies.pkl'):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def read_cvs1_mask_mat(file_path='data/data_cvs1/mask_mat_750.mat'):
    mat = scipy.io.loadmat(file_path)
    return mat

def read_cvs2_mask_mat(file_path='data/data_cvs2/blind_mask_mat_750.mat'):
    mat = scipy.io.loadmat(file_path)
    return mat

def read_cvs1_10_fold_split(split = 0):
    if split < 0 or split >= 10:
        raise Exception('index should be in [0, 10)')
    '''
    contains the ten fold split for the dataset under cvs1
    each fold contains a training and test set
    each training and test set contains dictionary, the key is drug_id and the value is a list of side effectt values
    '''
    frequency_table = read_frequency()
    mask_table = read_cvs1_mask_mat()

    mask_table_key = f'mask{split}'
    cur_fold_mask_table = mask_table[mask_table_key]
    cur_fold_frequency_table = frequency_table.copy()
    cur_fold_test_data = []
    cur_fold_train_data = list(range(750))
    for i in range(len(frequency_table)):
        for j in range(len(frequency_table[i])):
            if int(cur_fold_mask_table[i][j]) == 0 and frequency_table[i][j] > 0:
                cur_fold_frequency_table[i][j] = 0
            if int(cur_fold_mask_table[i][j]) == 0 and frequency_table[i][j] == 0:
                raise Exception('please check the mask table')
            
    return cur_fold_frequency_table, cur_fold_test_data, cur_fold_train_data, cur_fold_mask_table

def read_cvs2_10_fold_split(split = 0):
    if split < 0 or split >= 10:
        raise Exception('index should be in [0, 10)')
    '''
    contains the ten fold split for the dataset under csv2
    each fold contains a training and test set
    each training and test set contains dictionary, the key is drug_id and the value is a list of side effectt values
    '''
    frequency_table = read_frequency()
    blind_mask_table = read_cvs2_mask_mat()
    
    mask_table_key = f'mask{split}'
    cur_fold_mask_table = blind_mask_table[mask_table_key]
    cur_fold_frequency_table = frequency_table.copy()
    cur_fold_test_data = []
    cur_fold_train_data = []

    for i in range(len(frequency_table)):
        if cur_fold_mask_table[i][0] == 1:
            cur_fold_train_data.append(i)
        else:
            for j in range(len(frequency_table[i])):
                if cur_fold_mask_table[i][j] == 0:
                    cur_fold_frequency_table[i][j] = 0
    return cur_fold_frequency_table, cur_fold_test_data, cur_fold_train_data, cur_fold_mask_table

def read_drug_SMILES_xlsx(file_path='data/drug_SMILES.xlsx'):
    data = pd.read_excel(file_path, header=None)
    return data.values.tolist()

def read_drug_description_xlsx(file_path='data/drug_description.xlsx'):
    data = pd.read_excel(file_path, header=None)
    return data.values.tolist()

def read_side_description_xlsx(file_path='data/side_description.xlsx'):
    data = pd.read_excel(file_path, header=None)
    return data.values.tolist()

def read_drug_text_smilarity(file_path='data/Text_similarity_five.pkl'):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def read_drug_target_feature():
    with lzma.open('data/drug_target.xz', 'rb') as f:
        target_features = pickle.load(f)
    return target_features

def get_drug_morgan_fingerprints():
    drug_mfs = []
    smiles = pd.read_excel('data/drug_SMILES.xlsx', header=None, engine='openpyxl')[1].tolist()
    for i in range(len(smiles)):
        mol = Chem.MolFromSmiles(smiles[i])
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        drug_mfs.append(np.fromstring(fp.ToBitString(), 'i1') - 48)
        #self.v[i] = np.fromstring(fp.ToBitString(), 'i1') - 48
    print(len(drug_mfs))
    return drug_mfs

def get_all_side_descriptions():
    side_description = read_side_description_xlsx()
    all_side_descriptions = []
    for item in side_description:
        all_side_descriptions.append(item[1])
    return all_side_descriptions

def get_all_drug_descriptions():
    drug_description = read_drug_description_xlsx()
    all_drug_descriptions = []
    for item in drug_description:
        all_drug_descriptions.append(item[1])
    return all_drug_descriptions

def get_all_drug_smiles():
    drug_smiles = read_drug_SMILES_xlsx()
    all_drug_smiles = []
    for item in drug_smiles:
        all_drug_smiles.append(item[1])
    return all_drug_smiles
     
if __name__ == "__main__":
    # Example usage
    #frequencies = read_frequency()
    #print(f"Loaded drug side effect frequencies. Data shape: {len(frequencies)}")
    blind_mask_table = read_cvs1_mask_mat()['mask0']
    for item in blind_mask_table:
        print(item)