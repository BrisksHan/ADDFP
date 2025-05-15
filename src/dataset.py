

from torch_geometric.data import Data, Dataset, InMemoryDataset
from rdkit import Chem
import networkx as nx
import numpy as np
import torch
 

#-------------------------------------------------dataset------------------------------------------------------------




def atom_features(atom):
    HYB_list = [Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2,
                Chem.rdchem.HybridizationType.UNSPECIFIED, Chem.rdchem.HybridizationType.OTHER]
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Sm', 'Tc', 'Gd', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetExplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding(atom.GetFormalCharge(), [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding(atom.GetHybridization(), HYB_list) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))



def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature)

    features = np.array(features)

    edges = []
    edge_type = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_type.append(bond.GetBondTypeAsDouble())
    g = nx.Graph(edges).to_directed()
    edge_index = []

    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    if not edge_index:
        edge_index = [[0, 0]] #add self loop
        edge_index = np.array(edge_index).transpose(1, 0)
    else:
        edge_index = np.array(edge_index).transpose(1, 0)

    #if not edge_index:
    #    #print('an edge is empty:', )
    #    raise Exception('an edge is empty:', list(mol.GetBonds()))

    return c_size, features, edge_index, edge_type



class TrainDataset(InMemoryDataset):
    def __init__(self, data):
        super(TrainDataset, self).__init__()
        self.data = data
        self.drug_smiles = [item[5] for item in data]
        
        # Process molecular graphs
        self.graphs = []
        for smiles in self.drug_smiles:
            #print(smiles)
            c_size, features, edge_index, edge_type = smile_to_graph(smiles)
            g = Data(
                x=torch.FloatTensor(features),
                edge_index=torch.LongTensor(edge_index),
                edge_attr=torch.FloatTensor(edge_type) if len(edge_type) > 0 else torch.FloatTensor([]),
                num_nodes=c_size
            )
            self.graphs.append(g)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get molecular graph
        #mol_graph = self.graphs[idx]
        
        frequency, drug_text_similarity, smiles_encoding, drug_description_embedding, drug_mfp, smiles_str, drug_target_feature, llm_molecular_embedding = self.data[idx]
        
        return (
                self.graphs[idx],
                drug_text_similarity, 
                smiles_encoding,
                drug_description_embedding,
                drug_mfp,
                drug_target_feature,
                llm_molecular_embedding,
                frequency
            )



#----------------------------------------Test Custom Dataset-------------------------------------------------

class TestDataset(InMemoryDataset):
    def __init__(self, data):
        super(TestDataset, self).__init__()
        self.data = data
        self.drug_smiles = [item[0] for item in data]
        
        # Process molecular graphs
        self.graphs = []
        for smiles in self.drug_smiles:
            #print('the smile:',smiles)
            c_size, features, edge_index, edge_type = smile_to_graph(smiles)
            g = Data(
                x=torch.FloatTensor(features),
                edge_index=torch.LongTensor(edge_index),
                edge_attr=torch.FloatTensor(edge_type) if len(edge_type) > 0 else torch.FloatTensor([]),
                num_nodes=c_size
            )
            self.graphs.append(g)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get molecular graph
        #mol_graph = self.graphs[idx]
        
        smiles, smiles_feat, kg_embed1, seq_feat, kg_embed2 = self.data[idx]
        
        return (
                self.graphs[idx],
                torch.LongTensor(smiles_feat),
                torch.FloatTensor(kg_embed1), 
                torch.LongTensor(seq_feat),
                torch.FloatTensor(kg_embed2),
            )