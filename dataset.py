#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from torch.utils import data
from mol_graph import graph_from_mol_tuple


# In[2]:


#dataset
class MoleculeDataset(data.Dataset):

    def __init__(self, mol_list, label_list):
        self.mol_list = mol_list
        self.label_list = label_list

    def __len__(self):
        return len(self.mol_list)

    #要素を参照
    def __getitem__(self, index):
        return self.mol_list[index], self.label_list[index]


# In[3]:

#minibatch生成のための定義
def gcn_collate_fn(batch):

    mols = []
    atom_labels = []

    for mol, label in batch:
        mols.append(mol)
        atom_labels.append(label)

    molgraph = graph_from_mol_tuple(mols, atom_labels)
    arrayrep = {'atom_features' : molgraph.feature_array('atom'),
                'bond_features' : molgraph.feature_array('bond'),
                'atom_labels': molgraph.labels_array()
                }  

    degrees = [0,1,2,3,4,5,6]
    for degree in degrees:
        arrayrep[('atom_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'atom'), dtype=int)
        arrayrep[('bond_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'bond'), dtype=int)

    #print(arrayrep)
    #print(len(arrayrep["atom_list"]))
    #print(len(arrayrep[('atom_neighbors', 1)]))
    #print(len(arrayrep[('bond_neighbors', 1)]))
    #print(arrayrep['rdkit_ix'])
    return arrayrep
