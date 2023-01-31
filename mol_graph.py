#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
from featurize import get_atom_feats,bond_features


degrees = [0, 1, 2, 3, 4, 5, 6]


# In[21]:


class MolGraph(object):
    def __init__(self):
        self.nodes = {} # dict of lists of nodes, keyed by node type

    def new_node(self, ntype, features=None, labels=None):
        new_node = Node(ntype, features,labels)
        self.nodes.setdefault(ntype, []).append(new_node)
        return new_node

    def add_subgraph(self, subgraph):
        old_nodes = self.nodes
        new_nodes = subgraph.nodes
        for ntype in set(old_nodes.keys()) | set(new_nodes.keys()):
            old_nodes.setdefault(ntype, []).extend(new_nodes.get(ntype, []))

    def sort_nodes_by_degree(self, ntype):
        nodes_by_degree = {i : [] for i in degrees}
        for node in self.nodes[ntype]:
            nodes_by_degree[len(node.get_neighbors(ntype))].append(node)

        new_nodes = []
        for degree in degrees:
            cur_nodes = nodes_by_degree[degree]
            self.nodes[(ntype, degree)] = cur_nodes
            new_nodes.extend(cur_nodes)

        self.nodes[ntype] = new_nodes

    def feature_array(self, ntype):
        assert ntype in self.nodes
        return np.array([node.features for node in self.nodes[ntype]])
    
    def labels_array(self):
        labels_array=np.array([node.labels for node in self.nodes['atom']])
        return np.expand_dims(labels_array,1)


    def neighbor_list(self, self_ntype, neighbor_ntype):
        assert self_ntype in self.nodes and neighbor_ntype in self.nodes
        neighbor_idxs = {n : i for i, n in enumerate(self.nodes[neighbor_ntype])}
        return [[neighbor_idxs[neighbor]
                for neighbor in self_node.get_neighbors(neighbor_ntype)]
                for self_node in self.nodes[self_ntype]]


# In[22]:


class Node(object):
    __slots__ = ['ntype','labels', 'features', '_neighbors']

    def __init__(self, ntype, features, labels=None):
        self.ntype = ntype
        self.features = features
        self._neighbors = []
        self.labels = labels

    def add_neighbors(self, neighbor_list):
        for neighbor in neighbor_list:
            self._neighbors.append(neighbor)
            neighbor._neighbors.append(self)

    def get_neighbors(self, ntype):
        return [n for n in self._neighbors if n.ntype == ntype]


# In[23]:


def graph_from_mol_tuple(mol_tuple, atom_labels_list=None):
    #graph_list = [graph_from_mol(m) for m in mol_tuple]
    graph_list = []
    for i, mols in enumerate(mol_tuple):
        mol_labels = None
        if atom_labels_list:
            mol_labels = atom_labels_list[i]
        if mols is not None:
            graph_list.append(graph_from_mol(mols, mol_labels))

    big_graph = MolGraph()
    for subgraph in graph_list:
        big_graph.add_subgraph(subgraph)

    # This sorting allows an efficient (but brittle!) indexing later on.
    big_graph.sort_nodes_by_degree('atom')
    return big_graph


# In[24]:


def graph_from_mol(mol, mol_labels=None):
    graph = MolGraph()
    atoms_by_rd_idx = {}

    for atom in mol.GetAtoms():
        features = get_atom_feats(atom)
        if mol_labels is not None:
            new_atom_node = graph.new_node('atom', labels=mol_labels[atom.GetIdx()],features=features)
        else:
            new_atom_node = graph.new_node('atom', features=features)

        atoms_by_rd_idx[atom.GetIdx()] = new_atom_node
    

        #print(atoms_by_rd_idx)
    for bond in mol.GetBonds():
        atom1_node = atoms_by_rd_idx[bond.GetBeginAtom().GetIdx()]
        atom2_node = atoms_by_rd_idx[bond.GetEndAtom().GetIdx()]
        new_bond_node = graph.new_node('bond', features=bond_features(bond))
        new_bond_node.add_neighbors((atom1_node, atom2_node))
        atom1_node.add_neighbors((atom2_node,))

    mol_node = graph.new_node('molecule')
    mol_node.add_neighbors(graph.nodes['atom'])
    return graph





# In[ ]:




