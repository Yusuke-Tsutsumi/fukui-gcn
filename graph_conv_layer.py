#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import numpy as np


# In[3]:


#「torch.nn.Moduleクラスのサブクラス化」によるモデルの定義
class GraphConvLayer(nn.Module):

    def __init__(self,
                in_features,
                out_features,
                activation=nn.ReLU(),
                normalize=True
                ):

        super().__init__()

        self.degrees = [0,1, 2, 3, 4, 5, 6]
        self.activation = activation
        self.normalize = normalize

        self.in_features = in_features
        self.out_channel = out_features

        self.self_activation = nn.Linear(in_features, out_features, bias=False)#linerは全結合層
        # パラメーター（ニューロンへの入力で必要となるもの）の定義
        self.bias = nn.Parameter(torch.Tensor(np.zeros(out_features)))#バイアス

        self.degree_liner_layer_list = []
        for k in range(len(self.degrees)):
            self.degree_liner_layer_list.append(nn.Linear(in_features + 6, out_features, bias=False))
            #self.degree_liner_layer_list.append(nn.Linear(in_features, out_features, bias=False))

        self.degree_liner_layer_list = torch.nn.ModuleList(self.degree_liner_layer_list)

        if self.normalize:
            self.batch_norm = torch.nn.BatchNorm1d(out_features)

    #フォワードパスを定義、フォワードパス（＝活性化関数で変換しながらデータを流す処理）
    def forward(self, array_rep, atom_features, bond_features):
        self_activations = self.self_activation(atom_features)#活性化関数は変数として定義
        # 「出力＝活性化関数（第n層（入力））」の形式で記述する。
        activations_by_degree = []

        for i, degree in enumerate(self.degrees):
            # Convert to long for use as index
            atom_neighbors_list = torch.tensor(array_rep[('atom_neighbors', degree)], dtype=torch.long)
            bond_neighbors_list = torch.tensor(array_rep[('bond_neighbors', degree)], dtype=torch.long)

            if len(atom_neighbors_list) > 0:
                stacked_neighbors_tensor = torch.cat([atom_features[atom_neighbors_list], bond_features[bond_neighbors_list]], axis=2)
                #stacked_neighbors_tensor = torch.cat([atom_features[atom_neighbors_list]], axis=2)
                summed_neighbors_tensor = torch.sum(stacked_neighbors_tensor, axis=1)
                activations = self.degree_liner_layer_list[i](summed_neighbors_tensor)
                activations_by_degree.append(activations)

        neighbour_activations = torch.cat(activations_by_degree, axis=0)
        total_activations = neighbour_activations + self_activations + self.bias

        if self.activation:
            total_activations = self.activation(total_activations)

        if self.normalize:
            total_activations = self.batch_norm(total_activations)

        return total_activations






