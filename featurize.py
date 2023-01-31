#!/usr/bin/env python
# coding: utf-8



from atom_features import ATOM_FEATURES 
from rdkit import Chem
import numpy as np
import os



filepath="/home/tsutsumi/fukui-gcn-test"



def get_atom_feats(atom):
    filepath="/home/tsutsumi/fukui-gcn-test"
    features_path=os.path.join(filepath+"/data/feature_list.dat")
    features=[]
    feats_data=open(features_path, 'r')
    # 一行ずつ読み込んでは表示する
    for rows in feats_data:
        # コメントアウト部分を省く処理
        if rows[0]=='#':
            continue
        # 値を変数に格納する
        row = rows.rstrip('\n').split(' ')
        feats = row[0]
        features.append(feats)
    feats_data.close()
    atom_feature = np.empty(shape=(len(features),))
    atom_feature[:]=0.0
    for feature_id, f in enumerate(features):
        atom_feature[feature_id] = ATOM_FEATURES[f](atom)
    return atom_feature



def bond_features(bond):
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.DOUBLE,
                     bond.IsInRing(),
                     bt == Chem.rdchem.BondType.SINGLE,
                     bt == Chem.rdchem.BondType.AROMATIC,
                     bond.GetIsConjugated(),
                     bt == Chem.rdchem.BondType.TRIPLE,
                     ])






