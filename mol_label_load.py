#!/usr/bin/env python
# coding: utf-8

# In[13]:


import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdPartialCharges
from natsort import natsorted
import numpy as np
import pandas as pd


# In[14]:


filepath="/home/tsutsumi/fukui-gcn-test"

# In[26]:


def SDFDataLoader():
    filepath="/home/tsutsumi/fukui-gcn-test"
    mols_list=[]
    labels_list=[]
    sdf_path=os.path.join(filepath+"/data")
    os.chdir(sdf_path)
    sdf_mols = Chem.SDMolSupplier("DOWNLOAD-Y5rqtAZiCRg_r9ZlO5sI9iVZQvZl8zgDQp9qL23A2io=.sdf",removeHs=True)
    for sdf_mol in sdf_mols:
        if sdf_mol is not None:
            smiles=Chem.MolToSmiles(sdf_mol)
            label_value_list=[]
            for atom in sdf_mol.GetAtoms():
                res = atom.GetPropsAsDict().get("_GasteigerCharge", None)
                if res:
                    GasteigerCharge=float(res)
                else:
                    m = atom.GetOwningMol()
                    rdPartialCharges.ComputeGasteigerCharges(m)
                    GasteigerCharge=float(atom.GetProp("_GasteigerCharge"))
                label_value_list.append(GasteigerCharge)
            if any(np.isnan(label_value_list)):
                print("nan:",smiles)
            else:
                mols_list.append(sdf_mol)
                labels_list.append(label_value_list)
            #smiles=Chem.MolToSmiles(sdf_mol)
            #degree_list=[]
            #for atom in sdf_mol.GetAtoms():
                #degree=atom.GetDegree()
                #degree_list.append(degree)
            #if max(degree_list)==6:
                #print("degree:6:",smiles)                
            #else:
                #mols_list.append(sdf_mol)

            #if "P" in smiles:
                #degree_list=[]
                #for atom in sdf_mol.GetAtoms():
                    #degree=atom.GetDegree()
                    #degree_list.append(degree)
                #if max(degree_list)==6:
                    #print("P(degree:6):",smiles)
                #else:
                    #mols_list.append(sdf_mol)
                #print("P:",smiles)
            #else:
                #mols_list.append(sdf_mol)

    os.chdir(filepath)
    
    
    return mols_list,labels_list


def TestDataLoader():
    filepath="/home/tsutsumi/fukui-gcn"
    mols_list=[]
    labels_list=[]
    sdf_path=os.path.join(filepath+"/sdf_dataset")
    os.chdir(sdf_path)
    sdf_dir=os.listdir(sdf_path)
    sdf_dir=natsorted(sdf_dir,key=lambda y: y.lower())
    del sdf_dir[0:871]
    
    for i in range(len(sdf_dir)):
        sdf_filename = sdf_dir[i]
        sdf_mols = Chem.SDMolSupplier(sdf_filename,removeHs=True)
    
        for sdf_mol in sdf_mols:
            if sdf_mol is not None:
                smiles=Chem.MolToSmiles(sdf_mol)
                label_value_list=[]
                for atom in sdf_mol.GetAtoms():
                    res = atom.GetPropsAsDict().get("_GasteigerCharge", None)
                    if res:
                        GasteigerCharge=float(res)
                    else:
                        m = atom.GetOwningMol()
                        rdPartialCharges.ComputeGasteigerCharges(m)
                        GasteigerCharge=float(atom.GetProp("_GasteigerCharge"))
                    label_value_list.append(GasteigerCharge)
                if any(np.isnan(label_value_list)):
                    print("nan:",smiles)
                else:
                    mols_list.append(sdf_mol)
                    labels_list.append(label_value_list)

    os.chdir(filepath)
    
    
    return mols_list,labels_list

# In[28]:
def TestDataLoader_ChEMBL():
    filepath="/home/tsutsumi/fukui-gcn-test"
    mols_list=[]
    labels_list=[]
    sdf_path=os.path.join(filepath+"/data")
    os.chdir(sdf_path)
    sdf_mols = Chem.SDMolSupplier("DOWNLOAD-YUX2sWwOnO71QNCNikaw9AlYNPR4zuRqH4Y0dytjq3c=.sdf",removeHs=True)
    for sdf_mol in sdf_mols:
        if sdf_mol is not None:
            smiles=Chem.MolToSmiles(sdf_mol)
            label_value_list=[]
            for atom in sdf_mol.GetAtoms():
                res = atom.GetPropsAsDict().get("_GasteigerCharge", None)
                if res:
                    GasteigerCharge=float(res)
                else:
                    m = atom.GetOwningMol()
                    rdPartialCharges.ComputeGasteigerCharges(m)
                    GasteigerCharge=float(atom.GetProp("_GasteigerCharge"))
                label_value_list.append(GasteigerCharge)
            if any(np.isnan(label_value_list)):
                print("nan:",smiles)
            else:
                mols_list.append(sdf_mol)
                labels_list.append(label_value_list)

    os.chdir(filepath)
    
    
    return mols_list,labels_list

def PDBDataLoader():
    filepath="/home/tsutsumi/fukui-gcn_pooling/"
    mols_list=[]
    labels_list=[]
    # convert pdb to rdkit mol object
    os.chdir(os.path.join(filepath,"pdb_dataset/"))
    pdb_dir=os.listdir(".")
    pdb_dir=natsorted(pdb_dir)

    for i in range(len(pdb_dir)):
        pdb_filename=pdb_dir[i]
        pdb_mol=Chem.MolFromPDBBlock(
                pdb_filename, removeHs=True, proximityBonding=False)
        #mols = [mol for mol in pdb_mol if mol is not None]
        mols_list.append(pdb_mol)
    #load_label
    os.chdir(os.path.join(filepath,"fukui_data/"))
    label_list=os.listdir(".")
    label_list=natsorted(label_list)

    for i in range(len(label_list)):
        label_file_list=os.listdir(".")
        label_file_list=natsorted(label_file_list)
        label_filename = label_file_list[i]
        label_value= np.loadtxt(label_filename,delimiter=',')#array
        #print(label_value)
        fukui_value=label_value[:,3]
        #print(fukui_value)
        fukui_value=fukui_value.tolist()
        labels_list.append(fukui_value) 
    
    
    return mols_list,labels_list


# In[ ]:




