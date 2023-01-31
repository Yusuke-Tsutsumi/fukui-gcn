#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from graph_conv_layer import GraphConvLayer
import numpy as np
from torchmetrics import R2Score, MeanAbsoluteError
import os
import random


# In[ ]:


#class GraphConvModel(nn.Module):
class GraphConvModel(pl.LightningModule):

    def __init__(self,
                device_ext=torch.device('cpu'),
                atom_features_size=63,
                conv_layer_sizes=[20, 20, 20],
                mlp_layer_sizes=[100, 1],
                activation=nn.ReLU(),
                normalize=True,
                lr=0.001,
                ):

        super().__init__()

        self.device_ext = device_ext
        self.activation = activation
        self.normalize = normalize
        self.number_atom_features = atom_features_size
        self.graph_conv_layer_list = nn.ModuleList()
        self.mlp_layer_list = nn.ModuleList()
        self.batch_norm_list = nn.ModuleList()
        self.lr = lr

        prev_layer_size = atom_features_size

        for i, layer_size in enumerate(conv_layer_sizes):
            self.graph_conv_layer_list.append(GraphConvLayer(prev_layer_size, layer_size))
            prev_layer_size = layer_size


        prev_layer_size = conv_layer_sizes[-1]
        for i, layer_size in enumerate(mlp_layer_sizes):
            self.mlp_layer_list.append(torch.nn.Linear(prev_layer_size, layer_size, bias=True))
            prev_layer_size = layer_size

        if normalize:
            for i, mlp_layer in enumerate(self.mlp_layer_list):
                if i < len(self.mlp_layer_list) -1 :
                    self.batch_norm_list.append(torch.nn.BatchNorm1d(mlp_layer.out_features))

    def forward(self, array_rep, atom_features, bond_features):
        all_layer_fps = []
        for graph_conv_layer in self.graph_conv_layer_list:
            atom_features = graph_conv_layer(array_rep, atom_features, bond_features)
            all_layer_fps.append(torch.unsqueeze(atom_features, dim=0))
        
        layer_output = torch.cat(all_layer_fps, axis=0)
        layer_output = torch.sum(layer_output, axis=0)
        
        # MLP Layer
        x = layer_output.float()
        for i, mlp_layer in enumerate(self.mlp_layer_list):
            x = mlp_layer(x)
            if i < len(self.mlp_layer_list) - 1:
                if self.activation:
                    x = self.activation(x)
                if self.normalize:
                    x = self.batch_norm_list[i](x)

        return x
    
    
    def training_step(self, batch, batch_idx):

        array_rep = batch
        atom_features = array_rep['atom_features']
        bond_features = array_rep['bond_features']
        atom_labels = array_rep['atom_labels']
        
        """
        atom_features=atom_features.tolist()
        bond_features=bond_features.tolist()
        atom_labels= atom_labels.tolist()
        
        random.seed(0)
        random.shuffle(atom_features)
        random.seed(1)
        random.shuffle(bond_features)
        random.seed(3)
        random.shuffle(atom_labels)

        atom_features=np.array(atom_features)
        bond_features=np.array(bond_features)
        atom_labels=np.array(atom_labels)
        """
        
        atom_features = torch.tensor(atom_features.astype(np.float64), dtype=torch.float)
        bond_features = torch.tensor(bond_features.astype(np.float64), dtype=torch.float)
        atom_labels = torch.tensor(atom_labels.astype(np.float64), dtype=torch.float)


        atom_features = atom_features.to(self.device_ext)
        bond_features = bond_features.to(self.device_ext)
        atom_labels = atom_labels.to(self.device_ext)

        y_pred = self(array_rep, atom_features, bond_features)
        #y_pred = y_pred.squeeze(1)
        #atom_labels = atom_labels.unsqueeze(1)
        
        #print("y_pred:",len(y_pred))
        #print("atom_labels:",len(atom_labels))

        loss = F.mse_loss(y_pred, atom_labels)
        r2score = R2Score()
        r2score = R2Score().to(self.device)
        r2 = r2score(y_pred, atom_labels)
        mean_absolute_error=MeanAbsoluteError()
        mean_absolute_error=MeanAbsoluteError().to(self.device)
        MAE_loss=mean_absolute_error(y_pred, atom_labels)
        # https://github.com/pytorch/ignite/issues/453
        #var_y = torch.var(atom_labels, unbiased=False)
        #r2 = 1.0 - F.mse_loss(y_pred, atom_labels, reduction="mean") / var_y
        

        ret = {'loss': loss, 'train_r2': r2,'y_pred':y_pred,'atom_labels':atom_labels,'MAE_loss':MAE_loss}

        return ret
    
    def training_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', loss)

        MAE_loss = torch.stack([x['MAE_loss'] for x in outputs]).mean()
        self.log('train_mae_loss', MAE_loss)

        r2 = torch.stack([x['train_r2'] for x in outputs]).mean()
        self.log('train_r2', r2)
        
        

        train_y_pred=[]
        for x in outputs:
            y_pred=x['y_pred'].squeeze(1).detach().cpu().numpy()
            train_y_pred.append(y_pred)
        #print("train_y_pred:",len(train_y_pred))

        train_atom_labels=[]
        for x in outputs:
            atom_labels=x['atom_labels'].squeeze(1).detach().cpu().numpy()
            train_atom_labels.append(atom_labels)
        #print("train_labels:",len(train_atom_labels))
                
        os.chdir("/home/tsutsumi/fukui-gcn-test/")
        with open("train_pred.csv",'w') as f:
            for i in range(len(train_y_pred)):
                train_pred=train_y_pred[i]
                for j in range(len(train_pred)):
                    print(train_pred[j],file=f)
        with open("train_labels.csv",'w') as f:
            for i in range(len(train_atom_labels)):
                train_labels=train_atom_labels[i]
                for j in range(len(train_labels)):
                    print(train_labels[j],file=f)


    def validation_step(self, batch, batch_idx):

        array_rep = batch
        atom_features = array_rep['atom_features']
        bond_features = array_rep['bond_features']
        atom_labels = array_rep['atom_labels']
        
        """
        atom_features=atom_features.tolist()
        bond_features=bond_features.tolist()
        atom_labels= atom_labels.tolist()
        
        random.seed(4)
        random.shuffle(atom_features)
        random.seed(5)
        random.shuffle(bond_features)
        random.seed(6)
        random.shuffle(atom_labels)

        atom_features=np.array(atom_features)
        bond_features=np.array(bond_features)
        atom_labels=np.array(atom_labels)
        """

        atom_features = torch.tensor(atom_features.astype(np.float64), dtype=torch.float)
        bond_features = torch.tensor(bond_features.astype(np.float64), dtype=torch.float)
        atom_labels = torch.tensor(atom_labels.astype(np.float64), dtype=torch.float)
            
        atom_features = atom_features.to(self.device)
        bond_features = bond_features.to(self.device)
        atom_labels = atom_labels.to(self.device)

        y_pred = self(array_rep, atom_features, bond_features)
        #y_pred = y_pred.squeeze(1)
        #atom_labels = atom_labels.unsqueeze(1)

        loss = F.mse_loss(y_pred, atom_labels)
        r2score = R2Score()
        r2score = R2Score().to(self.device)
        r2 = r2score(y_pred, atom_labels)
        mean_absolute_error=MeanAbsoluteError()
        mean_absolute_error=MeanAbsoluteError().to(self.device)
        MAE_loss=mean_absolute_error(y_pred, atom_labels)
        # https://github.com/pytorch/ignite/issues/453
        #var_y = torch.var(atom_labels, unbiased=False)
        #r2 = 1.0 - F.mse_loss(y_pred, atom_labels, reduction="mean") / var_y
        

        ret = {'loss': loss, 'val_r2': r2,'y_pred':y_pred,'atom_labels':atom_labels,'MAE_loss':MAE_loss}
        
        return ret

    def validation_epoch_end(self, outputs):

        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('val_loss', loss)

        MAE_loss = torch.stack([x['MAE_loss'] for x in outputs]).mean()
        self.log('val_mae_loss', MAE_loss)

        r2 = torch.stack([x['val_r2'] for x in outputs]).mean()
        self.log('val_r2', r2)
        

        val_y_pred=[]
        for x in outputs:
            y_pred=x['y_pred'].squeeze(1).detach().cpu().numpy()
            val_y_pred.append(y_pred)
        #print("val_y_pred",len(val_y_pred))
        
        val_atom_labels=[]
        for x in outputs:
            atom_labels=x['atom_labels'].squeeze(1).detach().cpu().numpy()
            val_atom_labels.append(atom_labels)
        #print("val_labels",len(val_atom_labels))

        os.chdir("/home/tsutsumi/fukui-gcn-test/")
        with open("val_pred.csv",'w') as f:
            for i in range(len(val_y_pred)):
                val_pred=val_y_pred[i]
                for j in range(len(val_pred)):
                    print(val_pred[j],file=f)

        with open("val_labels.csv",'w') as f:
            for i in range(len(val_atom_labels)):
                val_labels=val_atom_labels[i]
                for j in range(len(val_labels)):
                    print(val_labels[j],file=f)

    def test_step(self, batch, batch_idx):

        array_rep = batch
        atom_features = array_rep['atom_features']
        bond_features = array_rep['bond_features']
        atom_labels = array_rep['atom_labels']

        atom_features = torch.tensor(atom_features.astype(np.float64), dtype=torch.float)
        bond_features = torch.tensor(bond_features.astype(np.float64), dtype=torch.float)
        atom_labels = torch.tensor(atom_labels.astype(np.float64), dtype=torch.float)
            
        atom_features = atom_features.to(self.device)
        bond_features = bond_features.to(self.device)
        atom_labels = atom_labels.to(self.device)

        y_pred = self(array_rep, atom_features, bond_features)
        #y_pred = y_pred.squeeze(1)
        #atom_labels = atom_labels.unsqueeze(1)

        loss = F.mse_loss(y_pred, atom_labels)
        r2score = R2Score()
        r2score = R2Score().to(self.device)
        r2 = r2score(y_pred, atom_labels)
        mean_absolute_error=MeanAbsoluteError()
        mean_absolute_error=MeanAbsoluteError().to(self.device)
        MAE_loss=mean_absolute_error(y_pred, atom_labels)
        # https://github.com/pytorch/ignite/issues/453
        #var_y = torch.var(atom_labels, unbiased=False)
        #r2 = 1.0 - F.mse_loss(y_pred, atom_labels, reduction="mean") / var_y
        

        ret = {'loss': loss, 'test_r2': r2,'y_pred':y_pred,'atom_labels':atom_labels,'MAE_loss':MAE_loss}
        
        return ret

    def test_epoch_end(self, outputs):

        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('test_loss', loss)

        MAE_loss = torch.stack([x['MAE_loss'] for x in outputs]).mean()
        self.log('test_mae_loss', MAE_loss)

        r2 = torch.stack([x['test_r2'] for x in outputs]).mean()
        self.log('test_r2', r2)
        

        test_y_pred=[]
        for x in outputs:
            y_pred=x['y_pred'].squeeze(1).detach().cpu().numpy()
            test_y_pred.append(y_pred)
        #print("test_y_pred",len(test_y_pred))
        
        test_atom_labels=[]
        for x in outputs:
            atom_labels=x['atom_labels'].squeeze(1).detach().cpu().numpy()
            test_atom_labels.append(atom_labels)
        #print("test_labels",len(test_atom_labels))

        os.chdir("/home/tsutsumi/fukui-gcn-test/")
        with open("test_pred.csv",'w') as f:
            for i in range(len(test_y_pred)):
                test_pred=test_y_pred[i]
                for j in range(len(test_pred)):
                    print(test_pred[j],file=f)

        with open("test_labels.csv",'w') as f:
            for i in range(len(test_atom_labels)):
                test_labels=test_atom_labels[i]
                for j in range(len(test_labels)):
                    print(test_labels[j],file=f)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return optimizer

