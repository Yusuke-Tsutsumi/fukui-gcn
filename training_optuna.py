#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import runpy
import torch
from torch.utils import data
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import optuna


# In[ ]:


from mol_label_load import SDFDataLoader, TestDataLoader, TestDataLoader_ChEMBL
from dataset import MoleculeDataset, gcn_collate_fn
from graph_conv_model import GraphConvModel


# In[ ]:


def print_result1(trainer):
    # print(trainer.callback_metrics)
    print("train mse loss={0}".format(trainer.callback_metrics['train_loss']))
    print("validation mse loss={0}".format(trainer.callback_metrics['val_loss']))
    print("train mae loss={0}".format(trainer.callback_metrics['train_mae_loss']))
    print("validation mae loss={0}".format(trainer.callback_metrics['val_mae_loss']))
    print("train r2={0}".format(trainer.callback_metrics['train_r2'].item()))
    print("validation r2={0}".format(trainer.callback_metrics['val_r2'].item()))

def print_result2(trainer):
    # print(trainer.callback_metrics)
    print("test mse loss={0}".format(trainer.callback_metrics['test_loss']))
    print("test mae loss={0}".format(trainer.callback_metrics['test_mae_loss']))
    print("test r2={0}".format(trainer.callback_metrics['test_r2'].item()))
# In[ ]:




def evaluation_function(trail):
    global parameters

    # reading hyper parameters
    parameters={
        'batch_size' :trail.suggest_int('batch_size',10,200),
        'conv_layer_width' : trail.suggest_int('conv_layer_width',1,20),
        'conv_layer_size' :trail.suggest_int('conv_layer_size',5,200),
        'mlp_layer_size' :trail.suggest_int('mlp_layer_size',100,300),
        'lr' : trail.suggest_float('lr',0.0,0.5)
    }

    

    data_loader_train = data.DataLoader(params["molecule_dataset_train"], batch_size=parameters["batch_size"], shuffle=False,
                                        collate_fn=gcn_collate_fn)

    data_loader_val = data.DataLoader(params["molecule_dataset_val"], batch_size=parameters["batch_size"], shuffle=False,
                                      collate_fn=gcn_collate_fn)

    print("batch_size={0}".format(parameters["batch_size"]))
    print("conv_layer_width={0}".format(parameters["conv_layer_width"]))
    print("conv_layer_size={0}".format(parameters["conv_layer_size"]))
    print("mlp_layer_size={0}".format(parameters["mlp_layer_size"]))

    conv_layer_sizes = []
    for i in range(parameters["conv_layer_width"]):
        conv_layer_sizes.append(parameters["conv_layer_size"])

    model = GraphConvModel(
        device_ext=params["device"],
        atom_features_size=params["atom_features_size"],
        conv_layer_sizes=conv_layer_sizes,
        mlp_layer_sizes=[parameters["mlp_layer_size"], 1],
        lr=parameters["lr"]
    )

    callbacks = []
    if params["es"]:
        early_stop_callback = EarlyStopping(
            min_delta=params["early_stopping"]["min_delta"],
            patience=params["early_stopping"]["patience"],
            verbose=params["early_stopping"]["verbose"],
            monitor="val_" + params["metrics"],
            mode='min' if params["minimize"] else 'max'
        )
        callbacks.append(early_stop_callback)

    trainer = pl.Trainer(
        max_epochs=params["num_epochs"],
        gpus=params["gpu"],
        callbacks=callbacks
    )
    trainer.fit(model, data_loader_train, data_loader_val)
    print_result1(trainer)

    key = "val_{0}".format(params["metrics"])
    return trainer.callback_metrics[key].item()




# In[ ]:


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-config", type=str, required=True)
    parser.add_argument("-mode", type=str, default="train")
    parser.add_argument("-hyper", action='store_true')
    parser.add_argument("-es", action='store_true')
    parser.add_argument("-extra", action='store_true')
    args = parser.parse_args()

    global params

    params = runpy.run_path(args.config).get('model_params', None)
    pl.seed_everything(params["random_seed"])




    from sklearn.model_selection import train_test_split

    if args.extra:
        #input extra data
        mols_list,labels_list=SDFDataLoader()
        #test_mols_list, test_labels_list=TestDataLoader()
        test_mols_list, test_labels_list=TestDataLoader_ChEMBL()
        print("test ChEMBL data:",len(test_mols_list))
        #use_X,unuse_X,use_y,unuse_y=train_test_split(mols_list, labels_list, shuffle=True, train_size=0.5, random_state=params["random_seed"])
        #print(len(use_X))
        #print(len(unuse_X))
        #print(len(use_y))
        #print(len(unuse_y))
        #X_train, X_val, y_train, y_val = train_test_split(use_X, use_y, shuffle=True, train_size=0.8, random_state=params["random_seed"])
        X_train, X_val, y_train, y_val = train_test_split(mols_list, labels_list, shuffle=True, train_size=0.8, random_state=params["random_seed"])
        molecule_dataset_train = MoleculeDataset(X_train, y_train)
        molecule_dataset_val = MoleculeDataset(X_val, y_val)
        molecule_dataset_test = MoleculeDataset(test_mols_list,test_labels_list)
    else:
        mols_list,labels_list=SDFDataLoader()
        #use_X,unuse_X,use_y,unuse_y=train_test_split(mols_list, labels_list, shuffle=True, train_size=0.00390625, random_state=params["random_seed"])
        #print(len(use_X))
        #print(len(unuse_X))
        #print(len(use_y))
        #print(len(unuse_y))
        #X_train1, X_test, y_train1, y_test = train_test_split(use_X, use_y, shuffle=True, train_size=0.8, random_state=params["random_seed"])
        X_train1, X_test, y_train1, y_test = train_test_split(mols_list, labels_list, shuffle=True, train_size=0.8, random_state=params["random_seed"])
        X_train2, X_val, y_train2, y_val = train_test_split(X_train1, y_train1, shuffle=True, train_size=0.8, random_state=params["random_seed"])
        molecule_dataset_train = MoleculeDataset(X_train2, y_train2)
        molecule_dataset_val = MoleculeDataset(X_val, y_val)
        molecule_dataset_test = MoleculeDataset(X_test,y_test)

    if params["use_gpu"] and torch.cuda.is_available():
        device = torch.device('cuda:0')
        gpu = 1
    else:
        device = torch.device('cpu')
        gpu = 0

    callbacks = [EarlyStopping(monitor='val_loss')]
    if args.hyper:
        params["molecule_dataset_train"] = molecule_dataset_train
        params["molecule_dataset_val"] = molecule_dataset_val
        params["device"] = device
        params["gpu"] = gpu
        params["es"] = True if args.es else False


        
        sampler=optuna.samplers.MOTPESampler(seed=params["random_seed"])
        study=optuna.create_study(sampler=sampler,direction='maxmize')
        study.optimize(evaluation_function,n_trials=150)
        opted_params=study.best_params
        opted_params['batch_size']=int(opted_params['batch_size'])
        opted_params['conv_layer_width']=int(opted_params['conv_layer_width'])
        opted_params['conv_layer_size']=int(opted_params['conv_layer_size'])
        opted_params['mlp_layer_size']=int(opted_params['mlp_layer_size'])
        print(opted_params)
        print('best score : ',study.best_value)
    else:
        data_loader_train = data.DataLoader(molecule_dataset_train, batch_size=params["batch_size"], shuffle=False,
                                            collate_fn=gcn_collate_fn)

        data_loader_val = data.DataLoader(molecule_dataset_val, batch_size=params["batch_size"], shuffle=False,
                                            collate_fn=gcn_collate_fn)

        data_loader_test = data.DataLoader(molecule_dataset_test, batch_size=params["batch_size"], shuffle=False,
                                            collate_fn=gcn_collate_fn)

        model = GraphConvModel(
         device_ext=device,
         atom_features_size=params["atom_features_size"],
         conv_layer_sizes=params["conv_layer_sizes"],
         mlp_layer_sizes=params["mlp_layer_sizes"],
         lr=params["lr"]
        )

        callbacks = []
        checkpoint_callback = ModelCheckpoint(
            monitor='val_' + params["metrics"],
            dirpath=params["check_point"]["dirpath"],
            filename="model-{epoch:02d}-{val_" + params["metrics"]+":.5f}",
            save_top_k=params["check_point"]["save_top_k"],
            mode='min' if params["minimize"] else 'max'
        )
        callbacks.append(checkpoint_callback)

        if args.es:
            early_stop_callback = EarlyStopping(
            min_delta=params["early_stopping"]["min_delta"],
            patience=params["early_stopping"]["patience"],
            verbose=params["early_stopping"]["verbose"],
            monitor="val_"+ params["metrics"],
            mode='min' if params["minimize"] else 'max'
            )
            callbacks.append(early_stop_callback)

        trainer = pl.Trainer(
             max_epochs=params["num_epochs"],
             gpus=gpu,
             callbacks=callbacks
        )
        trainer.fit(model, data_loader_train, data_loader_val)
        print_result1(trainer)
        trainer.test(model, data_loader_test)
        print_result2(trainer)
        #with open ("train_result.csv",'w') as f:
            #with open ("val_result.csv",'w') as f:
                #trainer.fit(model, data_loader_train, data_loader_val)
                #print_result(trainer)
                #print(trainer.callback_metrics['train_y_pred'].item(),file=f)
                #print(trainer.callback_metrics['val_y_pred'].item(),file=f)

# In[ ]:


if __name__ == "__main__":
    main()

