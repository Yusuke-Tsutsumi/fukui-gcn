#!/usr/bin/env python
# coding: utf-8

# In[1]:


model_params = {
    'random_seed': 42,
    'num_epochs': 75,#original=30
    'batch_size': 149,
    'use_gpu': True, # use gpu or not
    'atom_features_size': 12, # fixed value
    'conv_layer_sizes': [198,198,198,198,198,198,198],  # convolution layer sizes
    'mlp_layer_sizes': [183, 1], # multi layer perceptron sizes
    'lr': 0.01, #learning late
    'metrics': 'r2', # the metrics for 'check_point' , 'early_stopping', 'hyper'
    'minimize': False, # True if you want to minimize the 'metrics'
    'check_point':
        {
        "dirpath": 'trained_model', # model save path
        "save_top_k": 3, # save top k metrics model
        },
    'early_stopping': # see https://pytorch-lightning.readthedocs.io/en/stable/generated/pytorch_lightning.callbacks.EarlyStopping.html
        {
        "min_delta": 0.000000,
        "patience": 50,#何回連続して停滞したら止まるか
        "verbose": True,
        },
    'init_points':50,#初期観測点の数
    'n_iter':30,#何点評価するか
    'hyper':
        {
        'batch_size': (10, 150),
        'conv_layer_width':(1,20),
        'conv_layer_size': (5, 200),
        'mlp_layer_size': (100, 300), 
        'lr': (0.001, 0.5),
        }
}


# In[ ]:




