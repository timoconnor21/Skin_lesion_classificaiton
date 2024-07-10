# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 10:58:27 2024

@author: TimOConnor
"""

from datetime import date
import os


"""
Configuration file for setting up the model for training. 

'train_model': Boolean to indicate if a model should be trained. If false, the previous best performing model is loaded
'post_processing': Boolean to indicate if post-processing should be applied
'use_simple_CNN': Boolean to indicate if the simpleCNN architecture should be used. If False, a pretrained ResNet is used.
'seed': Variable to set the random seed for reproducibility.
'model_name': Optional variable for specifying a model name. If no name is provided, today's date is used.
'max_epochs': Maximum number of epochs during model training. 
'initial_lr': Initial learn rate for the optimization algorithm
'batch_size': Batch size for passing images to the model
'early_stop_patience': Number of epochs to train without improvement before stopping.
'train_frac': Fraction of training data to be used. Can be reduced for debugging/prototyping
'val_frac': Fraction of the training data to be used as validation data.
'classes': Class names corresponding to the subfolders within 'test' and 'train' folders.


"""

def get_config():
    main_dir = r'C:\Users\TimOConnor\Repos\Tristar\Tristar_take_home'
    os.chdir(main_dir)
    
    config = {
        'train_model': False,
        'post_processing': True,
        'use_simple_CNN': False,
        'seed': 42,
        'model_name': None,
        'max_epochs': 200,
        'initial_lr': .001,
        'batch_size': 64,
        'early_stop_patience': 10,
        'train_frac': 1,
        'val_frac': .2,
        'classes': ["Benign", "Malignant"],
        
        }
    
    if config['model_name'] == '' or config['model_name'] == None:
        #use default name of today's data
        config['model_name'] = date.today().strftime('%Y-%m-%d')
    
    return config


if __name__ == '__main__':
    config = get_config()