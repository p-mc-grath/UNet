import os
import numpy as np
import torch
import tensorflow as tf
import json
import warnings
from easydict import EasyDict as edict
from six.moves import cPickle as pickle
from datetime import datetime
from os import listdir
from os.path import join, isfile, isdir
from pathlib import Path

############################################################################
# json helper functions
############################################################################

def load_json_file(filepath):
    '''
    simply loads json file

    Arguments:
        filepath: full path path incl. filename and extension
    
    return:
        dict according to json file
    '''

    if isfile(filepath):
        with open(filepath, 'r') as jf:
                json_file = json.load(jf)
    else:
        raise FileNotFoundError

    return json_file

def save_json_file(filepath, save_file, indent=None):
    '''
    simply saves json file

    Arguments:
        filepath: full path path incl. filename and extension
        save_file: file to save
        indent: allows pretty json file for human readability | None most dense representation
    '''

    with open(filepath, 'w') as jf:
        json.dump(save_file, jf, indent=indent)

    print('Successfully saved ' + filepath)
    return 1

############################################################################
# Config functions
############################################################################

def load_config(loading_dir, file_name):
    '''
    tries to load config from json file

    Arguments:
        loading_dir: /path/to/repo/config
        file_name: file name with .json extension
    
    return:
        if exists: config
        else: None
    '''

    json_file = join(loading_dir, file_name)
     
    if isfile(json_file):
        # load
        config = load_json_file(json_file)

        return config
    else:
        return None

def save_config(config, file_name='config.json'):
    '''
    saves to json formatted with indent = 4 | human readable
    '''
    
    # save to json file | pretty with indent
    Path(config.dir.configs).mkdir(exist_ok=True)
    save_json_file(os.path.join(config.dir.configs, file_name), config, indent=4)

def create_config(root_dir):
    '''
    create according to
    https://github.com/moemen95/Pytorch-Project-Template/tree/4d2f7bea9819fe2e5e25153c5cc87c8b5f35f4b8
    put into python for convenience with directories

    Arguments:
        root_dir: /path/to/repo
    
    return:
        dictionary specifying all parameters in the setup
    '''

    # overall root dir
    config = {
        'dir': {
            'root': root_dir
        }
    }

    # all script names
    config['scripts'] = {
        'model': 'U_Net.py',
        'utils': 'U_Net_utils.py',
        'agent': 'U_Net_Agent.py',
        'dataset': 'MISData.py'
    }
    
    # model params
    config['model'] = {
        'memory_efficient': False,
        'encoder': [
            {'layers': [3, 64, 64], 'kernel_size': 3, 'stride': 1, 'batch_norm': False}, 
            {'layers': [64, 128, 128], 'kernel_size': 3, 'stride': 1, 'batch_norm': False}, 
            {'layers': [128, 256, 256], 'kernel_size': 3, 'stride': 1, 'batch_norm': False}, 
            {'layers': [256, 512, 512], 'kernel_size': 3, 'stride': 1, 'batch_norm': False}
        ],
        'bottom': {'layers': [512, 1024, 1024], 'kernel_size': 3, 'stride': 1, 'batch_norm': False},
        'decoder': [
            {'layers': [1024, 512, 512], 'kernel_size': 3, 'stride': 1, 'batch_norm': False}, 
            {'layers': [512, 256, 256], 'kernel_size': 3, 'stride': 1, 'batch_norm': False}, 
            {'layers': [256, 128, 128], 'kernel_size': 3, 'stride': 1, 'batch_norm': False}, 
            {'layers': [128, 64, 64, 1], 'kernel_size': [3,3,1], 'stride': 1, 'batch_norm': False}
        ]
    }

    # loader params
    config['loader'] = {
        'mode': 'train',
        'batch_size': 32,
        'pin_memory': True,                                                 
        'num_workers': 4,
        'async_loading': True,                                              
        'drop_last': True                                              # needs to be False if batch_size None
    }

    # optimizer params; currently torch.optim.Adam default
    config['optimizer'] = {
        'learning_rate': 1e-3,
        'beta1': 0.9,
        'beta2': 0.999,
        'eps': 1e-08,
        'amsgrad': False,
        'weight_decay': 0,
        'lr_scheduler': {
            'want': False,
            'every_n_epochs': 30,
            'gamma': 0.1
        }
    }

    # waymo dataset info
    config['dataset'] = {
        'batch_size': 1,                                                  # > 1 if serialized in batch -> laoder.batch_size = None
        'images': {
            'original.size': (3, 1920, 1280),
            'size': (3, 192, 128)
        },
        'datatypes': ['rgb', 'depth', 'normals'],
        'file_list_name': 'file_list.json'
    }

    # agent params
    config['agent'] = {
        'seed': 123,                                                        # fixed random seed ensures reproducibility
        'max_epoch': 100,
        'checkpoint': {                                                     # naming in checkpoint dict
            'epoch': 'epoch',
            'train_iteration': 'train_iteration',
            'val_iteration': 'val_iteration',
            'best_val_acc': 'best_val_acc',
            'state_dict': 'state_dict',
            'optimizer': 'optimizer'
        },
        'best_checkpoint_name': 'best_checkpoint.pth.tar'
    }

    # create subdirs according to pytorch project template: https://github.com/moemen95/Pytorch-Project-Template/tree/4d2f7bea9819fe2e5e25153c5cc87c8b5f35f4b8
    for subdir in ['agents', 'graphs', 'utils', 'datasets', 'experiments', 'configs']:
        config['dir'][subdir] = join(config['dir']['root'], subdir)
    config['dir']['graphs'] = {'models': join(config['dir']['graphs'], 'models')}

    # Current run: tensorBoard summary writers dir and checkpoint dir
    current_run = datetime.now().strftime('%Y-%m-%d-%H-%M')
    config['dir']['current_run'] = join(config['dir']['experiments'], current_run)     

    # directories of data
    config['dir']['data'] = {'root': join(config['dir']['root'], 'data')}
    for mode in ['train', 'val', 'test']:
        config['dir']['data'][mode] = {}
        for datatype in config['dataset']['datatypes']:
                config['dir']['data'][mode][datatype] = join(config['dir']['data']['root'], mode, datatype)
    
    return config

def get_config(config_dir='', file_name='config.json'):
    '''
    load from json file or create config

    Arguments:
        config_dir: /path/to/repo/config
        file_name: actual file name with .json extension
    
    return:
        easydict dictionary specifying all parameters in the setup
    '''

    config = load_config(config_dir, file_name)
    
    if config is None:
        if config_dir.endswith('config'):
            root_dir = join(*config_dir.split(os.sep)[:-1])
            if not root_dir.startswith(os.sep) or not root_dir.startswith('~'):             # colab compatibility
                root_dir = os.sep + root_dir
            config = create_config(root_dir)
        else:
            raise ValueError('You have to provide a pathh according to the specifications: path/to/repo/config')

    return edict(config)

############################################################################
# Metrics
############################################################################

def compute_accuracy(prediction, ground_truth, threshold = .1):
    '''
    computes pixel-wise difference between prediction and ground_truth
    if difference < threshold -> TP or TN
    computes accuracy score accordingly: (TP+TN)/(All)
    
    Arguments:
        ground_truth: ground truth map of one sample/ batch of maps: y, x
        prediction: heatmap of one sample/ batch of maps: y, x
        threshold: used to threshold the difference between prediction and gt
    
    return:
        class-wise accuracy
    '''

    number_correctly_predicted_pixels = torch.sum(abs(prediction-ground_truth) < threshold)
    number_all_pixels = ground_truth.numel()

    # (TP+TN)/(All)
    acc = number_correctly_predicted_pixels/number_all_pixels
    
    return acc

############################################################################
# Helper
############################################################################

def crop_around_center(shape_is, shape_should):
    '''
    Computes slices such that image/ batch of images is cropped around center

    Arguments:
        shape_is: torch size
        shape_should: torch size
    return:
        list of slices
    '''

    crop_y, crop_x = (torch.tensor(shape_is[-2:])-torch.tensor(shape_should[-2:]))//2
    return [slice(None,None), slice(None, None), slice(crop_y,shape_is[-2]-crop_y), slice(crop_x,shape_is[-1]-crop_x)]
