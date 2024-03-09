# Revised from 'train.py': To apply to the treadmill data
# This code is based on https://github.com/victoresque/pytorch-template

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_230703 import get_model
from optimizer_230703 import get_optimizer
from data import get_loader
import masked_loss
import trainers

import argparse
import collections
import time

import numpy as np
import scipy.stats as scistats
import json

def main(config):
    device = config['device']['type']
    seed = config['seed']

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

    core_cfg = config['arch']['core']

    # get train loader(s) and valid loader(s)
    train_loaders = get_loader(config['data_loader']['train_loader_args'])
    valid_loaders = get_loader(config['data_loader']['valid_loader_args'])

    core = get_model(core_cfg)
    optim_core = get_optimizer(core, **dict(config['optimizer']))
    
    core = core.to(device)

    # loss
    criterion = getattr(masked_loss, config['loss'])
    trainer = getattr(trainers, config['trainer']['type']).Trainer(
        model={
            'core': core,
        },
        criterion=criterion,
        optimizer={
            'core': optim_core,
        },
        data_loader={
            'train': train_loaders,
            'valid': valid_loaders,
        },
        config=config
    )

    trainer.train()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')

    args = argparser.parse_args()
    with open(args.config) as config_json:
        config = json.load(config_json)

    main(config)