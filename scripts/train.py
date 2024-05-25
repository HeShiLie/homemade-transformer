# author: HeShiLie
# md

import numpy as np
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer as hmtf

import argparse

# config loading function
def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

# set the random seed
if __name__ == '__main__':
    torch.manual_seed(0)

    # set the argument parser
    parser = argparse.ArgumentParser(description='Train the transformer model')
    parser.add_argument('--config_file', type=str, default='./config/config.yaml', help='the config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='the checkpoint file')
    parser.add_argument('--output', type=str, default='output', help='the output directory')
    args = parser.parse_args()

    # load the config file