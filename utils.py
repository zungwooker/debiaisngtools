import time
import os
import torch
import numpy as np
import argparse
import yaml


def safe_save(file_name, extended_name):
    new_file_name = file_name
    if os.path.isfile(f'{new_file_name}{extended_name}'): new_file_name += f'_{time.strftime("%m.%d-%H:%M:%S")}' 
    
    return  new_file_name


def yaml_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', help="configuration file *.yaml", type=str, required=False, default='config.yaml')
    args = parser.parse_args()
    
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    for key, value in config.items():
        parser.add_argument(f"--{key}", type=type(key), default=value)
        
    args = parser.parse_args()
    
    return args


def option_check(args):
    prompt = "Continue Training? (y/n): "

    print("\n==============================")
    print("Process details:")
    print(("------------------------------"))
    print(f"seed: {args.seed}")
    print(f"cuda: {args.cuda}")
    print(f"device: {args.device} \n")

    print(("------------------------------"))
    print("Training details:")
    print(("------------------------------"))
    print(f"epochs: {args.epochs}")
    print(f"LR: {args.lr}")
    print(f"batch size: {args.batch_size}")
    print(f"dataset: {args.dataset}")
    print(f"conflict ratio: {args.conflict_ratio}")
    print(f"save: {args.save} \n")

    print(("------------------------------"))
    print("Logging details:")
    print(("------------------------------"))
    print(f"remote: {args.remote}")
    print(f"run name: {args.run_name}")
    print(f"proj name: {args.project_name}")
    print("==============================\n")

    while True:
        response = input(prompt)
        if response == 'y':
            return True
        elif response == 'n':
            return False
        else:
            continue