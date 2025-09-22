import json
import argparse
import os
import logging
import yaml
from trainer import train
from trainer_optuna import train as train_optuna
from utils.data import set_data_root_path, get_data_root_path, DEFAULT_DATA_ROOT_PATH

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args) # Converting argparse Namespace to a dict.
    # args.update(param) # Add parameters from json

    merged = {**param, **args}
    
    data_root = args.get('data_root_path') or param.get('data_root_path') or DEFAULT_DATA_ROOT_PATH
    print(f"Using data root path: {data_root}")
    set_data_root_path(data_root)
    os.makedirs(get_data_root_path(), exist_ok=True)

    if args.override_seed != -1:
        merged['seed'] = args.override_seed

    if merged["model_name"] == "mos":
        merged["max_iter"] = 1
        merged["ensemble"] = False

    if merged['optuna']:
        pruning_thresholds = load_thresholds_config(
            merged.get('model_name'), 
            merged.get('dataset')
        )
        merged['use_ori'] = False
        merged['prefix'] = 'optuna_lca'
        train_optuna(merged, pruning_thresholds)
    else:
        if merged['lca']:
            merged['prefix'] = 'lca'
            merged['use_ori'] = False
        else:
            merged['prefix'] = 'base'
            merged['use_ori'] = True
            if merged['no_alignment']:
                merged['prefix'] += '_noalign'
        train(merged)

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def load_thresholds_config(method_name, dataset_name):
    thresholds_file = './exps/pruning_thresholds.yaml'
    
    if not os.path.exists(thresholds_file):
        print(f"Thresholds file not found: {thresholds_file}")
        return {}
    
    try:
        with open(thresholds_file, 'r') as data_file:
            thresholds_config = yaml.safe_load(data_file)
    except Exception as e:
        print(f"Error loading thresholds YAML file: {e}")
        return {}
    
    if not thresholds_config:
        return {}
    
    if method_name in thresholds_config:
        method_thresholds = thresholds_config[method_name]
        if dataset_name in method_thresholds:
            print(f"Loaded pruning thresholds: {method_thresholds[dataset_name]}")
            return method_thresholds[dataset_name]
    
    print(f"No thresholds found for {method_name} on {dataset_name}")
    return {}

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple pre-trained incremental learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/simplecil.json',
                        help='Json file of settings.')
    parser.add_argument('--data_root_path', type=str, default=None,
                        help=f'Path to the data root directory. Default: {DEFAULT_DATA_ROOT_PATH}')
    parser.add_argument('--reset', action=argparse.BooleanOptionalAction, default=False,
                        help='Reset the training and start from scratch.')
    parser.add_argument('--lca', action=argparse.BooleanOptionalAction, default=False,
                        help='Use LCA for feature augmentation.')
    parser.add_argument('--optuna', action=argparse.BooleanOptionalAction, default=False,
                        help='Use optuna for hyperparameter optimization.')
    parser.add_argument("--n_trials", type=int, default=100,
                        help="Number of optimization trials")
    parser.add_argument("--max_failed_trials", type=int, default=None,
                        help="Stop after N failed trials")
    parser.add_argument("--early_stop_patience", type=int, default=20,
                        help="Stop after N trials without improvement")
    parser.add_argument("--max_time_hours", type=float, default=None,
                        help="Stop after N hours")
    parser.add_argument('--no_alignment', action=argparse.BooleanOptionalAction, default=False,
                        help='Disable classifier alignment.')
    parser.add_argument('--override_seed', type=int, default=-1,
                        help='Random seed for reproducibility.')
    return parser

if __name__ == '__main__':
    main()
