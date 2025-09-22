import subprocess
import argparse
import os
import time

def run_experiment(config_file, additional_args=None):
    if not os.path.exists(config_file):
        print(f"Error: Config file not found: {config_file}")
        return 1, 0
    
    cmd = ["python", "main.py", "--config", config_file]
    
    if additional_args:
        cmd.extend(additional_args)

    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=False, capture_output=False)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nExperiment completed: {config_file}")
        print(f"Execution time: {execution_time/3600:.2f} hours")
        
        return result.returncode, execution_time
        
    except KeyboardInterrupt:
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\n\nExperiment interrupted: {config_file}")
        print(f"Execution time before interruption: {execution_time:.2f} seconds")
        return -1, execution_time

def main():
    parser = argparse.ArgumentParser(description='Run multiple experiments with different configurations')
    parser.add_argument('--data_root_path', type=str, default=None,
                        help=f'Path to the data root directory.')
    parser.add_argument('--optuna', action='store_true', default=False,
                        help='Enable Optuna optimization for all configs')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of Optuna trials (only used with --optuna)')
    parser.add_argument("--max_failed_trials", type=int, default=None,
                        help="Stop after N failed trials")
    parser.add_argument('--early_stop_patience', type=int, default=None,
                        help='Early stop patience for Optuna (only used with --optuna)')
    parser.add_argument('--max_time_hours', type=float, default=None,
                        help='Max time in hours for Optuna (only used with --optuna)')
    parser.add_argument('--reset', action='store_true', default=False,
                        help='Reset training and start from scratch')
    parser.add_argument('--no_alignment', action=argparse.BooleanOptionalAction, default=False,
                        help='Disable classifier alignment.')
    parser.add_argument('--lca', action=argparse.BooleanOptionalAction, default=False,
                        help='Use LCA for feature augmentation.')
    parser.add_argument('--override_seed', type=int, default=-1,
                        help='Random seed for reproducibility.')
    
    args = parser.parse_args()
    
    additional_args = []
    if args.data_root_path:
        additional_args.extend(['--data_root_path', args.data_root_path])
    if args.optuna:
        additional_args.append('--optuna')
        if args.n_trials:
            additional_args.extend(['--n_trials', str(args.n_trials)])
        if args.max_failed_trials:
            additional_args.extend(['--max_failed_trials', str(args.max_failed_trials)])
        if args.early_stop_patience:
            additional_args.extend(['--early_stop_patience', str(args.early_stop_patience)])
        if args.max_time_hours:
            additional_args.extend(['--max_time_hours', str(args.max_time_hours)])
    if args.reset:
        additional_args.append('--reset')
    if args.no_alignment:
        additional_args.append('--no_alignment')
    if args.lca:
        additional_args.append('--lca')
    if args.override_seed != -1:
        additional_args.extend(['--override_seed', str(args.override_seed)])

    METHODS = [
        # "aper_aperpter",
        # "aper_finetune",
        # "aper_ssf",
        # "aper_vpt_deep",
        # "aper_vpt_shallow",
        # "l2p",
        # "coda_prompt",
        # "dualprompt",
        # "slca",
        "mos",
        # "ease"
    ]
    DATASETS = [
        # "cifar",
        # "inr",
        # "ina",
        # "cub",
        # "omni",
        # "vtab",
        # "cars"
    ]

    try:
        CONFIGS = []
        for method in METHODS:
            for dataset in DATASETS:
                config_file = f"exps/{method}_{dataset}.json" if dataset != "cifar" else f"exps/{method}.json"
                if not os.path.exists(config_file):
                    print(f"Warning: Config file does not exist and will be skipped: {config_file}")
                    continue
                CONFIGS.append(config_file)
        for i, config_file in enumerate(CONFIGS, 1):
            print(f"\n\nProgress: {i}/{len(CONFIGS)}")
            run_experiment(config_file, additional_args)
    
    except KeyboardInterrupt:
        print("\n\nBatch execution interrupted by user!")

if __name__ == '__main__':
    main()