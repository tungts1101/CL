import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import numpy as np
import random


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    # Store results from all seeds
    all_seed_results = []
    
    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        seed_results = _train(args)
        all_seed_results.append(seed_results)
    
    # Calculate mean and std across seeds
    if len(all_seed_results) > 1:
        _calculate_seed_statistics(args, all_seed_results)


def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["backbone_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )
    
    args["nb_classes"] = data_manager.nb_classes # update args
    args["nb_tasks"] = data_manager.nb_tasks
    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []
    acc_history = []

    for task in range(data_manager.nb_tasks):
        # logging.info("All params: {}".format(count_parameters(model._network)))
        # logging.info(
        #     "Trainable params: {}".format(count_parameters(model._network, True))
        # )
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]    
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            nme_keys = [key for key in nme_accy["grouped"].keys() if '-' in key]
            nme_values = [nme_accy["grouped"][key] for key in nme_keys]
            nme_matrix.append(nme_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))
            
            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
            logging.info("Average Accuracy (NME): {}".format(sum(nme_curve["top1"])/len(nme_curve["top1"])))

            acc_history.append(sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

            logging.info("Average Accuracy (CNN): {} \n".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))

            acc_history.append(sum(cnn_curve["top1"])/len(cnn_curve["top1"]))

    acc_history = [float(np.round(v, 2)) for v in acc_history]
    logging.info(f"Final accuracy history: {acc_history}")

    # Prepare results to return
    results = {
        'seed': args['seed'],
        'acc_history': acc_history,
        'cnn_curve': cnn_curve,
        'nme_curve': nme_curve if nme_curve["top1"] else None,
        'cnn_matrix': cnn_matrix,
        'nme_matrix': nme_matrix if nme_matrix else None,
        'final_avg_acc_cnn': sum(cnn_curve["top1"])/len(cnn_curve["top1"]) if cnn_curve["top1"] else 0,
        'final_avg_acc_nme': sum(nme_curve["top1"])/len(nme_curve["top1"]) if nme_curve["top1"] else None
    }

    if 'print_forget' in args.keys() and args['print_forget'] is True:
        if len(cnn_matrix) > 0:
            np_acctable = np.zeros([task + 1, task + 1])
            for idxx, line in enumerate(cnn_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
            print('Accuracy Matrix (CNN):')
            print(np_acctable)
            logging.info('Forgetting (CNN): {}'.format(forgetting))
        if len(nme_matrix) > 0:
            np_acctable = np.zeros([task + 1, task + 1])
            for idxx, line in enumerate(nme_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
            print('Accuracy Matrix (NME):')
            print(np_acctable)
        logging.info('Forgetting (NME): {}'.format(forgetting))
    
    return results


def _calculate_seed_statistics(args, all_seed_results):
    """Calculate mean and standard deviation across multiple seeds"""
    
    # Setup logging for statistics without affecting existing handlers
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"], args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    stats_logfilename = "logs/{}/{}/{}/{}/{}_STATS_{}".format(
        args["model_name"],
        args["dataset"], 
        init_cls,
        args["increment"],
        args["prefix"],
        args["backbone_type"]
    )
    
    # Create a separate logger for statistics
    stats_logger = logging.getLogger('statistics')
    stats_logger.setLevel(logging.INFO)
    stats_logger.propagate = False  # Prevent propagation to root logger
    
    # Remove any existing handlers for this logger
    for handler in stats_logger.handlers[:]:
        stats_logger.removeHandler(handler)
    
    # Add file handler for statistics
    stats_file_handler = logging.FileHandler(filename=stats_logfilename + ".log")
    stats_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(filename)s] => %(message)s"))
    stats_logger.addHandler(stats_file_handler)
    
    # Add console handler for statistics
    stats_console_handler = logging.StreamHandler(sys.stdout)
    stats_console_handler.setFormatter(logging.Formatter("%(asctime)s [%(filename)s] => %(message)s"))
    stats_logger.addHandler(stats_console_handler)

    seeds = [result['seed'] for result in all_seed_results]
    stats_logger.info(f"=== SEED STATISTICS ACROSS {len(seeds)} SEEDS: {seeds} ===")
    
    # Calculate CNN statistics
    cnn_final_accs = [result['final_avg_acc_cnn'] for result in all_seed_results]
    cnn_mean = np.mean(cnn_final_accs)
    cnn_std = np.std(cnn_final_accs, ddof=1) if len(cnn_final_accs) > 1 else 0.0
    
    stats_logger.info(f"CNN Final Average Accuracies: {[round(acc, 2) for acc in cnn_final_accs]}")
    stats_logger.info(f"CNN Mean ± Std: {cnn_mean:.2f} ± {cnn_std:.2f}")
    
    # Calculate NME statistics if available
    nme_final_accs = [result['final_avg_acc_nme'] for result in all_seed_results if result['final_avg_acc_nme'] is not None]
    if nme_final_accs:
        nme_mean = np.mean(nme_final_accs)
        nme_std = np.std(nme_final_accs, ddof=1) if len(nme_final_accs) > 1 else 0.0
        stats_logger.info(f"NME Final Average Accuracies: {[round(acc, 2) for acc in nme_final_accs]}")
        stats_logger.info(f"NME Mean ± Std: {nme_mean:.2f} ± {nme_std:.2f}")
    
    # Calculate task-wise statistics
    max_tasks = max(len(result['acc_history']) for result in all_seed_results)
    task_stats = []
    
    for task_idx in range(max_tasks):
        task_accs = []
        for result in all_seed_results:
            if task_idx < len(result['acc_history']):
                task_accs.append(result['acc_history'][task_idx])
        
        if task_accs:
            task_mean = np.mean(task_accs)
            task_std = np.std(task_accs, ddof=1) if len(task_accs) > 1 else 0.0
            task_stats.append((task_mean, task_std))
            stats_logger.info(f"Task {task_idx + 1} Accuracy: {task_mean:.2f} ± {task_std:.2f}")
    
    # Log final summary
    stats_logger.info("=" * 60)
    stats_logger.info(f"FINAL RESULTS SUMMARY ({args['model_name']} on {args['dataset']})")
    stats_logger.info("=" * 60)
    stats_logger.info(f"Seeds: {seeds}")
    stats_logger.info(f"CNN Final Accuracy: {cnn_mean:.2f} ± {cnn_std:.2f}")
    if nme_final_accs:
        stats_logger.info(f"NME Final Accuracy: {nme_mean:.2f} ± {nme_std:.2f}")
    stats_logger.info("=" * 60)
    
    # Clean up handlers
    for handler in stats_logger.handlers[:]:
        handler.close()
        stats_logger.removeHandler(handler)


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))