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
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import traceback
import time
import gc
from pathlib import Path


def suggest_hyperparameters(trial):
    # ca_lr = trial.suggest_categorical("train_ca_lr", [1e-4, 1e-3, 1e-2])
    ca_lr = trial.suggest_float("train_ca_lr", 1e-4, 1e-2)

    # robust_weight_log = trial.suggest_categorical("robust_weight_log", [-4, -3, -2, -1, 0, 1, 2, 3, 4])
    robust_weight_log = trial.suggest_float("robust_weight_log", -2, 1)
    robust_weight = 10**robust_weight_log

    # entropy_weight_log = trial.suggest_categorical("entropy_weight_log", [-2, -1, 0, 1, 2])
    entropy_weight_log = trial.suggest_float("entropy_weight_log", -2, 1)
    entropy_weight = 10**entropy_weight_log

    ca_logit_norm = trial.suggest_float("ca_logit_norm", 0.1, 0.5)

    ca_lr = round(ca_lr, 5)
    robust_weight = round(robust_weight, 5)
    entropy_weight = round(entropy_weight, 5)
    ca_logit_norm = round(ca_logit_norm, 2)

    return {
        "ca_lr": ca_lr,
        "ca_robust_weight": robust_weight,
        "ca_entropy_weight": entropy_weight,
        "ca_logit_norm": ca_logit_norm
    }


def objective(trial, base_args, study_name, pruning_thresholds=None, data_manager=None):
    trial_start_time = time.time()

    try:
        hyperparams = suggest_hyperparameters(trial)

        args = copy.deepcopy(base_args)
        args.update(hyperparams)

        args["trial_number"] = trial.number
        args["study_name"] = study_name

        logging.info(
            f"\n[Trial {trial.number}] Start optimization with hyperparameters: {hyperparams}"
        )

        result = _train_optuna(args, trial, pruning_thresholds, data_manager)

        trial_end_time = time.time()
        trial_duration = trial_end_time - trial_start_time

        final_accuracy = result["final_avg_accuracy"]

        logging.info(
            f"[Trial {trial.number}] Final Average Accuracy: {final_accuracy:.2f}, Duration: {trial_duration:.2f}s"
        )

        gc.collect()
        torch.cuda.empty_cache()

        return final_accuracy

    except optuna.TrialPruned:
        trial_end_time = time.time()
        trial_duration = trial_end_time - trial_start_time

        gc.collect()
        torch.cuda.empty_cache()

        raise

    except Exception as e:
        trial_end_time = time.time()
        trial_duration = trial_end_time - trial_start_time

        logging.error(f"[Trial {trial.number}] Error during optimization: {str(e)}")
        logging.error(
            f"[Trial {trial.number}] Duration before error: {trial_duration:.2f}s"
        )
        logging.error(f"Error details: {traceback.format_exc()}")

        gc.collect()
        torch.cuda.empty_cache()

        return 0.0


def _train_optuna(args, trial, pruning_thresholds=None, data_manager=None):
    if data_manager is None:
        _set_random(args["seed"])
        _set_device(args)

        data_manager = DataManager(
            args["dataset"],
            args["shuffle"],
            args["seed"],
            args["init_cls"],
            args["increment"],
            args,
        )

        args["nb_classes"] = data_manager.nb_classes
        args["nb_tasks"] = data_manager.nb_tasks

    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []
    acc_history = []

    for task in range(data_manager.nb_tasks):
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if "-" in key]
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            nme_keys = [key for key in nme_accy["grouped"].keys() if "-" in key]
            nme_values = [nme_accy["grouped"][key] for key in nme_keys]
            nme_matrix.append(nme_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])

            current_avg_accuracy = sum(cnn_curve["top1"]) / len(cnn_curve["top1"])
            logging.info(
                f"[Task {task}] Current Average Accuracy (CNN): {current_avg_accuracy:.2f}"
            )
            acc_history.append(current_avg_accuracy)

        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if "-" in key]
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            current_avg_accuracy = sum(cnn_curve["top1"]) / len(cnn_curve["top1"])
            logging.info(
                f"[Task {task}] Current Average Accuracy (CNN): {current_avg_accuracy:.2f}"
            )
            acc_history.append(current_avg_accuracy)

        if pruning_thresholds and task in pruning_thresholds:
            threshold = pruning_thresholds[task]
            if current_avg_accuracy < threshold:
                logging.info(
                    f"[Pruning] Accuracy {current_avg_accuracy:.2f} < {threshold:.2f} at task {task}"
                )
                raise optuna.TrialPruned()

    final_avg_accuracy = sum(cnn_curve["top1"]) / len(cnn_curve["top1"])

    acc_history = [float(np.round(v, 2)) for v in acc_history]
    logging.info(f"Final accuracy history: {acc_history}")

    # Calculate forgetting if needed
    forgetting_cnn = None
    forgetting_nme = None

    if len(cnn_matrix) > 0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(cnn_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        np_acctable = np_acctable.T
        forgetting_cnn = np.mean(
            (np.max(np_acctable, axis=1) - np_acctable[:, task])[:task]
        )
        logging.info(f"Forgetting (CNN): {forgetting_cnn:.4f}")

    if len(nme_matrix) > 0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(nme_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        np_acctable = np_acctable.T
        forgetting_nme = np.mean(
            (np.max(np_acctable, axis=1) - np_acctable[:, task])[:task]
        )
        logging.info(f"Forgetting (NME): {forgetting_nme:.4f}")

    result = {
        "final_avg_accuracy": final_avg_accuracy,
        "cnn_curve": cnn_curve,
        "nme_curve": nme_curve,
        "forgetting_cnn": forgetting_cnn,
        "forgetting_nme": forgetting_nme,
        "cnn_matrix": cnn_matrix,
        "nme_matrix": nme_matrix,
    }

    return result


def run_optuna_optimization(
    base_args,
    study_name,
    pruning_thresholds={},
    n_trials=100,
    early_stop_patience=None,
    max_time_hours=None,
    max_failed_trials=None,  # New parameter for early stopping on total failed trials
    data_manager=None,
):
    log_dir = f"logs/optuna/{base_args['dataset']}/{base_args['model_name']}"
    os.makedirs(log_dir, exist_ok=True)

    logfilename = f"{log_dir}/optuna_{study_name}"
    
    # Clear any existing logging handlers to avoid conflicts
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True  # Force reconfiguration
    )

    logging.info(f"Starting Optuna optimization: {study_name}")
    logging.info(f"Number of trials: {n_trials}")
    if early_stop_patience is not None:
        logging.info(f"Early stopping patience (no improvement): {early_stop_patience}")
    if max_failed_trials is not None:
        logging.info(f"Early stopping on total failed trials: {max_failed_trials}")
    if max_time_hours is not None:
        logging.info(f"Maximum optimization time: {max_time_hours} hours")

    sampler = TPESampler(seed=base_args["seed"])
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)

    OPTUNA_DIR = "optuna"
    os.makedirs(OPTUNA_DIR, exist_ok=True)
    db_path = Path(OPTUNA_DIR) / f"{study_name}.db"
    storage_name = f"sqlite:///{db_path}"

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    logging.info(f"Storage: {storage_name}")

    try:
        best_value = study.best_value if study.best_value is not None else -float("inf")
        if best_value != -float("inf"):
            logging.info(f"Resuming study with existing best value: {best_value:.2f}")
    except ValueError:
        best_value = -float("inf")
        logging.info("Starting fresh study (no previous trials found)")

    min_delta = 0.01
    no_improvement_trials = 0
    
    total_failed_trials = 0
    if len(study.trials) > 0:
        total_failed_trials = sum(1 for trial in study.trials 
                                if trial.state in [optuna.trial.TrialState.FAIL, optuna.trial.TrialState.PRUNED])
        logging.info(f"Resuming study: {len(study.trials)} existing trials, {total_failed_trials} failed/pruned")

    def early_stopping_callback(study, trial):
        nonlocal best_value, no_improvement_trials, min_delta, total_failed_trials
        
        # Check for total failed trials early stopping
        if max_failed_trials is not None:
            if trial is not None and trial.state in [optuna.trial.TrialState.FAIL, optuna.trial.TrialState.PRUNED]:
                total_failed_trials += 1
                logging.info(
                    f"Trial {trial.number} failed/pruned. Total failed trials: {total_failed_trials}/{max_failed_trials}"
                )
                if total_failed_trials >= max_failed_trials:
                    logging.info(
                        f"Early stopping: {total_failed_trials} total failed trials reached."
                    )
                    study.stop()
                    return
        
        # Original early stopping logic for no improvement
        if trial is not None:
            if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None and trial.value - min_delta > best_value:
                best_value = trial.value
                no_improvement_trials = 0
                logging.info(f"New best value: {best_value:.2f} at trial {trial.number}")
            else:
                no_improvement_trials += 1
                logging.info(
                    f"No improvement in trial {trial.number}. Count: {no_improvement_trials}/{early_stop_patience}"
                )
                if early_stop_patience is not None and no_improvement_trials >= early_stop_patience:
                    logging.info(
                        f"Early stopping: No improvement in the last {early_stop_patience} trials."
                    )
                    study.stop()

    try:
        callbacks = [early_stopping_callback]
        
        study.optimize(
            lambda trial: objective(
                trial, base_args, study_name, pruning_thresholds, data_manager
            ),
            n_trials=n_trials,
            callbacks=callbacks,
            timeout=max_time_hours * 3600 if max_time_hours is not None else None
        )

        return study

    except KeyboardInterrupt:
        logging.info("Optimization interrupted by user")
        return study


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


def train(args, pruning_thresholds=None):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device

        dataset_name = args.get("dataset", "unknown_dataset")
        model_name = args.get("model_name", "unknown_model")
        study_name = f"seed{seed}_{model_name}_{dataset_name}"

        _set_random(args["seed"])
        _set_device(args)

        data_manager = DataManager(
            args["dataset"],
            args["shuffle"],
            args["seed"],
            args["init_cls"],
            args["increment"],
            args,
        )

        args["nb_classes"] = data_manager.nb_classes
        args["nb_tasks"] = data_manager.nb_tasks

        n_trials = args.get("n_trials", 100)
        early_stop_patience = args.get("early_stop_patience", None)
        max_time_hours = args.get("max_time_hours", None)
        max_failed_trials = args.get("max_failed_trials", None)

        run_optuna_optimization(
            base_args=args,
            study_name=study_name,
            pruning_thresholds=pruning_thresholds,
            n_trials=n_trials,
            early_stop_patience=early_stop_patience,
            max_time_hours=max_time_hours,
            max_failed_trials=max_failed_trials,
            data_manager=data_manager,
        )