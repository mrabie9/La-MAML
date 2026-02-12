# coding=utf-8
import os
import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import yaml

def get_parser():
    parser = argparse.ArgumentParser(description='Continual learning')
    parser.add_argument('--expt_name', type=str, default='test_lamaml',
                    help='name of the experiment')
    
    # model details
    parser.add_argument('--model', type=str, default='lamaml_cifar',
                        help='algo to train')
    parser.add_argument(
        '--arch',
        type=str,
        default='resnet1d',
        help='arch to use for training',
        choices=['resnet1d'],
    )
    parser.add_argument('--n_hiddens', type=int, default=100,
                        help='number of hidden neurons at each layer')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='number of hidden layers')
    parser.add_argument('--xav_init', default=False , action='store_true',
                        help='Use xavier initialization')
    
    parser.add_argument('--debug', default=False , action='store_true',
                        help='Debug mode with more frequent logging and smaller data splits')



    # optimizer parameters influencing all models
    parser.add_argument("--glances", default=1, type=int,
                        help="Number of times the model is allowed to train over a set of samples in the single pass setting") 
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Number of epochs per task')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='the amount of items received by the algorithm at one time (set to 1 across all ' +
                        'experiments). Variable name is from GEM project.')
    parser.add_argument('--replay_batch_size', type=float, default=20,
                        help='The batch size for experience replay.')
    parser.add_argument('--memories', type=int, default=5120, 
                        help='number of total memories stored in a reservoir sampling based buffer')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (For baselines)')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='optimizer name for models that support switching')

    
    # experiment parameters
    parser.add_argument('--cuda', default=True , action='store_true',
                        help='Use GPU')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed of model')
    parser.add_argument('--log_every', type=int, default=100,
                        help='frequency of checking the validation accuracy, in minibatches')
    parser.add_argument('--log_dir', type=str, default='logs/',
                        help='the directory where the logs will be saved')
    parser.add_argument('--tf_dir', type=str, default='',
                        help='(not set by user)')
    parser.add_argument('--calc_test_accuracy', default=False , action='store_true',
                        help='Calculate test accuracy along with val accuracy')
    parser.add_argument('--state_logging', default=False, action='store_true',
                        help='Print high-level state messages to stdout for debugging')

    # data parameters
    parser.add_argument('--data_path', default='data/tiny-imagenet-200/',
                        help='path where data is located')
    parser.add_argument('--loader', type=str, default='task_incremental_loader',
                        help='data loader to use')
    parser.add_argument('--samples_per_task', type=int, default=-1,
                        help='training samples per task (all if negative)')
    parser.add_argument('--shuffle_tasks', default=False, action='store_true',
                        help='present tasks in order')
    parser.add_argument('--classes_per_it', type=int, default=4,
                        help='number of classes in every batch')
    parser.add_argument('--iterations', type=int, default=5000,
                        help='number of classes in every batch')
    parser.add_argument("--dataset", default="tinyimagenet", type=str,
                    help="Dataset to train and test on.")
    parser.add_argument("--workers", default=3, type=int,
                        help="Number of workers preprocessing the data.")
    parser.add_argument("--validation", default=0., type=float,
                        help="Validation split (0. <= x <= 1.).")
    parser.add_argument("-order", "--class_order", default="old", type=str,
                        help="define classes order of increment ",
                        choices = ["random", "chrono", "old", "super"])
    parser.add_argument("-inc", "--increment", default=5, type=int,
                        help="number of classes to increment by in class incremental loader")
    parser.add_argument('--test_batch_size', type=int, default=100000 ,
                        help='batch size to use during testing.')
    parser.add_argument('--nc_per_task', type=int, default=None,
                        help='number of classes per task (uniform). Ignored if nc_per_task_list is provided.')
    parser.add_argument('--nc_per_task_list', type=str, default='',
                        help='comma-separated class counts per task (overrides nc_per_task)')
    parser.add_argument('--val_rate', type=int, default=10,
                        help='frequency (in epochs) of validation')


    # La-MAML parameters
    parser.add_argument('--opt_lr', type=float, default=1e-1,
                        help='learning rate for LRs')
    parser.add_argument('--opt_wt', type=float, default=1e-1,
                        help='learning rate for weights')
    parser.add_argument('--alpha_init', type=float, default=1e-3,
                        help='initialization for the LRs')
    parser.add_argument('--learn_lr', default=False, action='store_true',
                        help='model should update the LRs during learning')
    parser.add_argument('--sync_update', default=False , action='store_true',
                        help='the LRs and weights should be updated synchronously')

    parser.add_argument('--grad_clip_norm', type=float, default=2.0,
                        help='Clip the gradients by this value')
    parser.add_argument("--meta_batches", default=3, type=int,
                        help="Number of batches in inner trajectory") 
    parser.add_argument('--use_old_task_memory', default=False, action='store_true', 
                        help='Use only old task samples for replay buffer data')    
    parser.add_argument('--second_order', default=False , action='store_true',
                        help='use second order MAML updates')


   # memory parameters for GEM | AGEM | ICARL 
    parser.add_argument('--n_memories', type=int, default=5120,
                        help='number of memories per task')
    parser.add_argument('--memory_strength', default=0, type=float,
                        help='memory strength (meaning depends on memory)')
    parser.add_argument('--steps_per_sample', default=1, type=int,
                        help='training steps per batch')


    # # parameters specific to MER 
    # parser.add_argument('--gamma', type=float, default=1.0,
    #                     help='gamma learning rate parameter')
    # parser.add_argument('--beta', type=float, default=1.0,
    #                     help='beta learning rate parameter')
    # parser.add_argument('--s', type=float, default=1,
    #                     help='current example learning rate multiplier (s)')
    # parser.add_argument('--batches_per_example', type=float, default=1,
    #                     help='the number of batch per incoming example')


    # parameters specific to Meta-BGD
    parser.add_argument('--bgd_optimizer', type=str, default="bgd", choices=["adam", "adagrad", "bgd", "sgd"],
                    help='Optimizer.')
    parser.add_argument('--optimizer_params', default="{}", type=str, nargs='*',
                        help='Optimizer parameters')

    parser.add_argument('--train_mc_iters', default=5, type=int,
                        help='Number of MonteCarlo samples during training(default 10)')
    parser.add_argument('--std_init', default=5e-2, type=float,
                        help='STD init value (default 5e-2)')
    parser.add_argument('--mean_eta', default=1, type=float,
                        help='Eta for mean step (default 1)')
    parser.add_argument('--fisher_gamma', default=0.95, type=float,
                        help='')
    
    ## ANML parameters
    parser.add_argument('--rln', type=int, default=7,
                        help='number of hidden neurons in the representation layer')
    parser.add_argument('--update_steps', type=int, default=10,
                        help='number of inner updates during training')
    parser.add_argument('--meta_lr', type=float, default=0.001,
                        help='outer learning rate')
    parser.add_argument('--update_lr', type=float, default=0.1,
                        help='inner learning rate')
    
    # CTN parameters
    parser.add_argument('--beta', type=float, default=0.05,
                        help='Beta parameter for CTN')
    parser.add_argument('--n_meta', type=int, default=2,
                        help='Number of meta-updates for CTN')
    parser.add_argument('--inner_steps', type=int, default=2,
                        help='Number of inner updates for CTN')
    parser.add_argument('--temperature', type=float, default=5,
                        help='Temperature for CTN')
    parser.add_argument('--task_emb', type=int, default=64,
                        help='Task embedding dimension for CTN')
    
    # Parameters for HAT
    


    return parser

def _expanded_config_paths(config_sources: Sequence[str] | None) -> List[Path]:
    """Resolve config file and directory inputs into a concrete ordered list."""

    if not config_sources:
        return []

    paths: List[Path] = []
    for source in config_sources:
        if not source:
            continue
        path = Path(source).expanduser()
        if path.is_dir():
            candidates = list(path.glob("*.yaml")) + list(path.glob("*.yml"))
            for candidate in sorted(candidate for candidate in candidates if candidate.is_file()):
                paths.append(candidate)
            continue
        if not path.exists():
            raise FileNotFoundError(f"Config source '{source}' does not exist")
        paths.append(path)
    return paths


def _apply_config_overrides(args: argparse.Namespace, config_paths: Iterable[Path]) -> argparse.Namespace:
    """Apply YAML overrides from the provided config files to the namespace."""

    for path in config_paths:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        for key, value in data.items():
            if hasattr(args, key):
                setattr(args, key, value)
    return args


def parse_args_from_yaml(config_sources: Sequence[str] | str | None):
    """Load arguments from one or more YAML configuration files."""

    parser = get_parser()
    args = parser.parse_args([])
    if isinstance(config_sources, str) or isinstance(config_sources, os.PathLike):
        config_list: Sequence[str] = [str(config_sources)]
    else:
        config_list = config_sources or []
    config_paths = _expanded_config_paths(config_list)
    return _apply_config_overrides(args, config_paths)
