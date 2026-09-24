# coding=utf-8
import os
import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import yaml

# YAML keys that are applied under a different argparse dest. Historical spellings
# of the inner-loop step count, kept so old configs keep working.
CONFIG_KEY_ALIASES: Dict[str, str] = {
    "glances": "inner_steps",
    "update_steps": "inner_steps",
}

# YAML keys that are deliberately inert: they document the resulting behaviour
# for a reader but must NOT be wired to an argument, because something else
# decides the value. Anything here needs a comment saying what that something is.
INTENTIONALLY_UNUSED_CONFIG_KEYS: Set[str] = {
    # model.ucl_bresnet._infer_ucl_split_from_loader derives `split` from the
    # loader name (CIL -> concatenated heads, TIL -> per-task heads) and
    # overwrites whatever the config says. Registering it would let a config
    # appear to set something it cannot.
    "split",
}


def get_parser():
    parser = argparse.ArgumentParser(description="Continual learning")
    parser.add_argument(
        "--expt_name", type=str, default="test_lamaml", help="name of the experiment"
    )

    # model details
    parser.add_argument(
        "--model", type=str, default="lamaml_cifar", help="algo to train"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet1d",
        help="arch to use for training",
        choices=["resnet1d"],
    )
    parser.add_argument(
        "--n_hiddens",
        type=int,
        default=100,
        help="number of hidden neurons at each layer",
    )
    parser.add_argument(
        "--n_layers", type=int, default=2, help="number of hidden layers"
    )
    parser.add_argument(
        "--xav_init",
        default=False,
        action="store_true",
        help="Use xavier initialization",
    )

    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Debug mode with more frequent logging and smaller data splits",
    )
    parser.add_argument(
        "--use_groupnorm",
        default=False,
        action="store_true",
        help="Use GroupNorm in compatible backbones instead of BatchNorm.",
    )
    parser.add_argument(
        "--norm_type",
        type=str,
        default="batchnorm",
        choices=["batchnorm", "groupnorm", "adab1n"],
        help=(
            "Normalization layer used by compatible backbones (currently "
            "resnet1d). 'adab1n' is a task-aware adaptive BatchNorm1d "
            "(see model/adab1n.py); --use_groupnorm remains a legacy alias "
            "for norm_type=groupnorm."
        ),
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=1.0,
        help=(
            "AdaB1N running-stat momentum schedule exponent in [0, 1]: 0 is a "
            "cumulative average, 1 matches ordinary BatchNorm's fixed "
            "momentum. Ignored unless norm_type=adab1n."
        ),
    )
    parser.add_argument(
        "--adab1n_init_weight",
        type=float,
        default=0.0,
        help="Initial value of AdaB1N's per-task concentration logits.",
    )

    # optimizer parameters influencing all models
    parser.add_argument(
        "--inner_steps",
        default=1,
        type=int,
        help=(
            "Inner optimization passes per observe call: multi-pass training (ex-glances), "
            "alternating fast/meta rounds for CTN and BCL-Dual, ANML inner updates (ex-update_steps). "
            "La-MAML uses the effective total pass count (see LamamlBaseConfig: inner_steps × n_meta "
            "from merged args for backward-compatible YAML). CTN/BCL-Dual fold legacy "
            "inner_steps × n_meta from YAML into a single inner_steps count."
        ),
    )
    parser.add_argument(
        "--n_epochs", type=int, default=1, help="Number of epochs per task"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="the amount of items received by the algorithm at one time (set to 1 across all "
        + "experiments). Variable name is from GEM project.",
    )
    parser.add_argument(
        "--replay_batch_size",
        type=float,
        default=20,
        help="The batch size for experience replay.",
    )
    parser.add_argument(
        "--memories",
        type=int,
        default=5120,
        help="number of total memories stored in a reservoir sampling based buffer",
    )
    parser.add_argument(
        "--use_ring_buffer",
        default=False,
        action="store_true",
        help="Store La-MAML replay exemplars in a per-task ring buffer (FIFO) instead of the default reservoir sampler.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate (For baselines)"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        help="optimizer name for models that support switching",
    )
    parser.add_argument(
        "--prune_perc",
        type=float,
        default=0.75,
        help=(
            "PackNet: fraction of currently free (unowned) weights to drop after each task; "
            "the complement is kept and assigned to the completed task."
        ),
    )
    parser.add_argument(
        "--post_prune_epochs",
        type=int,
        default=0,
        help=(
            "PackNet: full passes over the task train loader after packing for optional finetune; "
            "gradients only on weights newly assigned to that task. 0 disables."
        ),
    )
    parser.add_argument(
        "--bn_mode",
        type=str,
        default="shared",
        choices=["task_specific", "shared"],
        help=(
            "BatchNorm statistics policy for task-incremental runs. "
            "'shared' (default) trains a single BatchNorm instance continuously "
            "across all tasks. 'task_specific' gives every task its own running "
            "mean/variance, selected by task id at train and eval time (see "
            "model/task_bn.py); the affine weight/bias stay shared across "
            "tasks. Not recommended: a task's statistics freeze at its task "
            "boundary while shared weights keep drifting, so old tasks collapse "
            "to chance for any method that does not freeze old-task weights. "
            "Ignored for class_incremental_loader runs and for norm_type "
            "groupnorm/adab1n."
        ),
    )
    parser.add_argument(
        "--eval_bn_stats",
        type=str,
        default="batch",
        choices=["batch", "running"],
        help=(
            "BatchNorm statistics read by evaluation forwards (metric loops and "
            "LwF's frozen teacher) in task-incremental runs with --bn_mode "
            "shared. 'batch' (default) normalizes each eval batch with its own "
            "statistics without writing any buffer; eval loaders are per task, "
            "so this is task-conditional, which TIL allows. 'running' reads the "
            "shared running statistics, which track the most recently trained "
            "task and so misnormalize every earlier one. Class-incremental runs "
            "always use running statistics."
        ),
    )
    parser.add_argument(
        "--no_class_weighted_ce",
        dest="class_weighted_ce",
        action="store_false",
        help=(
            "Disable inverse-frequency class weights in cross-entropy "
            "(default: weighted CE matches ucl_bresnet minibatch weighting)."
        ),
    )
    parser.set_defaults(class_weighted_ce=True)
    parser.add_argument(
        "--eralg4_masked_loss",
        action="store_true",
        help="eralg4 (ER-reservoir): apply per-sample TIL/CIL logit masking in "
        "the training loss (as er_ring and lamaml_cifar do). Now the DEFAULT; "
        "this flag is kept for script compatibility.",
    )
    parser.add_argument(
        "--eralg4_unmasked_loss",
        dest="eralg4_masked_loss",
        action="store_false",
        help="eralg4: ablation switch restoring the legacy unmasked global-softmax "
        "training loss (cross-task interference; ~-6 F1 / -3 BWT in single-epoch "
        "TIL on the evidential-cl benchmark).",
    )
    parser.set_defaults(eralg4_masked_loss=True)

    # experiment parameters
    parser.add_argument("--cuda", default=True, action="store_true", help="Use GPU")
    parser.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        help="Enable automatic mixed precision during training on CUDA.",
    )
    parser.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="Disable automatic mixed precision during training.",
    )
    parser.set_defaults(amp=True)
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="Autocast dtype when AMP is enabled.",
    )
    parser.add_argument(
        "--cudnn_benchmark",
        dest="cudnn_benchmark",
        action="store_true",
        help="Enable cuDNN benchmark mode for potentially faster convolutions.",
    )
    parser.add_argument(
        "--no-cudnn-benchmark",
        dest="cudnn_benchmark",
        action="store_false",
        help="Disable cuDNN benchmark mode.",
    )
    parser.set_defaults(cudnn_benchmark=True)
    parser.add_argument("--seed", type=int, default=0, help="random seed of model")
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,39,55",
        help=(
            "Comma-separated list of random seeds to sweep. When more than one "
            "seed is given, main.py re-invokes itself once per seed (fresh "
            "process each). Ignored when --single-seed is set."
        ),
    )
    parser.add_argument(
        "--single-seed",
        action="store_true",
        help=(
            "Run a single seed (the value of --seed), ignoring --seeds. "
            "Reproduces the legacy single-run behavior."
        ),
    )
    parser.add_argument(
        "--parallel-seeds",
        dest="parallel_seeds",
        type=int,
        default=1,
        help=(
            "Maximum number of seed subprocesses to run concurrently during a "
            "multi-seed sweep. 1 (default) runs seeds sequentially. Values >1 "
            "launch that many child processes at once; use --seed-gpu-ids to "
            "pin each worker to a distinct GPU and avoid contention."
        ),
    )
    parser.add_argument(
        "--seed-gpu-ids",
        dest="seed_gpu_ids",
        type=str,
        default="",
        help=(
            "Comma-separated GPU ids to distribute parallel seed workers across "
            "(round-robin via CUDA_VISIBLE_DEVICES), e.g. '0,1,2'. Only used when "
            "--parallel-seeds > 1. Empty leaves CUDA_VISIBLE_DEVICES untouched."
        ),
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default="",
        help=(
            "Internal: shared run timestamp passed from the multi-seed launcher "
            "to each child so all seeds group under one experiment directory. "
            "Not normally set by users."
        ),
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=100,
        help="frequency of checking the validation accuracy, in minibatches",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/",
        help="the directory where the logs will be saved",
    )
    parser.add_argument("--tf_dir", type=str, default="", help="(not set by user)")
    parser.add_argument(
        "--calc_test_accuracy",
        default=False,
        action="store_true",
        help="Calculate test accuracy along with val accuracy",
    )
    parser.add_argument(
        "--state_logging",
        default=False,
        action="store_true",
        help="Print high-level state messages to stdout for debugging",
    )

    # data parameters
    parser.add_argument(
        "--data_path",
        default="data/tiny-imagenet-200/",
        help="path where data is located",
    )
    parser.add_argument(
        "--task-order-files",
        dest="task_order_files",
        type=str,
        default="",
        help=(
            "Comma-separated list of IQ .npz file names or stems defining the task order. "
            "When provided, overrides the default alphabetical file order for IQ datasets."
        ),
    )
    parser.add_argument(
        "--loader",
        type=str,
        default="task_incremental_loader",
        help="data loader to use",
    )
    parser.add_argument(
        "--samples_per_task",
        type=int,
        default=-1,
        help="training samples per task (all if negative)",
    )
    parser.add_argument(
        "--task-order-seed",
        dest="task_order_seed",
        type=int,
        default=None,
        help=(
            "Seed for permuting task presentation order, applied after resolving "
            "--task-order-files / default alphabetical order via a private "
            "numpy.random.Generator. Omit (the default) to derive it from --seed, "
            "so sweeping seeds sweeps task order too. Set an integer to pin the "
            "order while --seed varies, which isolates training noise from "
            "task-order effects."
        ),
    )
    parser.add_argument(
        "--classes_per_it", type=int, default=4, help="number of classes in every batch"
    )
    parser.add_argument(
        "--iterations", type=int, default=5000, help="number of classes in every batch"
    )
    parser.add_argument(
        "--dataset",
        default="tinyimagenet",
        type=str,
        help="Dataset to train and test on.",
    )
    parser.add_argument(
        "--workers",
        default=3,
        type=int,
        help="Number of workers preprocessing the data.",
    )
    parser.add_argument(
        "--validation",
        default=0.0,
        type=float,
        help="Validation split (0. <= x <= 1.).",
    )
    parser.add_argument(
        "--data_scaling",
        default="none",
        type=str,
        choices=["none", "normalize", "standardize"],
        help=(
            "Apply scaling to IQ data: 'normalize' uses min/max scaling and "
            "'standardize' applies z-score based on training data."
        ),
    )
    parser.add_argument(
        "--use_iq_aug_features",
        default=False,
        action="store_true",
        help=(
            "When enabled, append exactly one derived IQ channel at model input "
            "time: I**2 + Q**2 (power) or I*Q (cross)."
        ),
    )
    parser.add_argument(
        "--iq_aug_feature_type",
        type=str,
        default="power",
        choices=["power", "cross"],
        help="When `--use_iq_aug_features` is enabled, select which derived IQ "
        "channel to append: `power` => I**2 + Q**2, `cross` => I*Q.",
    )
    parser.add_argument(
        "-order",
        "--class_order",
        default="old",
        type=str,
        help="define classes order of increment ",
        choices=["random", "chrono", "old", "super"],
    )
    parser.add_argument(
        "-inc",
        "--increment",
        default=5,
        type=int,
        help="number of classes to increment by in class incremental loader",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=100000,
        help="batch size to use during testing.",
    )
    parser.add_argument(
        "--nc_per_task",
        type=int,
        default=None,
        help="number of classes per task (uniform). Ignored if nc_per_task_list is provided.",
    )
    parser.add_argument(
        "--nc_per_task_list",
        type=str,
        default="",
        help="comma-separated class counts per task (overrides nc_per_task)",
    )
    parser.add_argument(
        "--val_rate", type=int, default=10, help="frequency (in epochs) of validation"
    )

    # La-MAML parameters
    parser.add_argument(
        "--opt_lr", type=float, default=1e-1, help="learning rate for LRs"
    )
    parser.add_argument(
        "--opt_wt", type=float, default=1e-1, help="learning rate for weights"
    )
    parser.add_argument(
        "--alpha_init", type=float, default=1e-3, help="initialization for the LRs"
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.0,
        help="Momentum used by La-MAML async per-parameter weight updates",
    )
    parser.add_argument(
        "--learn_lr",
        default=False,
        action="store_true",
        help="model should update the LRs during learning",
    )
    parser.add_argument(
        "--sync_update",
        default=False,
        action="store_true",
        help="the LRs and weights should be updated synchronously",
    )

    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=0.0,
        help="Clip gradients to this norm. 0 disables clipping (the default).",
    )
    parser.add_argument(
        "--meta_batches",
        default=3,
        type=int,
        help="Number of batches in inner trajectory",
    )
    parser.add_argument(
        "--use_old_task_memory",
        action="store_true",
        help="Use only old task samples for replay buffer data. Now the "
        "DEFAULT; this flag is kept for script compatibility.",
    )
    parser.add_argument(
        "--no_use_old_task_memory",
        dest="use_old_task_memory",
        action="store_false",
        help="Replay from the live buffer, including the current task's samples.",
    )
    parser.set_defaults(use_old_task_memory=True)
    parser.add_argument(
        "--second_order",
        default=False,
        action="store_true",
        help="use second order MAML updates",
    )

    # memory parameters for GEM | AGEM | ICARL
    parser.add_argument(
        "--n_memories",
        type=int,
        default=5120,
        help="total replay-buffer capacity across all tasks",
    )
    parser.add_argument(
        "--memory_strength",
        default=0,
        type=float,
        help="memory strength (meaning depends on memory)",
    )
    parser.add_argument(
        "--memory_loss_lambda",
        type=float,
        default=1.0,
        help="AGEM: scales replay/memory loss regularization strength.",
    )
    parser.add_argument(
        "--steps_per_sample", default=1, type=int, help="training steps per batch"
    )

    # # parameters specific to MER
    # parser.add_argument('--gamma', type=float, default=1.0,
    #                     help='gamma learning rate parameter')
    # parser.add_argument('--s', type=float, default=1,
    #                     help='current example learning rate multiplier (s)')
    # parser.add_argument('--batches_per_example', type=float, default=1,
    #                     help='the number of batch per incoming example')

    # parameters specific to Meta-BGD
    parser.add_argument(
        "--bgd_optimizer",
        type=str,
        default="bgd",
        choices=["adam", "adagrad", "bgd", "sgd"],
        help="Optimizer.",
    )
    parser.add_argument(
        "--optimizer_params",
        default="{}",
        type=str,
        nargs="*",
        help="Optimizer parameters",
    )

    parser.add_argument(
        "--train_mc_iters",
        default=5,
        type=int,
        help="Number of MonteCarlo samples during training(default 10)",
    )
    parser.add_argument(
        "--std_init", default=5e-2, type=float, help="STD init value (default 5e-2)"
    )
    parser.add_argument(
        "--mean_eta", default=1, type=float, help="Eta for mean step (default 1)"
    )
    parser.add_argument("--fisher_gamma", default=0.95, type=float, help="")

    ## ANML parameters
    parser.add_argument(
        "--rln",
        type=int,
        default=7,
        help="number of hidden neurons in the representation layer",
    )
    parser.add_argument(
        "--meta_lr", type=float, default=0.001, help="outer learning rate"
    )
    parser.add_argument(
        "--update_lr", type=float, default=0.1, help="inner learning rate"
    )

    # CTN parameters
    parser.add_argument(
        "--ctx_lr", type=float, default=0.05, help="Context learning rate for CTN"
    )
    parser.add_argument(
        "--n_meta",
        type=int,
        default=1,
        help=(
            "La-MAML: folded into inner_steps (inner_steps × n_meta) in LamamlBaseConfig. "
            "CTN/BCL-Dual: legacy only—multiplied with inner_steps when loading model config "
            "to match the old nested schedule; omit or set to 1 for a single inner_steps value."
        ),
    )
    parser.add_argument(
        "--temperature", type=float, default=5, help="Temperature for CTN"
    )
    parser.add_argument(
        "--task_emb", type=int, default=64, help="Task embedding dimension for CTN"
    )

    # Parameters for HAT

    # Model hyper-parameters set from YAML. Registered with default=None so an
    # unset flag falls through to each learner's own dataclass default: several
    # names are shared by models that give them different meanings.
    parser.add_argument(
        "--si_c",
        type=float,
        default=None,
        help="SI penalty strength c (weight on the path-integral anchor).",
    )
    parser.add_argument(
        "--si_epsilon",
        type=float,
        default=None,
        help="SI damping term in the per-task importance normaliser.",
    )
    parser.add_argument(
        "--lamb",
        type=float,
        default=None,
        help="EWC / RWalk anchor-penalty strength lambda.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="RWalk Fisher EMA momentum; UCL mu-regularisation strength.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=None,
        help="RWalk damping term in the parameter-importance score s.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Shared name, per-model meaning (None leaves each model's own default): "
        "UCL sigma-regularisation strength; BCL-Dual's Reptile-style meta-step "
        "amplification (new = before + (after-before)*beta); MER's meta-update rate.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=None,
        help="UCL initial posterior sigma as a ratio of the He init scale.",
    )
    parser.add_argument(
        "--lr_rho",
        type=float,
        default=None,
        help="UCL learning rate for the posterior rho (sigma) parameters.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Shared name, per-model meaning (None leaves each model's own default): "
        "GEM / GEM-R's margin added to the dual QP constraint (gamma in the paper); "
        "HAT's mask-sparsity penalty weight; MER's meta-update rate.",
    )
    parser.add_argument(
        "--smax",
        type=float,
        default=None,
        help="HAT maximum gate temperature s_max in the annealing schedule.",
    )
    parser.add_argument(
        "--distill_lambda",
        type=float,
        default=None,
        help="LwF's weight on the logit-distillation term.",
    )
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=None,
        help="UCL Monte-Carlo samples drawn per evaluation forward pass.",
    )

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
            for candidate in sorted(
                candidate for candidate in candidates if candidate.is_file()
            ):
                paths.append(candidate)
            continue
        if not path.exists():
            raise FileNotFoundError(f"Config source '{source}' does not exist")
        paths.append(path)
    return paths


def _apply_config_overrides(
    args: argparse.Namespace, config_paths: Iterable[Path]
) -> argparse.Namespace:
    """Apply YAML overrides from the provided config files to the namespace.

    A YAML key reaches a learner only if some ``add_argument`` in
    :func:`get_parser` declares that dest: the namespace is built by
    ``parser.parse_args([])`` and so contains exactly the registered dests and
    nothing else. Any other key is therefore not applicable, and this raises
    rather than skipping it. Silently dropping such keys is how
    ``configs/models/til/si.yaml``'s ``si_c: 0.4`` came to have no effect on any
    run for as long as it existed, while the file read as if it did.

    Args:
        args: Namespace of parser defaults to overwrite in place.
        config_paths: YAML files, applied in order; later files win.

    Returns:
        The same namespace, with every applicable key applied.

    Raises:
        ValueError: If any file contains a key that no argument declares and
            that is not listed in :data:`INTENTIONALLY_UNUSED_CONFIG_KEYS`.

    Usage:
        >>> _apply_config_overrides(args, [Path("configs/base.yaml")])
    """
    unrecognised: List[Tuple[str, str]] = []
    for path in config_paths:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        for key, value in data.items():
            if key in CONFIG_KEY_ALIASES:
                setattr(args, CONFIG_KEY_ALIASES[key], value)
                continue
            if hasattr(args, key):
                setattr(args, key, value)
                continue
            if key in INTENTIONALLY_UNUSED_CONFIG_KEYS:
                continue
            unrecognised.append((str(path), key))
    if unrecognised:
        listing = "\n".join(f"    {path}: {key}" for path, key in unrecognised)
        raise ValueError(
            "Config key(s) that no argparse argument declares, so they would "
            "have no effect on the run:\n"
            f"{listing}\n"
            "Register the argument in parser.get_parser(), remove the key, or "
            "add it to parser.INTENTIONALLY_UNUSED_CONFIG_KEYS with a comment "
            "explaining why it is inert."
        )
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
