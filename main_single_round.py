import argparse
import time
import importlib
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

import parser as file_parser
from main import (
    _split_labels,
    _noise_label_max_for_task,
    _get_det_logits,
    _false_alarm_rate,
    eval_tasks,
    save_results,
    log_state,
    _split_eval_output,
)
from metrics.metrics import confusion_matrix
from utils import misc_utils
from utils.training_metrics import (
    macro_f1_including_noise,
    macro_precision_signal_only,
    macro_recall,
)


def _default_main_config_chain() -> List[str]:
    """Return the default YAML config chain for single-round runs.

    This mirrors the behaviour in ``main.py`` so that, if no ``--config`` is
    given, the base configuration is applied automatically.

    Returns:
        List of YAML config file paths to apply in order.

    Usage:
        chain = _default_main_config_chain()
    """
    chain: List[str] = []
    base_cfg = Path("configs/base.yaml")
    if base_cfg.exists():
        chain.append(str(base_cfg))
    legacy = Path("config_all.yaml")
    if legacy.exists():
        chain.append(str(legacy))
    return chain


def _select_task_indices_from_order(task_names: List[str], order_arg: str) -> List[int]:
    """Select task indices based on the ``task_order_files`` argument.

    The task incremental IQ loader populates ``task_names`` from the IQ
    ``.npz`` filenames (stems). This helper resolves the names listed in
    ``task_order_files`` to task indices in that list.

    Args:
        task_names: List of task name stems provided by the loader.
        order_arg: Raw ``task_order_files`` string (comma-separated).

    Returns:
        List of task indices to include in the single-round experiment. If
        ``order_arg`` is empty, all tasks are returned.

    Usage:
        indices = _select_task_indices_from_order(loader.task_names, args.task_order_files)
    """
    if not task_names:
        return []
    if not order_arg:
        return list(range(len(task_names)))

    tokens = [token.strip() for token in order_arg.split(",") if token.strip()]
    if not tokens:
        return list(range(len(task_names)))

    stem_to_index = {stem: idx for idx, stem in enumerate(task_names)}
    selected_indices: List[int] = []
    for token in tokens:
        stem = os.path.splitext(token)[0]
        if stem not in stem_to_index:
            available = ", ".join(sorted(stem_to_index.keys()))
            raise SystemExit(
                f"--task-order-files references unknown task '{token}'. "
                f"Available tasks: {available}"
            )
        idx = stem_to_index[stem]
        if idx not in selected_indices:
            selected_indices.append(idx)
    return selected_indices


def build_single_round_loaders(
    args,
    loader,
) -> Tuple[DataLoader, DataLoader, List[int]]:
    """Build train and test loaders for a single-round (non-LL) experiment.

    Tasks are selected using ``args.task_order_files`` and the loader's
    ``task_names`` attribute. If multiple tasks are selected, their datasets
    are combined into a single effective task. When this happens, the function
    prints ``\"combining....\"`` to make the behaviour explicit.

    Args:
        args: Parsed experiment arguments / configuration.
        loader: An instance of the incremental loader created from
            ``dataloaders.<loader>.IncrementalLoader``.

    Returns:
        Tuple containing:

        - train_loader: DataLoader for the combined training data.
        - test_loader: DataLoader for the combined test/validation data.
        - selected_indices: List of integer task indices that were used.

    Usage:
        train_loader, test_loader, indices = build_single_round_loaders(args, loader)
    """
    task_names = getattr(loader, "task_names", [])
    selected_indices = _select_task_indices_from_order(
        task_names, getattr(args, "task_order_files", "")
    )
    if not selected_indices:
        raise SystemExit("No tasks selected for single-round experiment.")

    # Materialize all tasks once.
    train_loaders: List[DataLoader] = []
    test_loaders: List[DataLoader] = []
    all_task_infos: List[dict] = []

    loader._current_task = 0  # Reset to first task.
    for _ in range(loader.n_tasks):
        task_info, train_loader, _, test_loader = loader.new_task()
        all_task_infos.append(task_info)
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    selected_train_datasets: List[torch.utils.data.Dataset] = []
    selected_test_datasets: List[torch.utils.data.Dataset] = []

    for task_idx in selected_indices:
        task_train_loader = train_loaders[task_idx]
        task_test_loader = test_loaders[task_idx]

        selected_train_datasets.append(task_train_loader.dataset)
        selected_test_datasets.append(task_test_loader.dataset)

    if len(selected_indices) > 1:
        print("combining....")

    combined_train_dataset = (
        ConcatDataset(selected_train_datasets)
        if len(selected_train_datasets) > 1
        else selected_train_datasets[0]
    )
    combined_test_dataset = (
        ConcatDataset(selected_test_datasets)
        if len(selected_test_datasets) > 1
        else selected_test_datasets[0]
    )

    train_loader = DataLoader(
        combined_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    test_loader = DataLoader(
        combined_test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    return train_loader, test_loader, selected_indices


def run_single_round_training(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    args,
) -> Tuple[torch.Tensor, torch.Tensor, float, Dict[str, np.ndarray]]:
    """Run a non-lifelong single-round training loop for ``n_epochs``.

    This loop is intentionally similar to the per-task inner loop in
    ``life_experience`` but operates on a single (possibly combined) task.

    Args:
        model: The model to train.
        train_loader: Combined training DataLoader.
        test_loader: Combined test/validation DataLoader.
        args: Parsed experiment arguments / configuration.

    Returns:
        Tuple of:

        - result_val_t: Tensor of task indices (single value here).
        - result_val_a: Tensor of per-eval validation recalls.
        - time_spent: Total wall-clock time spent in seconds.

    Usage:
        result_val_t, result_val_a, time_spent = run_single_round_training(model, train_loader, test_loader, args)
    """
    device = torch.device(
        "cuda" if getattr(args, "cuda", False) and torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    from tqdm import tqdm  # Imported lazily to keep top-level imports minimal.

    result_val_a: List[List[float]] = []
    result_val_t: List[int] = []
    per_epoch_losses: List[float] = []
    per_epoch_train_recalls: List[float] = []
    per_epoch_val_cls_rec: List[float] = []
    per_epoch_train_det_rec: List[float] = []
    per_epoch_train_det_pfa: List[float] = []
    per_epoch_val_det_rec: List[float] = []
    per_epoch_val_det_pfa: List[float] = []
    per_epoch_train_f1: List[float] = []
    per_epoch_val_f1: List[float] = []

    time_start = time.time()

    current_task_index = 0
    noise_label_for_task = _noise_label_max_for_task(train_loader)

    for epoch in range(args.n_epochs):
        model.real_epoch = epoch
        epoch_losses: List[float] = []
        epoch_recalls: List[float] = []
        epoch_precisions: List[float] = []
        epoch_f1s: List[float] = []
        epoch_det_recalls: List[float] = []
        epoch_det_fas: List[float] = []

        progress_bar = tqdm(train_loader)
        for batch in progress_bar:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                xb, yb = batch
            elif isinstance(batch, (list, tuple)) and len(batch) == 3:
                xb, yb, _ = batch
            else:
                raise ValueError("Unexpected batch structure in single-round training.")

            xb = xb.to(device)
            y_cls = _split_labels(yb)
            if not torch.is_tensor(y_cls):
                y_cls = torch.as_tensor(y_cls)

            if getattr(args, "use_detector_arch", False):
                if isinstance(yb, (tuple, list)) and len(yb) == 2:
                    cls_part, det_part = yb[0], yb[1]
                    if not torch.is_tensor(cls_part):
                        cls_part = torch.as_tensor(cls_part)
                    if not torch.is_tensor(det_part):
                        det_part = torch.as_tensor(det_part)
                    y_for_observe = (cls_part.to(device), det_part.to(device))
                elif torch.is_tensor(yb) and yb.dim() == 2 and yb.size(1) == 2:
                    y_for_observe = (yb[:, 0].to(device), yb[:, 1].to(device))
                elif isinstance(yb, np.ndarray) and yb.ndim == 2 and yb.shape[1] == 2:
                    y_for_observe = (
                        torch.as_tensor(yb[:, 0]).to(device),
                        torch.as_tensor(yb[:, 1]).to(device),
                    )
                else:
                    y_for_observe = y_cls.to(device)
            else:
                y_for_observe = y_cls.to(device)

            model.train()
            loss, cls_tr_rec = model.observe(xb, y_for_observe, current_task_index)

            epoch_losses.append(float(loss))

            model.eval()
            with torch.no_grad():
                logits = (
                    model(xb, current_task_index)
                    if args.model != "anml"
                    else model(xb, fast_weights=None)
                )
                predictions = torch.argmax(logits, dim=1).cpu()
                det_logits = _get_det_logits(model, xb, current_task_index)
            model.train()

            y_cls_for_metric = y_cls.cpu()
            noise_label = noise_label_for_task
            if getattr(model, "split", False):
                offset1, _ = model.compute_offsets(current_task_index)
                y_cls_for_metric = y_cls_for_metric - offset1
                if noise_label is not None:
                    noise_label = noise_label - offset1

            precision = macro_precision_signal_only(
                predictions, y_cls_for_metric, noise_label
            )
            f1 = macro_f1_including_noise(predictions, y_cls_for_metric)
            if noise_label is not None:
                cls_mask = y_cls_for_metric != noise_label
                if cls_mask.any():
                    cls_tr_rec = macro_recall(
                        predictions[cls_mask], y_cls_for_metric[cls_mask]
                    )
                else:
                    cls_tr_rec = 0.0
            else:
                cls_tr_rec = macro_recall(predictions, y_cls_for_metric)

            det_rec = 0.0
            det_fa = 0.0
            if noise_label is not None:
                det_targets = (y_cls_for_metric != noise_label).long()
                if det_logits is not None:
                    det_pred = (det_logits >= 0).long().cpu()
                else:
                    det_pred = (predictions != noise_label).long()
                det_rec = macro_recall(det_pred, det_targets)
                det_fa = _false_alarm_rate(det_pred, det_targets)

            epoch_recalls.append(float(cls_tr_rec))
            epoch_precisions.append(float(precision))
            epoch_f1s.append(float(f1))
            epoch_det_recalls.append(float(det_rec))
            epoch_det_fas.append(float(det_fa))

            progress_bar.set_description(
                "Ep: {}/{} | Loss: {:.3f} | Rec: {:.3f} | Prec: {:.3f} | F1: {:.3f}".format(
                    epoch + 1,
                    args.n_epochs,
                    float(loss),
                    float(cls_tr_rec),
                    float(precision),
                    float(f1),
                )
            )

        # Validation at end of epoch on the combined test loader.
        val_loaders = [test_loader]
        val_outputs = eval_tasks(model, val_loaders, args)
        val_acc, val_prec, val_f1, val_det_acc, val_det_fa = _split_eval_output(
            val_outputs
        )
        if isinstance(val_acc, (list, tuple)):
            val_acc_values = [float(v) for v in val_acc]
            cur_val_acc = float(val_acc[0])
        else:
            val_acc_values = [float(val_acc)]
            cur_val_acc = float(val_acc)
        cur_val_det_rec = None
        cur_val_det_fa = None
        cur_val_f1 = None
        if val_det_acc is not None:
            if isinstance(val_det_acc, (list, tuple)):
                cur_val_det_rec = float(val_det_acc[0])
            else:
                cur_val_det_rec = float(val_det_acc)
        if val_det_fa is not None:
            if isinstance(val_det_fa, (list, tuple)):
                cur_val_det_fa = float(val_det_fa[0])
            else:
                cur_val_det_fa = float(val_det_fa)
        if val_f1 is not None:
            if isinstance(val_f1, (list, tuple)):
                cur_val_f1 = float(val_f1[0])
            else:
                cur_val_f1 = float(val_f1)

        result_val_a.append(val_acc_values)
        result_val_t.append(current_task_index)

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        avg_rec = float(np.mean(epoch_recalls)) if epoch_recalls else float("nan")
        avg_prec = (
            float(np.mean(epoch_precisions)) if epoch_precisions else float("nan")
        )
        avg_f1 = float(np.mean(epoch_f1s)) if epoch_f1s else float("nan")
        avg_det_rec = (
            float(np.mean(epoch_det_recalls)) if epoch_det_recalls else float("nan")
        )
        avg_det_fa = float(np.mean(epoch_det_fas)) if epoch_det_fas else float("nan")

        per_epoch_losses.append(avg_loss)
        per_epoch_train_recalls.append(avg_rec)
        per_epoch_train_det_rec.append(avg_det_rec)
        per_epoch_train_det_pfa.append(avg_det_fa)
        per_epoch_val_cls_rec.append(
            cur_val_acc if cur_val_acc is not None else float("nan")
        )
        per_epoch_val_det_rec.append(
            cur_val_det_rec if cur_val_det_rec is not None else float("nan")
        )
        per_epoch_val_det_pfa.append(
            cur_val_det_fa if cur_val_det_fa is not None else float("nan")
        )
        per_epoch_train_f1.append(avg_f1)
        per_epoch_val_f1.append(cur_val_f1 if cur_val_f1 is not None else float("nan"))

        print(
            "Epoch {}/{} | Avg Loss {:.4f} | Avg Rec {:.4f} | Avg Prec {:.4f} | Avg F1 {:.4f} | Val Rec {}".format(
                epoch + 1,
                args.n_epochs,
                avg_loss,
                avg_rec,
                avg_prec,
                avg_f1,
                val_acc_values,
            )
        )

    result_val_t_tensor = torch.as_tensor(result_val_t, dtype=torch.long)
    max_len = max(len(row) for row in result_val_a)
    padded_val = torch.full((len(result_val_a), max_len), 0.0, dtype=torch.float)
    for row_idx, row in enumerate(result_val_a):
        padded_val[row_idx, : len(row)] = torch.as_tensor(row, dtype=torch.float)

    time_spent = time.time() - time_start

    metrics_payload: Dict[str, np.ndarray] = {
        "losses": np.asarray(per_epoch_losses, dtype=float),
        "cls_tr_rec": np.asarray(per_epoch_train_recalls, dtype=float),
        "val_acc": np.asarray(per_epoch_val_cls_rec, dtype=float),
        "train_det_rec": np.asarray(per_epoch_train_det_rec, dtype=float),
        "train_det_pfa": np.asarray(per_epoch_train_det_pfa, dtype=float),
        "val_det_rec": np.asarray(per_epoch_val_det_rec, dtype=float),
        "val_det_pfa": np.asarray(per_epoch_val_det_pfa, dtype=float),
        "train_f1": np.asarray(per_epoch_train_f1, dtype=float),
        "val_f1_per_epoch": np.asarray(per_epoch_val_f1, dtype=float),
        # For compatibility with scripts that expect a final-task vector.
        "val_f1": np.asarray(
            [per_epoch_val_f1[-1]] if per_epoch_val_f1 else [], dtype=float
        ),
        "val_det_acc": np.asarray(
            [per_epoch_val_det_rec[-1]] if per_epoch_val_det_rec else [], dtype=float
        ),
        "val_det_fa": np.asarray(
            [per_epoch_val_det_pfa[-1]] if per_epoch_val_det_pfa else [], dtype=float
        ),
    }

    return result_val_t_tensor, padded_val, time_spent, metrics_payload


def main() -> None:
    """Entry point for non-lifelong (single-round) experiments.

    This script mirrors the high-level structure of ``main.py`` but trains on
    a single (possibly combined) task for ``n_epochs`` instead of running a
    full continual-learning schedule.

    Usage:
        python main_single_round.py \\
            --config configs/base.yaml \\
            --config configs/models/rwalk.yaml \\
            --config configs/non_ll_single_round.yaml
    """
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="FILE",
        help="YAML config fragment to apply (may be provided multiple times).",
    )
    config_parser.add_argument(
        "--config-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="Directory of YAML fragments to apply in alphabetical order.",
    )
    config_parser.add_argument(
        "--no-config",
        action="store_true",
        help="Skip loading YAML configs and rely solely on CLI arguments.",
    )
    config_cli, remaining = config_parser.parse_known_args()

    config_chain: List[str] = []
    if not config_cli.no_config:
        config_chain.extend(config_cli.config_dir)
        config_chain.extend(config_cli.config)
        if not config_chain:
            config_chain = _default_main_config_chain()

    base_args = file_parser.parse_args_from_yaml(config_chain or None)
    parser = file_parser.get_parser()
    args = parser.parse_args(remaining, namespace=base_args)

    args.lr = misc_utils.scale_learning_rate_for_batch_size(args.lr, args.batch_size)
    print("Running model (single-round): ", args.model)
    log_state(
        args.state_logging,
        "Single-round experiment '{}' starting with model '{}' (seed {})".format(
            args.expt_name, args.model, args.seed
        ),
    )

    misc_utils.init_seed(args.seed)

    Loader = importlib.import_module("dataloaders." + args.loader)
    loader = Loader.IncrementalLoader(args, seed=args.seed)
    n_inputs, n_outputs, n_tasks = loader.get_dataset_info()
    args.get_samples_per_task = getattr(loader, "get_samples_per_task", None)
    args.classes_per_task = getattr(loader, "classes_per_task", None)
    print("Classes per task:", args.classes_per_task)

    timestamp = misc_utils.get_date_time()
    config_name = Path(config_chain[-1]).stem if config_chain else None
    args.log_dir, args.tf_dir = misc_utils.log_dir(args, timestamp, config_name)
    log_state(args.state_logging, "Logging to {}".format(args.log_dir))

    Model = importlib.import_module("model." + args.model)
    model = Model.Net(n_inputs, n_outputs, n_tasks, args)
    if args.cuda:
        try:
            model.cuda()
        except RuntimeError:
            pass
    print(args.cuda)
    print("Model device:", next(model.parameters()).device)
    log_state(
        args.state_logging,
        "Model initialized on device {}".format(next(model.parameters()).device),
    )

    train_loader, test_loader, selected_indices = build_single_round_loaders(
        args, loader
    )
    print("Single-round using task indices:", selected_indices)

    result_val_t, result_val_a, time_spent, metrics_payload = run_single_round_training(
        model, train_loader, test_loader, args
    )

    # Save per-epoch metrics under the same /metrics layout used in main.py.
    logs_dir = os.path.join(args.log_dir, "metrics")
    os.makedirs(logs_dir, exist_ok=True)
    np.savez(os.path.join(logs_dir, "task0.npz"), **metrics_payload)

    # Record a human-readable task order entry for this combined run.
    task_order_path = os.path.join(logs_dir, "task_order.txt")
    task_names = getattr(loader, "task_names", None)
    if task_names and selected_indices:
        combined_name = "+".join(
            task_names[i] for i in selected_indices if 0 <= i < len(task_names)
        )
    else:
        combined_name = "task0"
    with open(task_order_path, "a", encoding="utf-8") as f_task_order:
        f_task_order.write(str(combined_name) + "\n")

    dummy_test_t = torch.empty((0,), dtype=torch.long)
    dummy_test_a = torch.empty((0, 0), dtype=torch.float)
    _ = confusion_matrix(
        result_val_t, result_val_a, args.log_dir, "results_single_round.txt"
    )
    save_results(
        args, result_val_t, result_val_a, dummy_test_t, dummy_test_a, model, time_spent
    )
    log_state(
        args.state_logging,
        "Single-round results saved; total runtime {:.2f}s".format(time_spent),
    )

    # Print and append total runtime for this single-round experiment.
    print("Total runtime: {:.2f} seconds".format(time_spent))
    results_txt_path = os.path.join(args.log_dir, "results.txt")
    try:
        with open(results_txt_path, "a", encoding="utf-8") as results_file:
            results_file.write("total_runtime_seconds: {:.3f}\n".format(time_spent))
    except OSError:
        # If results.txt cannot be written, fail silently to avoid breaking experiments.
        pass


if __name__ == "__main__":
    print("New Single-Round Experiment Starting...")
    main()
