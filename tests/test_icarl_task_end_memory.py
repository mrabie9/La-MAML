"""Regression tests for iCaRL task-end memory behavior."""

# ruff: noqa: E402

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.icarl import Net as IcarlNet


def _make_args() -> object:
    args = type("Args", (), {})()
    args.classes_per_task = [4]
    args.nc_per_task_list = ""
    args.nc_per_task = None
    args.batch_size = 8
    args.replay_batch_size = 8
    args.memories = 16
    args.n_memories = 16
    args.validation = 0.0
    args.inner_steps = 1
    args.n_meta = 1
    args.inner_steps = 1
    args.cuda = False
    args.learn_lr = False
    args.second_order = False
    args.sync_update = False
    args.meta_batches = 1
    args.opt_wt = 0.01
    args.opt_lr = 0.01
    args.alpha_init = 1e-3
    args.lr = 1e-3
    args.ctx_lr = 1e-3
    args.memory_strength = 0.1
    args.temperature = 5.0
    args.task_emb = 16
    args.samples_per_task = 8
    args.n_epochs = 1
    args.det_lambda = 1.0
    args.cls_lambda = 1.0
    args.det_memories = 0
    args.det_replay_batch = 0
    args.use_detector_arch = False
    args.arch = "resnet1d"
    args.dataset = "iq"
    args.class_weighted_ce = False
    args.grad_clip_norm = 5.0
    args.clipgrad = 5.0
    args.icarl_feature_chunk_size = 2
    return args


def test_icarl_task_end_rebuild_stores_exemplars_on_cpu() -> None:
    args = _make_args()
    model = IcarlNet(n_inputs=1024, n_outputs=4, n_tasks=1, args=args)
    model.train()

    x = torch.randn(8, 3, 1024)
    y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
    model.observe(x, y, t=0)

    assert model.memx is None
    assert model.memy is None
    assert model.mem_class_x, "Expected rebuilt exemplar memory."
    assert model.mem_class_y, "Expected distillation targets for exemplars."
    for exemplar_tensor in model.mem_class_x.values():
        assert exemplar_tensor.device.type == "cpu"
    for logits_tensor in model.mem_class_y.values():
        assert logits_tensor.device.type == "cpu"


def test_icarl_feature_extraction_is_chunked() -> None:
    args = _make_args()
    model = IcarlNet(n_inputs=1024, n_outputs=4, n_tasks=1, args=args)
    model.train()

    seen_batch_sizes: list[int] = []

    def _fake_feature_forward(batch: torch.Tensor) -> torch.Tensor:
        seen_batch_sizes.append(int(batch.size(0)))
        return torch.zeros((batch.size(0), model.n_feat), dtype=torch.float32)

    model.feature_forward = _fake_feature_forward  # type: ignore[method-assign]
    inputs = torch.randn(9, 3, 1024)
    feats = model._extract_features_chunked(inputs)

    assert feats.shape == (9, model.n_feat)
    assert seen_batch_sizes == [2, 2, 2, 2, 1]
