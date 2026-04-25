"""Unit tests for hyperparameter tuner path/writeback safeguards."""

from __future__ import annotations

# ruff: noqa: E402

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tuning import hyperparam_tuner


def test_resolve_cli_config_path_uses_caller_cwd(tmp_path: Path) -> None:
    """Relative config paths should resolve from caller working directory."""
    caller_cwd = tmp_path / "scripts"
    caller_cwd.mkdir(parents=True, exist_ok=True)
    target_config = tmp_path / "configs" / "models" / "til" / "hat.yaml"
    target_config.parent.mkdir(parents=True, exist_ok=True)
    target_config.write_text("model: hat\n", encoding="utf-8")

    previous_cwd = Path.cwd()
    try:
        os.chdir(caller_cwd)
        resolved = hyperparam_tuner.resolve_cli_config_path(
            "../configs/models/til/hat.yaml"
        )
    finally:
        os.chdir(previous_cwd)

    assert resolved == target_config.resolve()


def test_run_tuning_records_yaml_error_without_failing(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """YAML writeback failures should not fail a successful tuning run."""
    captured_summaries: list[dict[str, Any]] = []

    cli = SimpleNamespace(
        config=["../configs/models/til/hat.yaml"],
        config_dir=[],
        override=[],
        grid=[],
        num_samples=None,
        search_seed=0,
        max_trials=None,
        shuffle=False,
        tune_only=[],
        hierarchical=False,
        lr_first=False,
        lr_key="lr",
        dry_run=False,
        output_root=str(tmp_path / "out"),
        seed_offset=0,
        vary_seed=False,
        keep_expt_name=False,
    )

    class _FakeCliParser:
        def parse_args(self) -> SimpleNamespace:
            return cli

    monkeypatch.setattr(hyperparam_tuner, "build_cli", lambda _preset: _FakeCliParser())
    monkeypatch.setattr(
        hyperparam_tuner.file_parser,
        "parse_args_from_yaml",
        lambda _sources: SimpleNamespace(model="hat", expt_name="hat", seed=0),
    )
    monkeypatch.setattr(hyperparam_tuner, "parse_override_specs", lambda *_: {})
    monkeypatch.setattr(hyperparam_tuner, "expand_trials", lambda *_: [{}])
    monkeypatch.setattr(
        hyperparam_tuner.misc_utils, "get_date_time", lambda: "2026-01-01_00-00-00"
    )
    monkeypatch.setattr(
        hyperparam_tuner,
        "run_single_trial",
        lambda *args, **kwargs: {
            "status": "ok",
            "trial": 0,
            "params": {"lr": 0.001},
            "trial_params": {"lr": 0.001},
            "score": 0.9,
            "duration_sec": 0.1,
            "log_dir": str(tmp_path / "run"),
        },
    )
    monkeypatch.setattr(
        hyperparam_tuner,
        "dump_summary",
        lambda _session_dir, summary, _successes: captured_summaries.append(
            dict(summary)
        ),
    )
    monkeypatch.setattr(
        hyperparam_tuner,
        "write_best_params_to_yaml",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("write failed")),
    )

    preset = hyperparam_tuner.TuningPreset(model_name="hat")
    hyperparam_tuner.run_tuning(preset)

    assert captured_summaries, "Expected summary persistence calls."
    assert any(
        summary.get("updated_yaml_error") == "write failed"
        for summary in captured_summaries
    )
