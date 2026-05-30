#!/usr/bin/env python3
"""
Summarise results from `probe_concurrency_eta.py`.

Given one or more JSON result files, it prints per-file comparisons and a
small aggregate over all inputs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# Algorithms omitted from CIL full-experiment schedules (TIL-only / not run in CIL).
CIL_SCHEDULE_EXCLUDED_ALGORITHMS: frozenset[str] = frozenset({"ctn", "packnet", "hat"})


def _load_json(path: Path) -> Dict[str, Any]:
    """Load a single JSON probe output file."""

    return json.loads(path.read_text(encoding="utf-8"))


def _format_seconds(value: Any) -> str:
    """Format seconds values for display."""

    if value is None:
        return "NA"
    try:
        return f"{float(value):.3f}s"
    except (TypeError, ValueError):
        return str(value)


def _parse_comma_list(value: str | None) -> set[str]:
    """Parse a comma-separated list into a set.

    Args:
        value: Comma-separated tokens (e.g. "iid2,rwalk"). `None` or empty
            strings return an empty set.

    Returns:
        Set of stripped tokens.
    """

    if value is None:
        return set()
    trimmed = value.strip()
    if not trimmed:
        return set()
    return {token.strip() for token in trimmed.split(",") if token.strip()}


def _resolve_excluded_models(
    exclude_algorithms: str | None,
    *,
    cil_schedule: bool,
) -> set[str]:
    """Merge CLI exclusions with optional CIL schedule defaults.

    Args:
        exclude_algorithms: Value of ``--exclude-algorithms`` (comma-separated).
        cil_schedule: When true, also exclude ``CIL_SCHEDULE_EXCLUDED_ALGORITHMS``.

    Returns:
        Set of algorithm names to omit from generated schedules.

    Usage:
        >>> sorted(_resolve_excluded_models("iid2", cil_schedule=True))
        ['ctn', 'hat', 'iid2', 'packnet']
    """

    excluded = _parse_comma_list(exclude_algorithms)
    if cil_schedule:
        excluded |= set(CIL_SCHEDULE_EXCLUDED_ALGORITHMS)
    return excluded


def _extract_comparison(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract comparison block from probe payload."""

    comparison = payload.get("comparison") or {}
    return comparison


def _extract_util_percent(snapshot: Any) -> Any:
    """Best-effort extraction of GPU util% from a `nvidia-smi` snapshot dict."""

    if not isinstance(snapshot, dict):
        return None
    gpus = snapshot.get("gpus")
    if not isinstance(gpus, list) or not gpus:
        return None
    gpu0 = gpus[0]
    if not isinstance(gpu0, dict):
        return None
    return gpu0.get("utilization.gpu")


def summarise_files(paths: Iterable[Path]) -> None:
    """Print a readable summary for each probe JSON file."""

    rows: List[Tuple[str, Dict[str, Any]]] = []
    for path in paths:
        payload = _load_json(path)
        rows.append((str(path), payload))

    total = len(rows)
    faster_count = 0
    serial_only_count = 0

    for path_str, payload in rows:
        mode = payload.get("mode")
        file_name = Path(path_str).name

        if mode == "serial_only":
            serial_payload = payload.get("serial") or {}
            serial_models = list(serial_payload.keys())
            serial_only_count += 1
            print(f"=== {file_name} (serial_only) ===")
            print("Serial results:")
            for model_name in sorted(serial_models):
                job = serial_payload.get(model_name) or {}
                eta_seconds = job.get("eta_seconds")
                util_avg = job.get("util_avg_epochstart_to_eta_capture_percent")
                util_avg_n = job.get("util_avg_n_samples")
                if util_avg is None:
                    util_avg = _extract_util_percent(job.get("util_at_eta_capture"))
                print(
                    f"- {model_name}: eta={_format_seconds(eta_seconds)}, util_avg(epoch_start->eta)={util_avg}% (n={util_avg_n})"
                )
            print()
            continue

        if mode == "auto_pairing_serial_and_parallel":
            auto_pairs = payload.get("auto_pair_models") or []
            comparisons = payload.get("auto_pair_comparisons") or []
            serial_map = payload.get("serial") or {}
            serial_total_eta_seconds = 0.0
            for job in serial_map.values():
                eta = job.get("eta_seconds") if isinstance(job, dict) else None
                if eta is not None:
                    serial_total_eta_seconds += float(eta)

            kept_pair_models: set[str] = set()
            parallel_total_eta_seconds = 0.0
            print(f"=== {file_name} (auto_pairing_serial_and_parallel) ===")
            for comp in comparisons:
                pair_idx = comp.get("pair_idx")
                pair = (
                    auto_pairs[pair_idx]
                    if isinstance(pair_idx, int) and pair_idx < len(auto_pairs)
                    else {}
                )
                high_model = pair.get("A", "?")
                low_model = pair.get("B", "?")
                serial_sum_eta = comp.get("serial_sum_eta_seconds")
                parallel_max_eta = comp.get("parallel_max_eta_seconds")
                delta = comp.get("parallel_max_eta_minus_serial_sum_eta_seconds")
                parallel_wall = comp.get("parallel_wall_clock_seconds")
                high_util = comp.get("serial_util_high_percent")
                low_util = comp.get("serial_util_low_percent")
                parallel_is_faster = bool(
                    comp.get("parallel_is_faster_than_serial_sum")
                )
                print(
                    f"[pair {pair_idx+1}/{len(comparisons)}] "
                    f"{high_model}(util={high_util}%) + {low_model}(util={low_util}%) "
                    f"serial_sum_eta={_format_seconds(serial_sum_eta)} "
                    f"parallel_eta_max={_format_seconds(parallel_max_eta)} "
                    f"delta={_format_seconds(delta)} "
                    f"(parallel_wall={_format_seconds(parallel_wall)})"
                )
                if (
                    parallel_is_faster
                    and high_model in serial_map
                    and low_model in serial_map
                ):
                    kept_pair_models.add(high_model)
                    kept_pair_models.add(low_model)
                    if parallel_max_eta is not None:
                        parallel_total_eta_seconds += float(parallel_max_eta)
            print()

            unpaired_models = sorted(
                m for m in serial_map.keys() if m not in kept_pair_models
            )
            for model_name in unpaired_models:
                eta = serial_map.get(model_name, {}).get("eta_seconds")
                if eta is not None:
                    parallel_total_eta_seconds += float(eta)

            percent_time_saved = (
                (serial_total_eta_seconds - parallel_total_eta_seconds)
                / serial_total_eta_seconds
                * 100.0
                if serial_total_eta_seconds > 0
                else None
            )
            if percent_time_saved is not None:
                print(
                    "Total estimated time saved (parallel vs serial): "
                    f"{percent_time_saved:.2f}% "
                    f"(serial_total={_format_seconds(serial_total_eta_seconds)}, "
                    f"parallel_total={_format_seconds(parallel_total_eta_seconds)})"
                )
            else:
                print(
                    "Total estimated time saved (parallel vs serial): NA "
                    "(missing eta_seconds)"
                )

            if unpaired_models:
                print("Unpaired algorithms (run serially):")
                for model_name in unpaired_models:
                    job = serial_map.get(model_name, {}) or {}
                    eta_seconds = job.get("eta_seconds")
                    util_avg = job.get("util_avg_epochstart_to_eta_capture_percent")
                    util_avg_n = job.get("util_avg_n_samples")
                    if util_avg is None:
                        util_avg = _extract_util_percent(job.get("util_at_eta_capture"))
                    print(
                        f"- {model_name}: eta={_format_seconds(eta_seconds)}, "
                        f"util_avg(epoch_start->eta)={util_avg}% (n={util_avg_n})"
                    )
            else:
                print("Unpaired algorithms (run serially): none")
            continue

        # Default: serial_and_parallel (legacy schema)
        pair = payload.get("pair_models", {}) or {}
        model_a = pair.get("A", "?")
        model_b = pair.get("B", "?")
        comparison = _extract_comparison(payload)
        parallel_is_faster = bool(comparison.get("parallel_is_faster_than_serial_sum"))
        faster_count += 1 if parallel_is_faster else 0

        serial_sum_eta = comparison.get("serial_sum_eta_seconds")
        parallel_max_eta = comparison.get("parallel_max_eta_seconds")
        delta = comparison.get("parallel_max_eta_minus_serial_sum_eta_seconds")

        parallel_wall = payload.get("parallel_wall_clock_seconds")

        serial_a = payload.get("serial", {}).get(model_a, {})
        serial_b = payload.get("serial", {}).get(model_b, {})
        parallel_map = payload.get("parallel") or {}
        parallel_a = parallel_map.get(model_a, {})
        parallel_b = parallel_map.get(model_b, {})

        serial_a_eta = serial_a.get("eta_seconds")
        serial_b_eta = serial_b.get("eta_seconds")
        parallel_a_eta = parallel_a.get("eta_seconds")
        parallel_b_eta = parallel_b.get("eta_seconds")

        serial_a_util_avg = serial_a.get("util_avg_epochstart_to_eta_capture_percent")
        serial_b_util_avg = serial_b.get("util_avg_epochstart_to_eta_capture_percent")
        parallel_a_util_avg = parallel_a.get(
            "util_avg_epochstart_to_eta_capture_percent"
        )
        parallel_b_util_avg = parallel_b.get(
            "util_avg_epochstart_to_eta_capture_percent"
        )
        if serial_a_util_avg is None:
            serial_a_util_avg = _extract_util_percent(
                serial_a.get("util_at_eta_capture")
            )
        if serial_b_util_avg is None:
            serial_b_util_avg = _extract_util_percent(
                serial_b.get("util_at_eta_capture")
            )
        if parallel_a_util_avg is None:
            parallel_a_util_avg = _extract_util_percent(
                parallel_a.get("util_at_eta_capture")
            )
        if parallel_b_util_avg is None:
            parallel_b_util_avg = _extract_util_percent(
                parallel_b.get("util_at_eta_capture")
            )

        print(f"=== {file_name} ===")
        print(f"Pair: {model_a} + {model_b}")
        print(
            "Serial "
            f"{model_a}: eta={_format_seconds(serial_a_eta)}, util_avg={serial_a_util_avg}%, "
            f"{model_b}: eta={_format_seconds(serial_b_eta)}, util_avg={serial_b_util_avg}%"
        )
        print(
            "Parallel "
            f"{model_a}: eta={_format_seconds(parallel_a_eta)}, util_avg={parallel_a_util_avg}%, "
            f"{model_b}: eta={_format_seconds(parallel_b_eta)}, util_avg={parallel_b_util_avg}%"
        )
        print(
            "Comparison: "
            f"serial_sum_eta={_format_seconds(serial_sum_eta)}, "
            f"parallel_max_eta={_format_seconds(parallel_max_eta)}, "
            f"delta={_format_seconds(delta)} "
            f"(parallel_faster_than_serial_sum={parallel_is_faster})"
        )
        print(f"Parallel wall-clock elapsed: {_format_seconds(parallel_wall)}")
        print()

    if faster_count:
        print(
            f"Aggregate: {faster_count}/{total} parallel runs faster than serial sum."
        )
    else:
        # Avoid misleading aggregate when we only saw serial-only JSONs.
        if serial_only_count:
            print(f"Aggregate: {serial_only_count}/{total} serial_only runs.")


def _generate_host_schedule_from_probe(
    probe_payload: Dict[str, Any],
    host1: str,
    host2: str,
    excluded_models: set[str],
) -> Dict[str, Any]:
    """Generate a host split (pairs + singles) from auto-pair probe JSON.

    Pairing rule:
    - Keep a pair concurrent iff `parallel_is_faster_than_serial_sum` is true.
    - Otherwise, treat both algorithms in that pair as serial singles.

    Host split rule:
    - Assign pairs and singles greedily to the host with smaller accumulated
      estimated wall-clock time.
    - Pair wall-clock estimate is `parallel_max_eta_seconds`.
    - Single wall-clock estimate is `serial[algo].eta_seconds`.

    Returns:
        Schedule dict with `hosts` and `stats` keys.
    """

    serial_map = probe_payload.get("serial") or {}
    if not isinstance(serial_map, dict) or not serial_map:
        raise SystemExit("Probe JSON missing `serial` mapping.")

    auto_pairs = probe_payload.get("auto_pair_models") or []
    auto_comparisons = probe_payload.get("auto_pair_comparisons") or []
    if not isinstance(auto_pairs, list) or not isinstance(auto_comparisons, list):
        raise SystemExit("Probe JSON missing auto_pair_* lists.")

    excluded_models_normalised = {str(m) for m in excluded_models}
    candidate_models = [
        str(k) for k in serial_map.keys() if str(k) not in excluded_models_normalised
    ]
    serial_total_eta = 0.0
    for model_name in candidate_models:
        eta = serial_map.get(model_name, {}).get("eta_seconds")
        if eta is None:
            raise SystemExit(
                f"Probe JSON missing serial eta_seconds for '{model_name}'."
            )
        serial_total_eta += float(eta)

    # Start with everything as a serial single, then remove algorithms from
    # kept concurrent pairs.
    unpaired_models = set(candidate_models)
    kept_pairs: list[dict[str, Any]] = []

    for pair_idx, pair in enumerate(auto_pairs):
        if not isinstance(pair, dict):
            continue
        a = pair.get("A")
        b = pair.get("B")
        if not isinstance(a, str) or not isinstance(b, str):
            continue

        comp = auto_comparisons[pair_idx] if pair_idx < len(auto_comparisons) else {}
        parallel_faster = bool(comp.get("parallel_is_faster_than_serial_sum"))
        if not parallel_faster:
            # Keep both as serial singles.
            continue

        # If either algorithm is excluded, we never keep it as a concurrent pair.
        if a in excluded_models_normalised or b in excluded_models_normalised:
            continue

        parallel_max_eta_seconds = comp.get("parallel_max_eta_seconds")
        if parallel_max_eta_seconds is None:
            parallel_max_eta_seconds = comp.get("parallel_max_eta_seconds")

        kept_pairs.append(
            {
                "pair_idx": pair_idx,
                "A": a,
                "B": b,
                "parallel_max_eta_seconds": (
                    float(parallel_max_eta_seconds)
                    if parallel_max_eta_seconds is not None
                    else None
                ),
                "delta_seconds": comp.get(
                    "parallel_max_eta_minus_serial_sum_eta_seconds"
                ),
            }
        )
        unpaired_models.discard(a)
        unpaired_models.discard(b)

    # Greedy assignment of kept pairs.
    host_time: dict[str, float] = {host1: 0.0, host2: 0.0}
    host_pairs: dict[str, list[dict[str, Any]]] = {host1: [], host2: []}
    host_singles: dict[str, list[str]] = {host1: [], host2: []}

    def _pair_sort_key(pair_unit: Dict[str, Any]) -> tuple[float, int]:
        est = pair_unit.get("parallel_max_eta_seconds")
        est_f = float(est) if est is not None else 0.0
        return (-est_f, int(pair_unit.get("pair_idx", 0)))

    kept_pairs_sorted = sorted(kept_pairs, key=_pair_sort_key)
    for pair_unit in kept_pairs_sorted:
        # Assign to the host with smaller accumulated time.
        chosen_host = host1 if host_time[host1] <= host_time[host2] else host2
        pair_time = float(pair_unit.get("parallel_max_eta_seconds") or 0.0)
        host_pairs[chosen_host].append(pair_unit)
        host_time[chosen_host] += pair_time

    # Greedy assignment of remaining singles (descending eta).
    remaining_singles = sorted(
        unpaired_models,
        key=lambda m: float(serial_map[m].get("eta_seconds") or 0.0),
        reverse=True,
    )
    for model_name in remaining_singles:
        chosen_host = host1 if host_time[host1] <= host_time[host2] else host2
        host_time[chosen_host] += float(
            serial_map[model_name].get("eta_seconds") or 0.0
        )
        host_singles[chosen_host].append(model_name)

    makespan_est = max(host_time[host1], host_time[host2])
    percent_time_saved = (
        (serial_total_eta - makespan_est) / serial_total_eta * 100.0
        if serial_total_eta > 0
        else None
    )

    return {
        "hosts": {
            host1: {
                "pairs": host_pairs[host1],
                "singles": host_singles[host1],
                "estimated_host_wall_seconds": host_time[host1],
            },
            host2: {
                "pairs": host_pairs[host2],
                "singles": host_singles[host2],
                "estimated_host_wall_seconds": host_time[host2],
            },
        },
        "stats": {
            "serial_total_eta_sum_seconds": serial_total_eta,
            "parallel_makespan_estimated_seconds": makespan_est,
            "percent_time_saved_parallel_vs_serial": percent_time_saved,
        },
    }


def _models_from_serial_probe(
    probe_payload: Dict[str, Any],
    excluded_models: set[str],
) -> list[dict[str, Any]]:
    """Extract per-model ETA and util from a serial-only probe JSON.

    Args:
        probe_payload: Probe output dict with a `serial` mapping.
        excluded_models: Model names to omit from the schedule.

    Returns:
        List of dicts with keys `name`, `eta_seconds`, and `util_percent`.

    Raises:
        SystemExit: When required probe fields are missing.
    """

    serial_map = probe_payload.get("serial") or {}
    if not isinstance(serial_map, dict) or not serial_map:
        raise SystemExit("Probe JSON missing `serial` mapping.")

    excluded_models_normalised = {str(m) for m in excluded_models}
    models: list[dict[str, Any]] = []
    for model_name, job in serial_map.items():
        if str(model_name) in excluded_models_normalised:
            continue
        if not isinstance(job, dict):
            continue

        eta_seconds = job.get("eta_seconds")
        if eta_seconds is None:
            raise SystemExit(f"Probe JSON missing eta_seconds for '{model_name}'.")

        util_avg = job.get("util_avg_epochstart_to_eta_capture_percent")
        if util_avg is None:
            util_avg = _extract_util_percent(job.get("util_at_eta_capture"))

        if util_avg is None:
            raise SystemExit(
                "Probe JSON missing util_avg fields and could not fall back to "
                "`util_at_eta_capture` for util extraction."
            )

        models.append(
            {
                "name": str(model_name),
                "eta_seconds": float(eta_seconds),
                "util_percent": float(util_avg),
            }
        )

    if not models:
        raise SystemExit("No usable models found in serial probe JSON.")
    return models


def _split_models_by_util_tier(
    models: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split models into high- and low-util tiers (high gets ceil(n/2))."""

    models_sorted_by_util = sorted(
        models, key=lambda m: m["util_percent"], reverse=True
    )
    cut_high = int(math.ceil(len(models_sorted_by_util) / 2.0))
    return models_sorted_by_util[:cut_high], models_sorted_by_util[cut_high:]


def _fast_host_wall(high_sum: float, low_sum: float) -> float:
    """Estimated wall time for a host with concurrent high/low slots."""

    return max(high_sum, low_sum)


def _three_host_makespan(
    host_high_sum: dict[str, float],
    host_low_sum: dict[str, float],
    slow_host_serial_sum: float,
    fast_hosts: tuple[str, str],
    slow_host: str,
) -> float:
    """Global finish time across two fast hosts and one serial slow host."""

    fast_walls = [
        _fast_host_wall(host_high_sum[h], host_low_sum[h]) for h in fast_hosts
    ]
    return max(*fast_walls, slow_host_serial_sum)


def _build_three_host_state_from_assignment(
    assignment: dict[str, tuple[str, str]],
    eta_by_name: dict[str, float],
    host1: str,
    host2: str,
    slow_host: str,
    slow_host_speed_factor: float,
) -> tuple[
    dict[str, list[str]],
    dict[str, list[str]],
    dict[str, float],
    dict[str, float],
    float,
]:
    """Rebuild slot queues and load sums from a model->(host, slot) assignment."""

    host_slot_high: dict[str, list[str]] = {host1: [], host2: [], slow_host: []}
    host_slot_low: dict[str, list[str]] = {host1: [], host2: [], slow_host: []}
    host_high_sum = {host1: 0.0, host2: 0.0}
    host_low_sum = {host1: 0.0, host2: 0.0}
    slow_host_serial_sum = 0.0

    for model_name, (host, slot) in assignment.items():
        eta = eta_by_name[model_name]
        if host == slow_host:
            host_slot_high[slow_host].append(model_name)
            slow_host_serial_sum += eta * slow_host_speed_factor
            continue
        if slot == "high":
            host_slot_high[host].append(model_name)
            host_high_sum[host] += eta
        else:
            host_slot_low[host].append(model_name)
            host_low_sum[host] += eta

    return (
        host_slot_high,
        host_slot_low,
        host_high_sum,
        host_low_sum,
        slow_host_serial_sum,
    )


def _assign_three_host_queue_schedule(
    high_models: list[dict[str, Any]],
    low_models: list[dict[str, Any]],
    host1: str,
    host2: str,
    slow_host: str,
    slow_host_speed_factor: float,
) -> tuple[
    dict[str, list[str]],
    dict[str, list[str]],
    dict[str, float],
    dict[str, float],
    float,
]:
    """Assign models across three hosts to minimise estimated finish time.

    Fast hosts (`host1`, `host2`) run util-ranked `slot_high` and `slot_low`
    queues concurrently (two jobs at a time). The slow host runs jobs strictly
    one after another; each job's ETA is scaled by `slow_host_speed_factor`.

    Uses longest-processing-time greedy placement with a local-search pass.

    Returns:
        Tuple of (slot_high, slot_low, host_high_sum, host_low_sum,
        slow_host_serial_sum).
    """

    if slow_host_speed_factor <= 0.0:
        raise SystemExit("--host3-speed-factor must be positive.")

    fast_hosts = (host1, host2)
    all_models = high_models + low_models
    eta_by_name = {m["name"]: float(m["eta_seconds"]) for m in all_models}
    util_tier_by_name = {m["name"]: "high" for m in high_models} | {
        m["name"]: "low" for m in low_models
    }

    assignment: dict[str, tuple[str, str]] = {}

    def _state_from_assignment(
        trial_assignment: dict[str, tuple[str, str]],
    ) -> tuple[dict[str, float], dict[str, float], float]:
        _, _, high_sum, low_sum, slow_sum = _build_three_host_state_from_assignment(
            trial_assignment,
            eta_by_name,
            host1,
            host2,
            slow_host,
            slow_host_speed_factor,
        )
        return high_sum, low_sum, slow_sum

    def _makespan_for_assignment(trial_assignment: dict[str, tuple[str, str]]) -> float:
        high_sum, low_sum, slow_sum = _state_from_assignment(trial_assignment)
        return _three_host_makespan(high_sum, low_sum, slow_sum, fast_hosts, slow_host)

    def _best_placement(
        model_name: str,
        util_tier: str,
        base_assignment: dict[str, tuple[str, str]],
    ) -> tuple[float, str, str]:
        best: tuple[float, str, str] | None = None

        if util_tier == "high":
            for fast_host in fast_hosts:
                trial = dict(base_assignment)
                trial[model_name] = (fast_host, "high")
                makespan = _makespan_for_assignment(trial)
                if best is None or makespan < best[0]:
                    best = (makespan, fast_host, "high")
            trial = dict(base_assignment)
            trial[model_name] = (slow_host, "high")
            makespan = _makespan_for_assignment(trial)
            if best is None or makespan < best[0]:
                best = (makespan, slow_host, "high")
        else:
            for fast_host in fast_hosts:
                trial = dict(base_assignment)
                trial[model_name] = (fast_host, "low")
                makespan = _makespan_for_assignment(trial)
                if best is None or makespan < best[0]:
                    best = (makespan, fast_host, "low")
            trial = dict(base_assignment)
            trial[model_name] = (slow_host, "low")
            makespan = _makespan_for_assignment(trial)
            if best is None or makespan < best[0]:
                best = (makespan, slow_host, "low")

        assert best is not None
        return best

    for util_tier, tier_models in (("high", high_models), ("low", low_models)):
        ordered = sorted(tier_models, key=lambda m: m["eta_seconds"], reverse=True)
        for model in ordered:
            _, chosen_host, chosen_slot = _best_placement(
                model["name"], util_tier, assignment
            )
            assignment[model["name"]] = (chosen_host, chosen_slot)

    improved = True
    while improved:
        improved = False
        baseline = _makespan_for_assignment(assignment)
        for model_name in sorted(assignment.keys()):
            util_tier = util_tier_by_name[model_name]
            without = {k: v for k, v in assignment.items() if k != model_name}
            makespan, chosen_host, chosen_slot = _best_placement(
                model_name, util_tier, without
            )
            if makespan >= baseline:
                continue
            assignment[model_name] = (chosen_host, chosen_slot)
            baseline = makespan
            improved = True

    return _build_three_host_state_from_assignment(
        assignment,
        eta_by_name,
        host1,
        host2,
        slow_host,
        slow_host_speed_factor,
    )


def _generate_queue_host_schedule_from_serial_probe(
    probe_payload: Dict[str, Any],
    host1: str,
    host2: str,
    excluded_models: set[str],
    *,
    host3: str | None = None,
    host3_speed_factor: float = 1.7,
) -> Dict[str, Any]:
    """Generate a util-ranked high/low queue schedule from serial probe JSON.

    With two fast hosts, each runs `slot_high` and `slot_low` concurrently.
    When `host3` is set, a third slow host is included: jobs run serially there
    with ETA scaled by `host3_speed_factor`, and the assignment minimises the
    estimated global finish time across all three hosts.

    Returns:
        A schedule dict suitable for `scripts/full_experiments.sh`.
    """

    models = _models_from_serial_probe(probe_payload, excluded_models)
    high_models, low_models = _split_models_by_util_tier(models)
    models_sorted_by_util = sorted(
        models, key=lambda m: m["util_percent"], reverse=True
    )
    serial_total_eta = sum(m["eta_seconds"] for m in models_sorted_by_util)

    if host3 is not None:
        host_slot_high, host_slot_low, host_high_sum, host_low_sum, slow_sum = (
            _assign_three_host_queue_schedule(
                high_models=high_models,
                low_models=low_models,
                host1=host1,
                host2=host2,
                slow_host=host3,
                slow_host_speed_factor=host3_speed_factor,
            )
        )
        host1_wall = _fast_host_wall(host_high_sum[host1], host_low_sum[host1])
        host2_wall = _fast_host_wall(host_high_sum[host2], host_low_sum[host2])
        host3_wall = slow_sum
        makespan_est = max(host1_wall, host2_wall, host3_wall)
        percent_time_saved = (
            (serial_total_eta - makespan_est) / serial_total_eta * 100.0
            if serial_total_eta > 0
            else None
        )
        return {
            "hosts": {
                host1: {
                    "slot_high": host_slot_high[host1],
                    "slot_low": host_slot_low[host1],
                    "estimated_host_wall_seconds": host1_wall,
                },
                host2: {
                    "slot_high": host_slot_high[host2],
                    "slot_low": host_slot_low[host2],
                    "estimated_host_wall_seconds": host2_wall,
                },
                host3: {
                    "slot_high": host_slot_high[host3],
                    "slot_low": [],
                    "serial_only": True,
                    "estimated_host_wall_seconds": host3_wall,
                },
            },
            "stats": {
                "serial_total_eta_sum_seconds": serial_total_eta,
                "parallel_makespan_estimated_seconds": makespan_est,
                "percent_time_saved_parallel_vs_serial": percent_time_saved,
                "high_count": len(high_models),
                "low_count": len(low_models),
                "host3_speed_factor": host3_speed_factor,
            },
            "mode": "util_high_low_queue_schedule_3host",
        }

    host_high_sum: dict[str, float] = {host1: 0.0, host2: 0.0}
    host_low_sum: dict[str, float] = {host1: 0.0, host2: 0.0}
    host_slot_high: dict[str, list[str]] = {host1: [], host2: []}
    host_slot_low: dict[str, list[str]] = {host1: [], host2: []}

    def _choose_host_for_high(eta_seconds: float) -> str:
        candidates: list[tuple[float, float, str]] = []
        for h in (host1, host2):
            candidate_high = host_high_sum[h] + eta_seconds
            candidate_wall = max(candidate_high, host_low_sum[h])
            candidates.append((candidate_wall, candidate_high, h))
        candidates_sorted = sorted(candidates, key=lambda x: (x[0], x[1]))
        return candidates_sorted[0][2]

    def _choose_host_for_low(eta_seconds: float) -> str:
        candidates: list[tuple[float, float, str]] = []
        for h in (host1, host2):
            candidate_low = host_low_sum[h] + eta_seconds
            candidate_wall = max(host_high_sum[h], candidate_low)
            candidates.append((candidate_wall, candidate_low, h))
        candidates_sorted = sorted(candidates, key=lambda x: (x[0], x[1]))
        return candidates_sorted[0][2]

    for m in high_models:
        chosen_host = _choose_host_for_high(m["eta_seconds"])
        host_slot_high[chosen_host].append(m["name"])
        host_high_sum[chosen_host] += float(m["eta_seconds"])

    for m in low_models:
        chosen_host = _choose_host_for_low(m["eta_seconds"])
        host_slot_low[chosen_host].append(m["name"])
        host_low_sum[chosen_host] += float(m["eta_seconds"])

    host1_wall = _fast_host_wall(host_high_sum[host1], host_low_sum[host1])
    host2_wall = _fast_host_wall(host_high_sum[host2], host_low_sum[host2])
    makespan_est = max(host1_wall, host2_wall)

    percent_time_saved = (
        (serial_total_eta - makespan_est) / serial_total_eta * 100.0
        if serial_total_eta > 0
        else None
    )

    return {
        "hosts": {
            host1: {
                "slot_high": host_slot_high[host1],
                "slot_low": host_slot_low[host1],
                "estimated_host_wall_seconds": host1_wall,
            },
            host2: {
                "slot_high": host_slot_high[host2],
                "slot_low": host_slot_low[host2],
                "estimated_host_wall_seconds": host2_wall,
            },
        },
        "stats": {
            "serial_total_eta_sum_seconds": serial_total_eta,
            "parallel_makespan_estimated_seconds": makespan_est,
            "percent_time_saved_parallel_vs_serial": percent_time_saved,
            "high_count": len(high_models),
            "low_count": len(low_models),
        },
        "mode": "util_high_low_queue_schedule",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise ETA probe JSON results.")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Paths to JSON files produced by probe_concurrency_eta.py.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default=None,
        help="Optional glob pattern (e.g. 'logs/eta_probe/*.json') to select inputs.",
    )
    parser.add_argument(
        "--generate-host-schedule",
        action="store_true",
        default=False,
        help="Generate host split (pairs + singles) from an auto-pair probe JSON.",
    )
    parser.add_argument(
        "--generate-queue-host-schedule",
        action="store_true",
        default=False,
        help=(
            "Generate host schedule using util-ranked high/low 2-slot queues "
            "from a serial-only probe JSON."
        ),
    )
    parser.add_argument(
        "--probe-json",
        type=str,
        default="logs/eta_probe/eta_probe_elkk-1-algos_10-iter__avg-util-auto-pair.json",
        help="Probe JSON to generate the schedule from.",
    )
    parser.add_argument(
        "--serial-probe-json",
        type=str,
        default="logs/eta_probe/eta_probe_elkk-1-algos_10-iter_avg-util-2.json",
        help="Serial-only probe JSON to generate the queue schedule from.",
    )
    parser.add_argument(
        "--schedule-out-json",
        type=str,
        default="logs/eta_probe/full_experiments_host_schedule.json",
        help="Where to write the generated schedule JSON.",
    )
    parser.add_argument(
        "--host1",
        type=str,
        default="lnx-elkk-1",
        help="Host shortname for host 1 in the generated schedule.",
    )
    parser.add_argument(
        "--host2",
        type=str,
        default="lnx-elkk-2",
        help="Host shortname for host 2 in the generated schedule.",
    )
    parser.add_argument(
        "--exclude-algorithms",
        type=str,
        default=None,
        help=(
            "Comma-separated list of algorithm/model names to exclude from the "
            "generated schedule (e.g. 'iid2,rwalk')."
        ),
    )
    parser.add_argument(
        "--cil-schedule",
        action="store_true",
        default=False,
        help=(
            "CIL schedule preset: exclude ctn, packnet, and hat from the "
            "generated schedule (merged with --exclude-algorithms)."
        ),
    )
    parser.add_argument(
        "--three-host-schedule",
        action="store_true",
        default=False,
        help=(
            "Include a third slow host in the queue schedule (serial-only, no "
            "concurrent jobs). Requires --host3."
        ),
    )
    parser.add_argument(
        "--host3",
        type=str,
        default="win-lbo-22410",
        help="Slow host shortname for 3-host queue schedules.",
    )
    parser.add_argument(
        "--host3-speed-factor",
        type=float,
        default=1.7,
        help=(
            "ETA multiplier on the slow host relative to probe timings "
            "(default: 1.7)."
        ),
    )
    args = parser.parse_args()

    excluded_models = _resolve_excluded_models(
        args.exclude_algorithms,
        cil_schedule=bool(args.cil_schedule),
    )

    if args.generate_queue_host_schedule:
        probe_path = Path(args.serial_probe_json)
        if not probe_path.exists():
            raise SystemExit(f"Missing serial probe JSON: {probe_path}")
        payload = _load_json(probe_path)
        host3 = args.host3 if args.three_host_schedule else None
        schedule = _generate_queue_host_schedule_from_serial_probe(
            probe_payload=payload,
            host1=args.host1,
            host2=args.host2,
            excluded_models=excluded_models,
            host3=host3,
            host3_speed_factor=float(args.host3_speed_factor),
        )
        schedule_out = Path(args.schedule_out_json)
        schedule_out.parent.mkdir(parents=True, exist_ok=True)
        schedule_out.write_text(json.dumps(schedule, indent=2), encoding="utf-8")

        stats = schedule.get("stats") or {}
        percent_saved = stats.get("percent_time_saved_parallel_vs_serial")
        serial_total = stats.get("serial_total_eta_sum_seconds")
        parallel_makespan = stats.get("parallel_makespan_estimated_seconds")
        if percent_saved is not None:
            print(
                f"Estimated time saved (parallel vs serial): {percent_saved:.2f}% "
                f"(serial_total={_format_seconds(serial_total)}, "
                f"parallel_makespan={_format_seconds(parallel_makespan)})"
            )
        else:
            print("Estimated time saved (parallel vs serial): NA")
        print(f"Wrote schedule to: {schedule_out}")
        return

    if args.generate_host_schedule:
        probe_path = Path(args.probe_json)
        if not probe_path.exists():
            raise SystemExit(f"Missing probe JSON: {probe_path}")
        payload = _load_json(probe_path)
        schedule = _generate_host_schedule_from_probe(
            probe_payload=payload,
            host1=args.host1,
            host2=args.host2,
            excluded_models=excluded_models,
        )
        schedule_out = Path(args.schedule_out_json)
        schedule_out.parent.mkdir(parents=True, exist_ok=True)
        schedule_out.write_text(json.dumps(schedule, indent=2), encoding="utf-8")

        stats = schedule.get("stats") or {}
        percent_saved = stats.get("percent_time_saved_parallel_vs_serial")
        serial_total = stats.get("serial_total_eta_sum_seconds")
        parallel_makespan = stats.get("parallel_makespan_estimated_seconds")
        if percent_saved is not None:
            print(
                f"Estimated time saved (parallel vs serial): {percent_saved:.2f}% "
                f"(serial_total={_format_seconds(serial_total)}, "
                f"parallel_makespan={_format_seconds(parallel_makespan)})"
            )
        else:
            print("Estimated time saved (parallel vs serial): NA")
        print(f"Wrote schedule to: {schedule_out}")
        return

    input_paths: List[Path] = []
    if args.paths:
        input_paths = [Path(p) for p in args.paths]
    elif args.glob:
        input_paths = [Path(p) for p in sorted(Path().glob(args.glob))]
    else:
        raise SystemExit("Provide JSON paths or use --glob.")

    missing = [p for p in input_paths if not p.exists()]
    if missing:
        raise SystemExit(f"Missing input files: {', '.join(str(p) for p in missing)}")

    summarise_files(input_paths)


if __name__ == "__main__":
    main()
