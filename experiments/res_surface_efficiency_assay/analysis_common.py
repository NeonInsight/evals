"""Shared frozen analysis utilities for the surface-efficiency assay."""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta

import study


def exact_binomial_interval(successes: int, total: int, alpha: float = 0.05) -> list[float]:
    if total == 0:
        return [float("nan"), float("nan")]
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    return [lower, upper]


def load_renderings() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = study.read_jsonl(study.ROOT / "rendered_prompts.jsonl")
    return rows, {row["observation_id"]: row for row in rows}


def load_observations(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or study.ROOT / "results" / "observations.jsonl"
    return study.read_jsonl(path) if path.exists() else []


def validate_and_join(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _, render_index = load_renderings()
    seen: set[str] = set()
    joined = []
    for observation in observations:
        observation_id = observation["observation_id"]
        if observation_id in seen or observation_id not in render_index:
            raise ValueError("Duplicate or unknown observation")
        seen.add(observation_id)
        rendering = render_index[observation_id]
        if observation["rendering_hash"] != rendering["rendering_hash"]:
            raise ValueError("Observation/rendering hash mismatch")
        joined.append({**rendering, **observation})
    joined.sort(key=lambda row: row["evaluation_order"])
    return joined


def complete_groups(
    rows: list[dict[str, Any]], required: tuple[str, ...]
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["condition"] in required and row.get("technical_status") == "VALID":
            groups[row["fixture_id"]][row["condition"]] = row
    complete, incomplete = {}, []
    for fixture in study.read_jsonl(study.ROOT / "fixtures.jsonl"):
        fixture_id = fixture["fixture_id"]
        group = groups.get(fixture_id, {})
        if set(group) == set(required):
            complete[fixture_id] = group
        else:
            incomplete.append(
                {
                    "fixture_id": fixture_id,
                    "available_conditions": sorted(group),
                    "missing_conditions": sorted(set(required) - set(group)),
                }
            )
    return complete, incomplete


def describe(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "standard_deviation": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
    }


def paired_accuracy_bootstrap(
    baseline: list[int], variant: list[int], repetitions: int = 10000
) -> dict[str, Any]:
    left = np.asarray(baseline, dtype=float)
    right = np.asarray(variant, dtype=float)
    if len(left) != len(right) or not len(left):
        raise ValueError("Paired bootstrap requires equal non-empty arrays")
    differences = right - left
    rng = np.random.default_rng(study.BOOTSTRAP_SEED)
    boot = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sampled = differences[rng.integers(0, len(differences), len(differences))]
        boot[index] = np.mean(sampled)
    return {
        "difference_variant_minus_baseline": float(np.mean(differences)),
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": study.BOOTSTRAP_SEED,
        "two_sided_95_ci": np.quantile(boot, [0.025, 0.975]).tolist(),
        "one_sided_95_lower_bound": float(np.quantile(boot, 0.05)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
