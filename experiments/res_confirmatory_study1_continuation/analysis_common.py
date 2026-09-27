"""Shared frozen analysis utilities for the RES Study 1 continuation."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta

import study

CELL_ORDER = ("CODE_MAPPED_LOW", "NEUTRAL_LOW", "CODE_MAPPED_HIGH", "NEUTRAL_HIGH")


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
    if not path.exists():
        return []
    return study.read_jsonl(path)


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
        row = {**rendering, **observation}
        joined.append(row)
    joined.sort(key=lambda row: row["evaluation_order"])
    return joined


def complete_primary_groups(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["condition"] in study.PRIMARY_CONDITIONS and row.get("technical_status") == "VALID":
            groups[row["fixture_id"]][row["condition"]] = row
    complete, incomplete = {}, []
    all_fixture_ids = [row["fixture_id"] for row in study.read_jsonl(study.ROOT / "fixtures.jsonl")]
    for fixture_id in all_fixture_ids:
        group = groups.get(fixture_id, {})
        if set(group) == set(study.PRIMARY_CONDITIONS):
            complete[fixture_id] = group
        else:
            incomplete.append(
                {
                    "fixture_id": fixture_id,
                    "available_primary_conditions": sorted(group),
                    "missing_primary_conditions": sorted(set(study.PRIMARY_CONDITIONS) - set(group)),
                }
            )
    return complete, incomplete


def matrix_from_groups(groups: dict[str, dict[str, dict[str, Any]]]) -> tuple[list[str], np.ndarray]:
    fixture_ids = sorted(groups)
    matrix = np.asarray([[int(groups[fid][condition]["passed"]) for condition in CELL_ORDER] for fid in fixture_ids], dtype=int)
    return fixture_ids, matrix


def probability_effects(matrix: np.ndarray) -> dict[str, float]:
    cm_low, neutral_low, cm_high, neutral_high = matrix.mean(axis=0)
    low = neutral_low - cm_low
    high = neutral_high - cm_high
    return {
        "code_mapped_low": float(cm_low),
        "neutral_low": float(neutral_low),
        "code_mapped_high": float(cm_high),
        "neutral_high": float(neutral_high),
        "framing_effect_low": float(low),
        "framing_effect_high": float(high),
        "interaction": float(high - low),
    }


def bootstrap_probability_effects(matrix: np.ndarray, repetitions: int = 10000) -> dict[str, Any]:
    point = probability_effects(matrix)
    rng = np.random.default_rng(study.BOOTSTRAP_SEED)
    samples = np.empty((repetitions, 3), dtype=float)
    for index in range(repetitions):
        sampled = matrix[rng.integers(0, len(matrix), len(matrix))]
        effect = probability_effects(sampled)
        samples[index] = (effect["framing_effect_low"], effect["framing_effect_high"], effect["interaction"])
    intervals = np.quantile(samples, [0.025, 0.975], axis=0)
    return {
        **point,
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": study.BOOTSTRAP_SEED,
        "framing_effect_low_95_ci": intervals[:, 0].tolist(),
        "framing_effect_high_95_ci": intervals[:, 1].tolist(),
        "interaction_95_ci": intervals[:, 2].tolist(),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value

