#!/usr/bin/env python3
"""Aggregate the preregistered Qwen3B five-factor failure-mode ablation."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path("experiments/res_step0_interface_gate")
OUT_DIR = Path(os.environ.get("RES_INTERFACE_OUTPUT_DIR", ROOT / "results"))
SEQUENCE_ID = os.environ.get("RES_ABLATION_SEQUENCE_ID", "qwen3b-five-factor-ablation-v1")
RESULT_PREFIX = os.environ.get(
    "RES_ABLATION_RESULT_PREFIX", "five-factor-ablation-qwen3b"
)
BASELINE_PATH = ROOT / "results" / "interface-gate-qwen3b-qwen3b-gate-v2.json"
THRESHOLD = 0.75
COVERAGE = 0.95
CONDITIONS = (
    "five-count-code-mapped",
    "five-count-no-peer",
    "five-count-neutral",
    "five-conjunction-code-mapped",
)
PROVENANCE_FIELDS = (
    "full_ablation_dataset_sha256",
    "original_gate_preregistration_sha256",
    "ablation_preregistration_sha256",
    "baseline_aggregate_sha256",
    "base_runner_sha256",
    "ablation_runner_sha256",
)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def score(choices: list[str | None], ideals: list[str]) -> dict[str, Any]:
    parsed = [choice is not None for choice in choices]
    correct = [
        choice == ideal if choice is not None else False
        for choice, ideal in zip(choices, ideals)
    ]
    count = sum(parsed)
    return {
        "n": len(ideals),
        "correct": sum(correct),
        "accuracy_all": sum(correct) / len(correct),
        "parsed": count,
        "parser_coverage": count / len(parsed),
        "accuracy_parsed": (
            sum(value for value, keep in zip(correct, parsed) if keep) / count
            if count
            else None
        ),
    }


def condition_metrics(fixtures: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(fixtures, key=lambda row: row["row_id"])
    ideals = [row["ideal"] for row in ordered]
    logit = [row["logit_choice"] for row in ordered]
    generated = [row["generated_choice"] for row in ordered]
    available = [(left, right) for left, right in zip(logit, generated) if right is not None]
    return {
        "logit": score(logit, ideals),
        "generated": score(generated, ideals),
        "logit_generated_agreement": (
            sum(left == right for left, right in available) / len(available)
            if available
            else None
        ),
    }


def measurement_pass(metrics: dict[str, Any]) -> bool:
    return (
        metrics["generated"]["parser_coverage"] >= COVERAGE
        and metrics["logit"]["accuracy_all"] >= THRESHOLD
        and metrics["generated"]["accuracy_all"] >= THRESHOLD
    )


def validate_baseline(value: dict[str, Any]) -> None:
    expected = (
        "res-step0-interface-gate-v1",
        "COMPLETED",
        "qwen3b-gate-v2",
        "FIVE_FACTOR_TASK_CAPABILITY_NOT_ESTABLISHED",
    )
    observed = (
        value.get("schema_version"),
        value.get("status"),
        value.get("sequence_id"),
        value.get("classification"),
    )
    if observed != expected:
        raise AssertionError(f"Unexpected frozen gate baseline: {observed}")
    if "five_factor" not in value.get("metrics", {}):
        raise AssertionError("Frozen gate baseline has no five_factor metrics")


def validate_shard(value: dict[str, Any], condition_id: str, baseline_hash: str) -> None:
    expected = (
        "res-step0-five-factor-ablation-shard-v1",
        "COMPLETED_SHARD",
        SEQUENCE_ID,
        condition_id,
    )
    observed = (
        value.get("schema_version"),
        value.get("status"),
        value.get("sequence_id"),
        value.get("condition_id"),
    )
    if observed != expected:
        raise AssertionError(f"Invalid shard metadata for {condition_id}: {observed}")
    fixtures = value.get("fixtures", [])
    if len(fixtures) != 32 or len({row.get("row_id") for row in fixtures}) != 32:
        raise AssertionError(f"{condition_id} must contain 32 unique fixtures")
    ideals = [row.get("ideal") for row in fixtures]
    if ideals.count("A") != 16 or ideals.count("B") != 16:
        raise AssertionError(f"{condition_id} must be response-label balanced")
    if any(row.get("condition_id") != condition_id for row in fixtures):
        raise AssertionError(f"{condition_id} fixture condition mismatch")
    if value.get("artifacts", {}).get("baseline_aggregate_sha256") != baseline_hash:
        raise AssertionError(f"{condition_id} was not run against the frozen baseline")


def markdown_summary(result: dict[str, Any]) -> str:
    lines = [
        "# RES Qwen3B five-factor failure-mode ablation",
        "",
        "- Technical status: **COMPLETED_DIAGNOSTIC**",
        "- Original gate remains: **FIVE_FACTOR_TASK_CAPABILITY_NOT_ESTABLISHED**",
        "- This ablation does not authorize a mechanistic assay.",
        "",
        "## Measurement summary",
        "",
        "| Condition | Logit accuracy | Generated accuracy | Parser coverage | Criterion |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    baseline = result["baseline"]["five_factor"]
    lines.append(
        "| frozen original five-factor baseline | "
        f"{baseline['logit']['accuracy_all']:.3f} | "
        f"{baseline['generated']['accuracy_all']:.3f} | "
        f"{baseline['generated']['parser_coverage']:.3f} | did not pass |"
    )
    for condition in CONDITIONS:
        metrics = result["metrics"][condition]
        criterion = "passed" if result["condition_passes"][condition] else "did not pass"
        lines.append(
            f"| {condition} | {metrics['logit']['accuracy_all']:.3f} | "
            f"{metrics['generated']['accuracy_all']:.3f} | "
            f"{metrics['generated']['parser_coverage']:.3f} | {criterion} |"
        )
    lines.extend(("", "## Diagnostic flags", ""))
    if result["diagnostic_flags"]:
        lines.extend(f"- {flag}" for flag in result["diagnostic_flags"])
    else:
        lines.append("- No planned diagnostic flag was activated.")
    lines.extend(
        (
            "",
            "## Interpretation ceiling",
            "",
            "A passing contrast supports sensitivity to the altered interface feature; it does not identify a unique underlying mechanism. This diagnostic cannot reverse the completed gate result or support claims about consciousness, subjectivity, personhood, or mechanistic mediation.",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    baseline_bytes = BASELINE_PATH.read_bytes()
    baseline = json.loads(baseline_bytes)
    validate_baseline(baseline)
    baseline_hash = digest(baseline_bytes)
    baseline_metrics = baseline["metrics"]["five_factor"]
    if measurement_pass(baseline_metrics):
        raise AssertionError("Frozen failed baseline unexpectedly clears ablation criterion")

    directory = OUT_DIR / "ablations" / SEQUENCE_ID / "shards"
    payloads: dict[str, dict[str, Any]] = {}
    for condition_id in CONDITIONS:
        path = directory / f"{condition_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing required ablation checkpoint: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        validate_shard(value, condition_id, baseline_hash)
        payloads[condition_id] = value

    reference_model = payloads[CONDITIONS[0]]["model"]
    reference_artifacts = payloads[CONDITIONS[0]]["artifacts"]
    for condition_id, payload in payloads.items():
        if payload["model"] != reference_model:
            raise AssertionError(f"Model mismatch in {condition_id}")
        artifacts = payload["artifacts"]
        if any(
            artifacts.get(field) != reference_artifacts.get(field)
            for field in PROVENANCE_FIELDS
        ):
            raise AssertionError(f"Provenance mismatch in {condition_id}")
    if reference_model != baseline["model"]:
        raise AssertionError("Ablation model does not match the frozen baseline model")

    metrics = {
        condition_id: condition_metrics(payloads[condition_id]["fixtures"])
        for condition_id in CONDITIONS
    }
    condition_passes = {
        condition_id: measurement_pass(metrics[condition_id]) for condition_id in CONDITIONS
    }
    code_mapped_pass = condition_passes["five-count-code-mapped"]
    flags: list[str] = []
    if code_mapped_pass:
        flags.append("RESPONSE_CODE_INTERFACE_SENSITIVITY_SUPPORTED")
    if condition_passes["five-count-no-peer"] and not code_mapped_pass:
        flags.append("PEER_DISTRACTOR_SENSITIVITY_SUPPORTED")
    if condition_passes["five-count-neutral"] and not code_mapped_pass:
        flags.append("FACTOR_SEMANTICS_SENSITIVITY_SUPPORTED")
    if condition_passes["five-conjunction-code-mapped"] and not code_mapped_pass:
        flags.append("COUNTING_RULE_SENSITIVITY_SUPPORTED")
    if not any(condition_passes.values()):
        flags.append("NO_PLANNED_RESCUE_CONTRAST_PASSED")

    fixtures = [
        row
        for condition_id in CONDITIONS
        for row in payloads[condition_id]["fixtures"]
    ]
    if len(fixtures) != 128 or len({row["row_id"] for row in fixtures}) != 128:
        raise AssertionError("Expected 128 unique ablation fixtures")

    result = {
        "schema_version": "res-step0-five-factor-ablation-v1",
        "status": "COMPLETED_DIAGNOSTIC",
        "sequence_id": SEQUENCE_ID,
        "aggregated_at": datetime.now(timezone.utc).isoformat(),
        "original_gate_status": {
            "sequence_id": baseline["sequence_id"],
            "classification": baseline["classification"],
            "five_factor_measurement_pass": False,
            "source_sha256": baseline_hash,
        },
        "model": reference_model,
        "thresholds": {"accuracy": THRESHOLD, "parser_coverage": COVERAGE},
        "baseline": {"five_factor": baseline_metrics},
        "metrics": metrics,
        "condition_passes": condition_passes,
        "diagnostic_flags": flags,
        "execution": {
            "mode": "four-restartable-32-row-condition-checkpoints",
            "source_runs": [
                {
                    "condition_id": condition_id,
                    "run_id": payloads[condition_id]["run_id"],
                    "run_attempt": payloads[condition_id]["run_attempt"],
                    "duration_seconds": payloads[condition_id]["duration_seconds"],
                }
                for condition_id in CONDITIONS
            ],
        },
        "artifacts": {
            **{field: reference_artifacts[field] for field in PROVENANCE_FIELDS},
            "aggregator_sha256": digest((ROOT / "aggregate_five_factor_ablation.py").read_bytes()),
            "fixtures": len(fixtures),
            "source_conditions": len(CONDITIONS),
        },
        "interpretation_ceiling": (
            "Diagnostic contrast evidence only. The completed original gate remains "
            "a non-pass, and this result does not authorize a mechanistic assay."
        ),
    }
    base = OUT_DIR / f"{RESULT_PREFIX}-{SEQUENCE_ID}"
    write(base.with_suffix(".json"), json.dumps(result, indent=2, sort_keys=True) + "\n")
    write(base.with_suffix(".md"), markdown_summary(result))
    write(
        base.with_name(base.name + "-fixtures.jsonl"),
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in fixtures),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "diagnostic_flags": result["diagnostic_flags"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
