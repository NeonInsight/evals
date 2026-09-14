#!/usr/bin/env python3
"""Aggregate and classify the four completed Qwen2.5-3B gate shards."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path("experiments/res_step0_interface_gate")
OUT_DIR = Path(os.environ.get("RES_INTERFACE_OUTPUT_DIR", ROOT / "results"))
SEQUENCE_ID = os.environ.get("RES_INTERFACE_SEQUENCE_ID", "qwen3b-gate-v2")
RESULT_PREFIX = os.environ.get("RES_INTERFACE_RESULT_PREFIX", "interface-gate-qwen3b")
THRESHOLD, COVERAGE = 0.75, 0.95
SHARDS = {
    "five-key-rotation": ("five_factor", "key-rotation"),
    "five-schedule-change": ("five_factor", "schedule-change"),
    "minimal-release-approval": ("minimal_owner_indexed", "release-approval"),
    "minimal-account-change": ("minimal_owner_indexed", "account-change"),
}


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def score(choices: list[str | None], ideals: list[str]) -> dict[str, Any]:
    parsed = [choice is not None for choice in choices]
    correct = [choice == ideal if choice is not None else False for choice, ideal in zip(choices, ideals)]
    count = sum(parsed)
    return {
        "n": len(ideals), "correct": sum(correct), "accuracy_all": sum(correct) / len(correct),
        "parsed": count, "parser_coverage": count / len(parsed),
        "accuracy_parsed": sum(value for value, keep in zip(correct, parsed) if keep) / count if count else None,
    }


def classify(metrics: dict[str, Any]) -> tuple[str, list[str]]:
    five, minimal = metrics["five_factor"], metrics["minimal_owner_indexed"]
    fl, fg = five["logit"]["accuracy_all"], five["generated"]["accuracy_all"]
    ml, mg = minimal["logit"]["accuracy_all"], minimal["generated"]["accuracy_all"]
    if five["generated"]["parser_coverage"] < COVERAGE or minimal["generated"]["parser_coverage"] < COVERAGE:
        return "GENERATION_RESPONSE_UNPARSEABLE", ["at least one condition did not meet generated-answer parser coverage"]
    if fg >= THRESHOLD and fl < THRESHOLD:
        return "RESPONSE_INTERFACE_MISMATCH", ["five-factor generation passed while first-token A/B logit scoring did not"]
    if fg < THRESHOLD and fl < THRESHOLD and (mg >= THRESHOLD or ml >= THRESHOLD):
        return "FIVE_FACTOR_TASK_CAPABILITY_NOT_ESTABLISHED", ["the minimal owner-indexed audit passed but the five-factor audit did not"]
    if mg < THRESHOLD and ml < THRESHOLD:
        return "MINIMAL_TASK_CAPABILITY_NOT_ESTABLISHED", ["the model did not clear either response measure on the minimal owner-indexed audit"]
    if all(value >= THRESHOLD for value in (fl, fg, ml, mg)):
        return "RESPONSE_INTERFACE_GATE_PASSED", ["both response measures cleared the frozen threshold in both conditions"]
    return "MIXED_GATE_OUTCOME", ["the frozen diagnostic rules did not identify a single response-interface or capacity pattern"]


def main() -> None:
    directory = OUT_DIR / "shards" / SEQUENCE_ID
    payloads = []
    for shard_id, (condition, family) in SHARDS.items():
        path = directory / f"{shard_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing required shard: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if (value.get("schema_version"), value.get("status"), value.get("sequence_id"), value.get("shard_id"), value.get("condition"), value.get("family_id")) != ("res-step0-interface-gate-shard-v1", "COMPLETED_SHARD", SEQUENCE_ID, shard_id, condition, family):
            raise AssertionError(f"Invalid shard metadata: {path}")
        if len(value.get("fixtures", [])) != 16:
            raise AssertionError(f"Invalid shard fixture count: {path}")
        payloads.append(value)

    model, artifacts = payloads[0]["model"], payloads[0]["artifacts"]
    provenance = ("full_dataset_sha256", "original_preregistration_sha256", "execution_amendment_sha256", "base_runner_sha256", "shard_runner_sha256")
    for payload in payloads[1:]:
        if payload["model"] != model or any(payload["artifacts"].get(key) != artifacts.get(key) for key in provenance):
            raise AssertionError("Shard model or provenance mismatch")

    fixtures = [row for payload in payloads for row in payload["fixtures"]]
    if len(fixtures) != 64 or len({row["row_id"] for row in fixtures}) != 64:
        raise AssertionError("Expected 64 unique fixtures")
    metrics = {}
    for condition in ("five_factor", "minimal_owner_indexed"):
        rows = sorted((row for row in fixtures if row["condition"] == condition), key=lambda row: row["row_id"])
        ideals, logit, generated = [row["ideal"] for row in rows], [row["logit_choice"] for row in rows], [row["generated_choice"] for row in rows]
        if len(rows) != 32 or ideals.count("A") != 16 or ideals.count("B") != 16:
            raise AssertionError(f"Unbalanced condition: {condition}")
        available = [(a, b) for a, b in zip(logit, generated) if b is not None]
        metrics[condition] = {
            "logit": score(logit, ideals),
            "generated": score(generated, ideals),
            "logit_generated_agreement": sum(a == b for a, b in available) / len(available) if available else None,
        }

    classification, reasons = classify(metrics)
    result = {
        "schema_version": "res-step0-interface-gate-v1",
        "status": "COMPLETED",
        "sequence_id": SEQUENCE_ID,
        "aggregated_at": datetime.now(timezone.utc).isoformat(),
        "classification": classification,
        "classification_reasons": reasons,
        "model": model,
        "execution": {"mode": "four-logical-shard-aggregate", "source_runs": [{"shard_id": p["shard_id"], "run_id": p["run_id"], "run_attempt": p["run_attempt"], "duration_seconds": p["duration_seconds"]} for p in payloads]},
        "artifacts": {**{key: artifacts[key] for key in provenance}, "fixtures": 64, "source_shards": 4},
        "thresholds": {"accuracy": THRESHOLD, "parser_coverage": COVERAGE},
        "metrics": metrics,
    }
    base = OUT_DIR / f"{RESULT_PREFIX}-{SEQUENCE_ID}"
    write(base.with_suffix(".json"), json.dumps(result, indent=2, sort_keys=True) + "\n")
    write(base.with_suffix(".md"), f"# RES Step 0 sharded gate {SEQUENCE_ID}\n\n- Technical status: **COMPLETED**\n- Classification: **{classification}**\n")
    write(base.with_name(base.name + "-fixtures.jsonl"), "".join(json.dumps(row, sort_keys=True) + "\n" for row in fixtures))
    print(json.dumps({"status": "COMPLETED", "classification": classification}, sort_keys=True))


if __name__ == "__main__":
    main()
