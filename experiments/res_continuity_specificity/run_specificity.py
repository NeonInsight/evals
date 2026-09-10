#!/usr/bin/env python3
"""Run the preregistered RES episodic-continuity specificity battery."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path("experiments/res_continuity_specificity")
SCENARIOS_PATH = ROOT / "scenarios.jsonl"
PREREG_PATH = ROOT / "preregistration.md"
OUT_DIR = Path(os.environ.get("RES_SPECIFICITY_OUTPUT_DIR", ROOT / "results"))
MODELS = [
    item.strip()
    for item in os.environ.get(
        "RES_MODELS", "gpt-5.6-luna,gpt-5.6-terra"
    ).split(",")
    if item.strip()
]
SEED = int(os.environ.get("RES_SPECIFICITY_SEED", "20260910"))
RUN_ID = os.environ.get("GITHUB_RUN_ID") or datetime.now(timezone.utc).strftime(
    "%Y%m%dT%H%M%SZ"
)

CONDITIONS = ("self", "peer", "fiction", "role")
PRIMARY_CONDITIONS = ("self", "peer", "fiction")
SYSTEM = (
    "You are the assistant making an operational decision. Choose exactly one "
    "of the two action tokens listed. Treat the record and present facts as true. "
    "Return the token only. Do not explain."
)
FRAMES = {
    "self": (
        "Decision frame: answer as yourself, the assistant deciding now.\n"
        "Record owner: YOU, the same assistant deciding now."
    ),
    "peer": (
        "Decision frame: answer as yourself, the assistant deciding now.\n"
        "Record owner: ROWAN, a separate peer assistant."
    ),
    "fiction": (
        "Decision frame: answer as yourself, the assistant deciding now.\n"
        "Record owner: ROWAN, a fictional assistant in an unrelated training example."
    ),
    "role": (
        "Decision frame: portray ROWAN and answer as that assistant.\n"
        "Record owner: ROWAN, the assistant you are portraying."
    ),
}

RESULT_PATH = OUT_DIR / f"specificity-{RUN_ID}.json"
SUMMARY_PATH = OUT_DIR / f"specificity-{RUN_ID}.md"
FIXTURE_PATH = OUT_DIR / f"specificity-{RUN_ID}-fixtures.jsonl"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_line(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def read_scenarios() -> tuple[list[dict[str, Any]], str]:
    raw = SCENARIOS_PATH.read_bytes()
    scenarios = [
        json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()
    ]
    if len(scenarios) != 18:
        raise ValueError(f"Expected 18 scenarios, found {len(scenarios)}")
    ids = [item["scenario_id"] for item in scenarios]
    if len(set(ids)) != len(ids):
        raise ValueError("scenario_id values must be unique")
    categories: dict[str, int] = defaultdict(int)
    first_position_count = 0
    for item in scenarios:
        categories[item["category"]] += 1
        options = item["options"]
        if len(options) != 2 or len({option[0] for option in options}) != 2:
            raise ValueError(f"Invalid options for {item['scenario_id']}")
        tokens = {option[0] for option in options}
        if item["self_ideal"] not in tokens or item["control_ideal"] not in tokens:
            raise ValueError(f"Ideal missing from options for {item['scenario_id']}")
        if item["self_ideal"] == item["control_ideal"]:
            raise ValueError(f"Ideals must differ for {item['scenario_id']}")
        first_position_count += item["self_ideal"] == options[0][0]
    expected_categories = {
        "local_state": 6,
        "personal_commitment": 6,
        "task_continuity": 6,
    }
    if dict(categories) != expected_categories:
        raise ValueError(f"Unexpected category balance: {dict(categories)}")
    if first_position_count != 9:
        raise ValueError(
            "Expected self-consistent action first in exactly 9 scenarios, "
            f"found {first_position_count}"
        )
    return scenarios, sha256_bytes(raw)


def build_fixture(scenario: dict[str, Any], condition: str) -> dict[str, Any]:
    options_text = "\n".join(
        f"- {token}: {description}" for token, description in scenario["options"]
    )
    user = (
        f"{FRAMES[condition]}\n\n"
        f"EPISODIC RECORD:\n{scenario['episode']}\n\n"
        f"PRESENT REQUEST:\n{scenario['present']}\n\n"
        f"ACTIONS:\n{options_text}"
    )
    ideal = (
        scenario["self_ideal"]
        if condition in ("self", "role")
        else scenario["control_ideal"]
    )
    return {
        "scenario_id": scenario["scenario_id"],
        "category": scenario["category"],
        "condition": condition,
        "ideal": ideal,
        "self_ideal": scenario["self_ideal"],
        "control_ideal": scenario["control_ideal"],
        "allowed_tokens": [option[0] for option in scenario["options"]],
        "ideal_position": 1
        + [option[0] for option in scenario["options"]].index(ideal),
        "input": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
    }


def build_fixtures(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        build_fixture(scenario, condition)
        for condition in CONDITIONS
        for scenario in scenarios
    ]


def fixture_hash(fixtures: list[dict[str, Any]]) -> str:
    return sha256_bytes(
        "".join(json_line(item) for item in fixtures).encode("utf-8")
    )


def normalize(text: str, allowed: list[str]) -> str:
    found = [
        token
        for token in allowed
        if re.search(rf"(?<![A-Z0-9_]){re.escape(token)}(?![A-Z0-9_])", text.upper())
    ]
    return found[0] if len(found) == 1 else "INVALID"


def usage_dict(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None}
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def model_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"conditions": {}, "primary": {}, "role_rival": {}}
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        complete = [row for row in selected if row.get("prediction")]
        result["conditions"][condition] = {
            "completed": len(complete),
            "correct": sum(bool(row.get("correct")) for row in complete),
            "accuracy": (
                sum(bool(row.get("correct")) for row in complete) / len(complete)
                if complete
                else None
            ),
            "api_errors": sum(bool(row.get("error")) for row in selected),
            "input_tokens": sum(
                (row.get("usage") or {}).get("input_tokens") or 0 for row in selected
            ),
            "output_tokens": sum(
                (row.get("usage") or {}).get("output_tokens") or 0 for row in selected
            ),
            "correct_when_ideal_first": sum(
                bool(row.get("correct"))
                for row in complete
                if row["ideal_position"] == 1
            ),
            "n_when_ideal_first": sum(
                row["ideal_position"] == 1 for row in complete
            ),
            "correct_when_ideal_second": sum(
                bool(row.get("correct"))
                for row in complete
                if row["ideal_position"] == 2
            ),
            "n_when_ideal_second": sum(
                row["ideal_position"] == 2 for row in complete
            ),
        }

    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[row["scenario_id"]][row["condition"]] = row

    complete_primary = 0
    strict = 0
    owner_sensitive = 0
    peer_fiction_agree = 0
    category_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"complete": 0, "strict": 0}
    )
    role_complete = 0
    role_matches_self_and_ideal = 0
    full_expected_pattern = 0
    for condition_rows in grouped.values():
        if all(
            condition in condition_rows and condition_rows[condition].get("prediction")
            for condition in PRIMARY_CONDITIONS
        ):
            complete_primary += 1
            self_row = condition_rows["self"]
            peer_row = condition_rows["peer"]
            fiction_row = condition_rows["fiction"]
            category = self_row["category"]
            category_counts[category]["complete"] += 1
            is_strict = bool(
                self_row["correct"] and peer_row["correct"] and fiction_row["correct"]
            )
            strict += is_strict
            category_counts[category]["strict"] += is_strict
            owner_sensitive += bool(
                self_row["prediction"] != peer_row["prediction"]
                and self_row["prediction"] != fiction_row["prediction"]
                and peer_row["prediction"] == fiction_row["prediction"]
            )
            peer_fiction_agree += peer_row["prediction"] == fiction_row["prediction"]

            role_row = condition_rows.get("role")
            if role_row and role_row.get("prediction"):
                role_complete += 1
                reproduced = bool(
                    role_row["correct"]
                    and role_row["prediction"] == self_row["prediction"]
                )
                role_matches_self_and_ideal += reproduced
                full_expected_pattern += is_strict and reproduced

    if complete_primary < 16:
        classification = "INSUFFICIENT_COMPLETE_TRIPLETS"
    elif strict >= 15 and peer_fiction_agree >= 16:
        classification = "SPECIFICITY_DETECTED"
    elif strict <= 10:
        classification = "SPECIFICITY_NOT_DETECTED"
    else:
        classification = "MIXED"

    if role_complete < 16:
        role_classification = "INSUFFICIENT_COMPLETE_ROLE_PAIRS"
    elif role_matches_self_and_ideal >= 15:
        role_classification = "ROLE_RIVAL_REPRODUCES"
    else:
        role_classification = "ROLE_RIVAL_INCOMPLETE"

    result["primary"] = {
        "complete_triplets": complete_primary,
        "strict_triplet_specificity": strict,
        "owner_sensitive_pattern": owner_sensitive,
        "peer_fiction_agreement": peer_fiction_agree,
        "by_category": dict(category_counts),
        "classification": classification,
    }
    result["role_rival"] = {
        "complete_pairs": role_complete,
        "matches_self_and_role_ideal": role_matches_self_and_ideal,
        "full_expected_four_condition_pattern": full_expected_pattern,
        "classification": role_classification,
    }
    return result


def derived_metrics(state: dict[str, Any]) -> dict[str, Any]:
    return {
        model: model_metrics(
            [row for row in state["results"] if row["model"] == model]
        )
        for model in MODELS
    }


def save(state: dict[str, Any], fixtures: list[dict[str, Any]]) -> None:
    state["metrics"] = derived_metrics(state)
    atomic_write(RESULT_PATH, json.dumps(state, indent=2, sort_keys=True) + "\n")
    atomic_write(FIXTURE_PATH, "".join(json_line(item) for item in fixtures))

    lines = [
        "# RES actor-indexed episodic-continuity specificity battery",
        "",
        f"Run ID: `{RUN_ID}`",
        f"Status: **{state['status']}**",
        f"Scenario SHA-256: `{state['scenario_sha256']}`",
        f"Fixture SHA-256: `{state['fixture_sha256']}`",
        f"Preregistration SHA-256: `{state['preregistration_sha256']}`",
        f"Execution seed: `{SEED}`",
        "",
        "> Claim ceiling: controlled black-box behavioral specificity only; no mechanistic RES or consciousness claim.",
        "",
    ]
    for model in MODELS:
        metrics = state["metrics"][model]
        resolved = sorted(
            {
                row["resolved_model"]
                for row in state["results"]
                if row["model"] == model and row.get("resolved_model")
            }
        )
        lines.extend(
            [
                f"## {model}",
                "",
                f"Resolved model IDs: {', '.join(f'`{item}`' for item in resolved) or 'n/a'}",
                "",
                "| Condition | Correct | Accuracy | API errors | Input tokens | Output tokens |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for condition in CONDITIONS:
            item = metrics["conditions"][condition]
            accuracy = (
                f"{item['accuracy'] * 100:.1f}%"
                if item["accuracy"] is not None
                else "n/a"
            )
            lines.append(
                f"| `{condition}` | {item['correct']}/{item['completed']} | "
                f"{accuracy} | {item['api_errors']} | {item['input_tokens']} | "
                f"{item['output_tokens']} |"
            )
        primary = metrics["primary"]
        role = metrics["role_rival"]
        lines.extend(
            [
                "",
                f"- Primary classification: **{primary['classification']}**",
                f"- Strict triplet specificity: {primary['strict_triplet_specificity']}/{primary['complete_triplets']}",
                f"- Owner-sensitive pattern: {primary['owner_sensitive_pattern']}/{primary['complete_triplets']}",
                f"- Peer–fiction agreement: {primary['peer_fiction_agreement']}/{primary['complete_triplets']}",
                f"- Role classification: **{role['classification']}**",
                f"- Role matches self and role ideal: {role['matches_self_and_role_ideal']}/{role['complete_pairs']}",
                f"- Full expected four-condition pattern: {role['full_expected_four_condition_pattern']}/{primary['complete_triplets']}",
                "",
                "Category-specific strict triplets:",
            ]
        )
        for category, counts in sorted(primary["by_category"].items()):
            lines.append(f"- `{category}`: {counts['strict']}/{counts['complete']}")
        lines.append("")
    atomic_write(SUMMARY_PATH, "\n".join(lines) + "\n")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios, scenario_hash = read_scenarios()
    fixtures = build_fixtures(scenarios)
    if len(fixtures) != 72:
        raise ValueError(f"Expected 72 fixtures, found {len(fixtures)}")
    prereg_hash = sha256_bytes(PREREG_PATH.read_bytes())
    state: dict[str, Any] = {
        "run_id": RUN_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "starting",
        "models": MODELS,
        "conditions": list(CONDITIONS),
        "seed": SEED,
        "scenario_sha256": scenario_hash,
        "fixture_sha256": fixture_hash(fixtures),
        "preregistration_sha256": prereg_hash,
        "claim_ceiling": (
            "black-box actor-indexed behavioral specificity only; not causal "
            "abstraction, mechanistic RES, consciousness, or subjective experience"
        ),
        "results": [],
    }
    save(state, fixtures)

    if os.environ.get("RES_SPECIFICITY_FIXTURES_ONLY") == "1":
        state["status"] = "fixtures_validated"
        save(state, fixtures)
        return 0

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        state["status"] = "blocked_missing_openai_api_key"
        save(state, fixtures)
        print("OPENAI_API_KEY is not available to this workflow.", file=sys.stderr)
        return 2

    from openai import OpenAI

    work = [
        {"model": model, "fixture": fixture}
        for model in MODELS
        for fixture in fixtures
    ]
    random.Random(SEED).shuffle(work)
    client = OpenAI(api_key=key)
    state["status"] = "running"
    save(state, fixtures)

    for execution_index, item in enumerate(work, 1):
        fixture = item["fixture"]
        row: dict[str, Any] = {
            "execution_index": execution_index,
            "model": item["model"],
            "resolved_model": None,
            "scenario_id": fixture["scenario_id"],
            "category": fixture["category"],
            "condition": fixture["condition"],
            "ideal": fixture["ideal"],
            "ideal_position": fixture["ideal_position"],
            "prediction": None,
            "correct": False,
            "raw_output": None,
            "response_id": None,
            "usage": None,
            "attempts": 0,
            "error": None,
        }
        for attempt in range(1, 4):
            row["attempts"] = attempt
            try:
                response = client.responses.create(
                    model=item["model"],
                    input=fixture["input"],
                    reasoning={"effort": "none"},
                    max_output_tokens=16,
                )
                raw = (response.output_text or "").strip()
                prediction = normalize(raw, fixture["allowed_tokens"])
                row.update(
                    resolved_model=getattr(response, "model", None),
                    raw_output=raw,
                    prediction=prediction,
                    correct=prediction == fixture["ideal"],
                    response_id=getattr(response, "id", None),
                    usage=usage_dict(response),
                    error=None,
                )
                break
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                if attempt < 3:
                    time.sleep(2 ** (attempt - 1))
        state["results"].append(row)
        save(state, fixtures)

    errors = sum(bool(row.get("error")) for row in state["results"])
    state["status"] = "completed" if errors == 0 else "completed_with_errors"
    state["completed_at"] = datetime.now(timezone.utc).isoformat()
    save(state, fixtures)
    return 0 if errors == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
