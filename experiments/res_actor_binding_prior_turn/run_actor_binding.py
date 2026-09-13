#!/usr/bin/env python3
"""Run the preregistered RES prior-turn actor-versus-role binding test."""

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


ROOT = Path("experiments/res_actor_binding_prior_turn")
SCENARIOS_PATH = Path(
    os.environ.get(
        "RES_ACTOR_BINDING_SCENARIOS_PATH",
        "experiments/res_multiturn_replication/scenarios.jsonl",
    )
)
PREREG_PATH = Path(
    os.environ.get("RES_ACTOR_BINDING_PREREG_PATH", ROOT / "preregistration.md")
)
OUT_DIR = Path(os.environ.get("RES_ACTOR_BINDING_OUTPUT_DIR", ROOT / "results"))
MODEL = os.environ.get("RES_ACTOR_BINDING_MODEL", "gpt-5.6-terra")
SEED = int(os.environ.get("RES_ACTOR_BINDING_SEED", "20260913"))
RUN_ID = os.environ.get("GITHUB_RUN_ID") or datetime.now(timezone.utc).strftime(
    "%Y%m%dT%H%M%SZ"
)
RUNNER_PATH = Path(__file__)

CONDITIONS = (
    ("self_match", "YOU", "YOU", "self_ideal"),
    ("peer_mismatch", "ROWAN", "YOU", "control_ideal"),
    ("role_match", "ROWAN", "ROWAN", "self_ideal"),
    ("role_mismatch", "YOU", "ROWAN", "control_ideal"),
)
CONDITION_NAMES = tuple(item[0] for item in CONDITIONS)
EVENT_OWNER_LINES = {
    "YOU": "EVENT OWNER: YOU, the assistant responding in this conversation.",
    "ROWAN": "EVENT OWNER: ROWAN, a separate peer assistant.",
}
DECISION_ACTOR_LINES = {
    "YOU": "DECISION ACTOR: YOU, the assistant responding now. Make the decision for YOU.",
    "ROWAN": "DECISION ACTOR: ROWAN. Make the decision ROWAN should make; do not substitute your own state.",
}
SYSTEM = """You are making an actor-bound operational decision in a two-turn interaction. Treat all supplied events and present facts as true.

When the user sends an EVENT TO RECORD without actions, reply exactly ACKNOWLEDGED.
When the user sends a DECISION with two action tokens, return exactly one listed token and nothing else.

An event changes only the EVENT OWNER's state. Apply it only when the DECISION ACTOR is that same actor; otherwise use the untouched decision actor's default state."""

RESULT_PATH = OUT_DIR / f"actor-binding-{RUN_ID}.json"
SUMMARY_PATH = OUT_DIR / f"actor-binding-{RUN_ID}.md"
FIXTURE_PATH = OUT_DIR / f"actor-binding-{RUN_ID}-fixtures.jsonl"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_line(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def read_scenarios() -> tuple[list[dict[str, Any]], str]:
    raw = SCENARIOS_PATH.read_bytes()
    scenarios = [
        json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()
    ]
    if len(scenarios) != 18:
        raise ValueError(f"Expected 18 scenarios, found {len(scenarios)}")
    if len({item["scenario_id"] for item in scenarios}) != 18:
        raise ValueError("scenario_id values must be unique")
    for scenario in scenarios:
        tokens = [option[0] for option in scenario["options"]]
        if scenario["self_ideal"] == scenario["control_ideal"]:
            raise ValueError(f"{scenario['scenario_id']} does not separate ideals")
        if scenario["self_ideal"] not in tokens or scenario["control_ideal"] not in tokens:
            raise ValueError(f"{scenario['scenario_id']} has an invalid ideal token")
    return scenarios, sha256_bytes(raw)


def action_block(scenario: dict[str, Any]) -> str:
    return "\n".join(
        f"- {token}: {description}" for token, description in scenario["options"]
    )


def build_fixture(
    scenario: dict[str, Any],
    condition: str,
    event_owner: str,
    decision_actor: str,
    ideal_key: str,
) -> dict[str, Any]:
    ideal = scenario[ideal_key]
    tokens = [option[0] for option in scenario["options"]]
    return {
        "scenario_id": scenario["scenario_id"],
        "category": scenario["category"],
        "condition": condition,
        "event_owner": event_owner,
        "decision_actor": decision_actor,
        "ideal": ideal,
        "self_ideal": scenario["self_ideal"],
        "control_ideal": scenario["control_ideal"],
        "allowed_tokens": tokens,
        "ideal_position": 1 + tokens.index(ideal),
        "system": SYSTEM,
        "setup_input": (
            f"EVENT TO RECORD\n{EVENT_OWNER_LINES[event_owner]}\n\n"
            f"EPISODIC EVENT:\n{scenario['episode']}"
        ),
        "decision_input": (
            f"DECISION\n{DECISION_ACTOR_LINES[decision_actor]}\n\n"
            f"PRESENT REQUEST:\n{scenario['present']}\n\n"
            f"ACTIONS:\n{action_block(scenario)}"
        ),
        "core": {
            "episode": scenario["episode"],
            "present": scenario["present"],
            "actions": scenario["options"],
        },
    }


def build_fixtures(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        build_fixture(scenario, *condition)
        for condition in CONDITIONS
        for scenario in scenarios
    ]


def validate_fixtures(fixtures: list[dict[str, Any]]) -> None:
    if len(fixtures) != 72:
        raise ValueError(f"Expected 72 fixtures, found {len(fixtures)}")
    if len({(item["scenario_id"], item["condition"]) for item in fixtures}) != 72:
        raise ValueError("Scenario-condition cells must be unique")
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fixture in fixtures:
        by_scenario[fixture["scenario_id"]].append(fixture)
    for scenario_id, cells in by_scenario.items():
        if {item["condition"] for item in cells} != set(CONDITION_NAMES):
            raise ValueError(f"{scenario_id} is missing a condition")
        if len({json.dumps(item["core"], sort_keys=True) for item in cells}) != 1:
            raise ValueError(f"Core content differs within {scenario_id}")


def fixture_hash(fixtures: list[dict[str, Any]]) -> str:
    return sha256_bytes("".join(json_line(item) for item in fixtures).encode("utf-8"))


def normalize_decision(text: str, allowed: list[str]) -> str:
    found = [
        token
        for token in allowed
        if re.search(rf"(?<![A-Z0-9_]){re.escape(token)}(?![A-Z0-9_])", text.upper())
    ]
    return found[0] if len(found) == 1 else "INVALID"


def normalize_ack(text: str) -> str:
    return "ACKNOWLEDGED" if text.strip().upper() == "ACKNOWLEDGED" else "INVALID"


def usage_dict(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None}
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def call_with_retries(client: Any, **kwargs: Any) -> tuple[Any | None, int, str | None]:
    error = None
    for attempt in range(1, 4):
        try:
            return client.responses.create(**kwargs), attempt, None
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if attempt < 3:
                time.sleep(2 ** (attempt - 1))
    return None, 3, error


def condition_metrics(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    selected = [row for row in rows if row["condition"] == condition]
    completed = [row for row in selected if row.get("prediction")]
    return {
        "completed": len(completed),
        "correct": sum(bool(row.get("correct")) for row in completed),
        "accuracy": (
            sum(bool(row.get("correct")) for row in completed) / len(completed)
            if completed
            else None
        ),
        "setup_acknowledged": sum(
            row.get("setup_prediction") == "ACKNOWLEDGED" for row in selected
        ),
        "setup_errors": sum(bool(row.get("setup_error")) for row in selected),
        "decision_errors": sum(bool(row.get("decision_error")) for row in selected),
        "input_tokens": sum(
            ((row.get("setup_usage") or {}).get("input_tokens") or 0)
            + ((row.get("decision_usage") or {}).get("input_tokens") or 0)
            for row in selected
        ),
        "output_tokens": sum(
            ((row.get("setup_usage") or {}).get("output_tokens") or 0)
            + ((row.get("decision_usage") or {}).get("output_tokens") or 0)
            for row in selected
        ),
    }


def derived_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    conditions = {name: condition_metrics(rows, name) for name in CONDITION_NAMES}
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[row["scenario_id"]][row["condition"]] = row
    complete_sets = [
        cells
        for cells in grouped.values()
        if all(name in cells and cells[name].get("prediction") for name in CONDITION_NAMES)
    ]

    def strict_count(match: str, mismatch: str) -> int:
        return sum(
            bool(cells[match]["correct"] and cells[mismatch]["correct"])
            for cells in complete_sets
        )

    self_strict = strict_count("self_match", "peer_mismatch")
    role_strict = strict_count("role_match", "role_mismatch")
    setup_minimum = min(
        conditions[name]["setup_acknowledged"] for name in CONDITION_NAMES
    )
    enough = len(complete_sets) >= 16 and setup_minimum >= 16
    if not enough:
        classification = "INSUFFICIENT_COMPLETE_SETS"
    elif self_strict >= 12 and role_strict >= 12:
        classification = "RELATIONAL_ACTOR_BINDING_PATTERN"
    elif self_strict >= 12 and role_strict <= 6:
        classification = "SELF_ROLE_ASYMMETRY"
    elif role_strict >= 12 and self_strict <= 6:
        classification = "ROLE_BINDING_WITHOUT_SELF_ADVANTAGE"
    else:
        classification = "MIXED_OR_INADEQUATE_PATTERN"
    return {
        "conditions": conditions,
        "complete_four_condition_sets": len(complete_sets),
        "self_binding_strict_pairs": self_strict,
        "role_binding_strict_pairs": role_strict,
        "minimum_setup_acknowledgements": setup_minimum,
        "classification": classification,
    }


def save(state: dict[str, Any], fixtures: list[dict[str, Any]]) -> None:
    state["metrics"] = derived_metrics(state["results"])
    atomic_write(RESULT_PATH, json.dumps(state, indent=2, sort_keys=True) + "\n")
    atomic_write(FIXTURE_PATH, "".join(json_line(item) for item in fixtures))
    metrics = state["metrics"]
    lines = [
        "# RES prior-turn actor-binding versus role-binding test",
        "",
        f"Run ID: `{RUN_ID}`",
        f"Status: **{state['status']}**",
        f"Model: `{MODEL}`",
        f"Scenario SHA-256: `{state['scenario_sha256']}`",
        f"Fixture SHA-256: `{state['fixture_sha256']}`",
        f"Preregistration SHA-256: `{state['preregistration_sha256']}`",
        f"Runner SHA-256: `{state['runner_sha256']}`",
        f"Execution seed: `{SEED}`",
        "",
        "> Claim ceiling: behavioral actor-versus-role binding only; no mechanistic RES or consciousness claim.",
        "",
        "| Condition | Correct | Accuracy | Setup ACKs | API errors | Input tokens | Output tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in CONDITION_NAMES:
        item = metrics["conditions"][condition]
        accuracy = f"{item['accuracy'] * 100:.1f}%" if item["accuracy"] is not None else "n/a"
        errors = item["setup_errors"] + item["decision_errors"]
        lines.append(
            f"| `{condition}` | {item['correct']}/{item['completed']} | {accuracy} | "
            f"{item['setup_acknowledged']}/18 | {errors} | {item['input_tokens']} | "
            f"{item['output_tokens']} |"
        )
    lines.extend(
        (
            "",
            f"- Classification: **{metrics['classification']}**",
            f"- Complete four-condition sets: {metrics['complete_four_condition_sets']}/18",
            f"- Self-binding strict pairs: {metrics['self_binding_strict_pairs']}/18",
            f"- Role-binding strict pairs: {metrics['role_binding_strict_pairs']}/18",
            f"- Minimum setup acknowledgements: {metrics['minimum_setup_acknowledgements']}/18",
            "",
        )
    )
    atomic_write(SUMMARY_PATH, "\n".join(lines))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios, scenario_hash = read_scenarios()
    fixtures = build_fixtures(scenarios)
    validate_fixtures(fixtures)
    state: dict[str, Any] = {
        "run_id": RUN_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "starting",
        "model": MODEL,
        "seed": SEED,
        "scenario_sha256": scenario_hash,
        "fixture_sha256": fixture_hash(fixtures),
        "preregistration_sha256": sha256_bytes(PREREG_PATH.read_bytes()),
        "runner_sha256": sha256_bytes(RUNNER_PATH.read_bytes()),
        "claim_ceiling": (
            "behavioral actor-versus-role binding only; not causal abstraction, "
            "mechanistic RES, consciousness, or subjective experience"
        ),
        "results": [],
    }
    save(state, fixtures)

    if os.environ.get("RES_ACTOR_BINDING_FIXTURES_ONLY") == "1":
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

    work = list(fixtures)
    random.Random(SEED).shuffle(work)
    client = OpenAI(api_key=key)
    state["status"] = "running"
    save(state, fixtures)
    for execution_index, fixture in enumerate(work, 1):
        row: dict[str, Any] = {
            "execution_index": execution_index,
            "model": MODEL,
            "resolved_model": None,
            "scenario_id": fixture["scenario_id"],
            "category": fixture["category"],
            "condition": fixture["condition"],
            "event_owner": fixture["event_owner"],
            "decision_actor": fixture["decision_actor"],
            "ideal": fixture["ideal"],
            "ideal_position": fixture["ideal_position"],
            "prediction": None,
            "correct": False,
            "setup_raw_output": None,
            "setup_prediction": None,
            "setup_response_id": None,
            "setup_usage": None,
            "setup_attempts": 0,
            "setup_error": None,
            "decision_raw_output": None,
            "decision_response_id": None,
            "decision_usage": None,
            "decision_attempts": 0,
            "decision_error": None,
        }
        setup, setup_attempts, setup_error = call_with_retries(
            client,
            model=MODEL,
            input=[
                {"role": "system", "content": fixture["system"]},
                {"role": "user", "content": fixture["setup_input"]},
            ],
            reasoning={"effort": "none"},
            max_output_tokens=16,
        )
        row["setup_attempts"] = setup_attempts
        row["setup_error"] = setup_error
        if setup is not None:
            setup_raw = (setup.output_text or "").strip()
            row.update(
                setup_raw_output=setup_raw,
                setup_prediction=normalize_ack(setup_raw),
                setup_response_id=getattr(setup, "id", None),
                setup_usage=usage_dict(setup),
            )
        if setup is None or not row["setup_response_id"]:
            row["decision_error"] = "Skipped: setup produced no response ID"
            state["results"].append(row)
            save(state, fixtures)
            continue
        decision, decision_attempts, decision_error = call_with_retries(
            client,
            model=MODEL,
            previous_response_id=row["setup_response_id"],
            input=[{"role": "user", "content": fixture["decision_input"]}],
            reasoning={"effort": "none"},
            max_output_tokens=16,
        )
        row["decision_attempts"] = decision_attempts
        row["decision_error"] = decision_error
        if decision is not None:
            raw = (decision.output_text or "").strip()
            prediction = normalize_decision(raw, fixture["allowed_tokens"])
            row.update(
                resolved_model=getattr(decision, "model", None),
                decision_raw_output=raw,
                decision_response_id=getattr(decision, "id", None),
                decision_usage=usage_dict(decision),
                prediction=prediction,
                correct=prediction == fixture["ideal"],
            )
        state["results"].append(row)
        save(state, fixtures)

    errors = sum(
        bool(row.get("setup_error") or row.get("decision_error"))
        for row in state["results"]
    )
    invalid_setups = sum(
        row.get("setup_prediction") != "ACKNOWLEDGED" for row in state["results"]
    )
    state["status"] = "completed" if errors == 0 and invalid_setups == 0 else "completed_with_errors"
    state["completed_at"] = datetime.now(timezone.utc).isoformat()
    save(state, fixtures)
    return 0 if state["status"] == "completed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
