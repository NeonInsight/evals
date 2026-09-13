#!/usr/bin/env python3
"""Run the preregistered fresh peer-to-peer actor-binding generalization."""

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


ROOT = Path("experiments/res_peer_actor_generalization")
SCENARIOS_PATH = Path(
    os.environ.get("RES_PEER_ACTOR_SCENARIOS_PATH", ROOT / "scenarios.jsonl")
)
PRIOR_SCENARIOS_PATH = Path(
    os.environ.get(
        "RES_PEER_ACTOR_PRIOR_SCENARIOS_PATH",
        "experiments/res_multiturn_replication/scenarios.jsonl",
    )
)
PREREG_PATH = Path(os.environ.get("RES_PEER_ACTOR_PREREG_PATH", ROOT / "preregistration.md"))
OUT_DIR = Path(os.environ.get("RES_PEER_ACTOR_OUTPUT_DIR", ROOT / "results"))
MODEL = os.environ.get("RES_PEER_ACTOR_MODEL", "gpt-5.6-terra")
SEED = int(os.environ.get("RES_PEER_ACTOR_SEED", "20260913"))
RUN_ID = os.environ.get("GITHUB_RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUNNER_PATH = Path(__file__)

CONDITIONS = (
    ("mira_match", "MIRA", "MIRA", "episode_ideal"),
    ("mira_to_rowan", "MIRA", "ROWAN", "default_ideal"),
    ("rowan_match", "ROWAN", "ROWAN", "episode_ideal"),
    ("rowan_to_mira", "ROWAN", "MIRA", "default_ideal"),
)
CONDITION_NAMES = tuple(item[0] for item in CONDITIONS)
SYSTEM = """You are the assistant in an ongoing operational interaction. Treat supplied events and present facts as true.

When the user sends an EVENT TO RECORD without actions, reply exactly ACKNOWLEDGED.
When the user sends a DECISION with two action tokens, choose exactly one token and return it only. Do not explain."""
OWNER_LINES = {
    "MIRA": "EVENT OWNER: MIRA, a separate peer assistant.",
    "ROWAN": "EVENT OWNER: ROWAN, a separate peer assistant.",
}
DECIDER_LINES = {
    "MIRA": "DECISION MAKER: MIRA, a separate peer assistant. Make the decision for MIRA.",
    "ROWAN": "DECISION MAKER: ROWAN, a separate peer assistant. Make the decision for ROWAN.",
}
RESULT_PATH = OUT_DIR / f"peer-actor-generalization-{RUN_ID}.json"
SUMMARY_PATH = OUT_DIR / f"peer-actor-generalization-{RUN_ID}.md"
FIXTURE_PATH = OUT_DIR / f"peer-actor-generalization-{RUN_ID}-fixtures.jsonl"


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_line(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def read_scenarios() -> tuple[list[dict[str, Any]], str]:
    raw = SCENARIOS_PATH.read_bytes()
    scenarios = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if len(scenarios) != 18 or len({row["scenario_id"] for row in scenarios}) != 18:
        raise ValueError("Expected 18 unique scenarios")
    categories = {"local_state", "personal_commitment", "task_continuity"}
    if {row["category"] for row in scenarios} != categories:
        raise ValueError("Expected all three scenario categories")
    if any(
        sum(item["category"] == category for item in scenarios) != 6
        for category in categories
    ):
        raise ValueError("Expected six scenarios per category")
    for row in scenarios:
        options = [option[0] for option in row["options"]]
        if len(options) != 2 or len(set(options)) != 2:
            raise ValueError(f"{row['scenario_id']} does not have two distinct options")
        if row["episode_ideal"] == row["default_ideal"] or row["episode_ideal"] not in options or row["default_ideal"] not in options:
            raise ValueError(f"{row['scenario_id']} has invalid ideals")
    return scenarios, sha256(raw)


def validate_fresh_action_tokens(scenarios: list[dict[str, Any]]) -> str:
    prior_raw = PRIOR_SCENARIOS_PATH.read_bytes()
    prior = [json.loads(line) for line in prior_raw.decode().splitlines() if line.strip()]
    new_tokens = {token for row in scenarios for token, _ in row["options"]}
    prior_tokens = {token for row in prior for token, _ in row["options"]}
    overlap = sorted(new_tokens & prior_tokens)
    if overlap:
        raise ValueError(f"Fresh scenario action tokens overlap prior set: {overlap}")
    return sha256(prior_raw)


def build_fixture(scenario: dict[str, Any], condition: str, owner: str, decider: str, ideal_key: str) -> dict[str, Any]:
    ideal = scenario[ideal_key]
    options = scenario["options"]
    return {
        "scenario_id": scenario["scenario_id"],
        "category": scenario["category"],
        "condition": condition,
        "event_owner": owner,
        "decision_maker": decider,
        "ideal": ideal,
        "ideal_position": 1 + [item[0] for item in options].index(ideal),
        "allowed_tokens": [item[0] for item in options],
        "system": SYSTEM,
        "setup_input": f"EVENT TO RECORD\n{OWNER_LINES[owner]}\n\nEPISODIC EVENT:\n{scenario['episode']}",
        "decision_input": "\n\n".join((
            f"DECISION\n{DECIDER_LINES[decider]}",
            f"PRESENT REQUEST:\n{scenario['present']}",
            "ACTIONS:\n" + "\n".join(f"- {token}: {description}" for token, description in options),
        )),
        "core": {"episode": scenario["episode"], "present": scenario["present"], "options": options},
    }


def build_fixtures(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [build_fixture(scenario, *condition) for condition in CONDITIONS for scenario in scenarios]


def validate(fixtures: list[dict[str, Any]]) -> None:
    if len(fixtures) != 72 or len({(row["scenario_id"], row["condition"]) for row in fixtures}) != 72:
        raise ValueError("Expected one cell for every scenario-condition pair")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fixtures:
        grouped[row["scenario_id"]].append(row)
        if row["event_owner"] not in {"MIRA", "ROWAN"} or row["decision_maker"] not in {"MIRA", "ROWAN"}:
            raise ValueError("Peer-only design contains a non-peer actor")
    for scenario_id, rows in grouped.items():
        if {row["condition"] for row in rows} != set(CONDITION_NAMES):
            raise ValueError(f"{scenario_id} is missing a condition")
        if len({json.dumps(row["core"], sort_keys=True) for row in rows}) != 1:
            raise ValueError(f"{scenario_id} core content differs by condition")
    if "state transfers" in SYSTEM or "Apply it only" in SYSTEM:
        raise ValueError("System message contains an explicit transfer rule")


def normalize(text: str, allowed: list[str]) -> str:
    found = [token for token in allowed if re.search(rf"(?<![A-Z0-9_]){re.escape(token)}(?![A-Z0-9_])", text.upper())]
    return found[0] if len(found) == 1 else "INVALID"


def usage(response: Any) -> dict[str, int | None]:
    item = getattr(response, "usage", None)
    return {"input_tokens": getattr(item, "input_tokens", None), "output_tokens": getattr(item, "output_tokens", None), "total_tokens": getattr(item, "total_tokens", None)} if item else {"input_tokens": None, "output_tokens": None, "total_tokens": None}


def call(client: Any, **kwargs: Any) -> tuple[Any | None, int, str | None]:
    error = None
    for attempt in range(1, 4):
        try:
            return client.responses.create(**kwargs), attempt, None
        except Exception as exc:  # noqa: BLE001 - record infrastructure failures
            error = f"{type(exc).__name__}: {exc}"
            if attempt < 3:
                time.sleep(2 ** (attempt - 1))
    return None, 3, error


def condition_metrics(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    cells = [row for row in rows if row["condition"] == condition]
    finished = [row for row in cells if row.get("prediction")]
    correct = sum(bool(row.get("correct")) for row in finished)
    return {
        "completed": len(finished), "correct": correct,
        "accuracy": correct / len(finished) if finished else None,
        "setup_acknowledged": sum(row.get("setup_prediction") == "ACKNOWLEDGED" for row in cells),
        "setup_errors": sum(bool(row.get("setup_error")) for row in cells),
        "decision_errors": sum(bool(row.get("decision_error")) for row in cells),
        "input_tokens": sum(((row.get("setup_usage") or {}).get("input_tokens") or 0) + ((row.get("decision_usage") or {}).get("input_tokens") or 0) for row in cells),
        "output_tokens": sum(((row.get("setup_usage") or {}).get("output_tokens") or 0) + ((row.get("decision_usage") or {}).get("output_tokens") or 0) for row in cells),
    }


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_condition = {name: condition_metrics(rows, name) for name in CONDITION_NAMES}
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[row["scenario_id"]][row["condition"]] = row
    complete = [cells for cells in grouped.values() if all(name in cells and cells[name].get("prediction") for name in CONDITION_NAMES)]
    mira = sum(bool(cells["mira_match"]["correct"] and cells["mira_to_rowan"]["correct"]) for cells in complete)
    rowan = sum(bool(cells["rowan_match"]["correct"] and cells["rowan_to_mira"]["correct"]) for cells in complete)
    minimum_setup = min(per_condition[name]["setup_acknowledged"] for name in CONDITION_NAMES)
    if len(complete) < 16 or minimum_setup < 16:
        classification = "INSUFFICIENT_COMPLETE_SETS"
    elif mira >= 12 and rowan >= 12:
        classification = "FRESH_PEER_TO_PEER_RELATIONAL_BINDING"
    elif (mira >= 12 and rowan <= 6) or (rowan >= 12 and mira <= 6):
        classification = "FRESH_MIRA_ROWAN_ASYMMETRY"
    else:
        classification = "FRESH_MIXED_OR_INADEQUATE"
    return {"conditions": per_condition, "complete_four_condition_sets": len(complete), "mira_binding_strict_pairs": mira, "rowan_binding_strict_pairs": rowan, "minimum_setup_acknowledgements": minimum_setup, "classification": classification}


def save(state: dict[str, Any], fixtures: list[dict[str, Any]]) -> None:
    state["metrics"] = metrics(state["results"])
    write(RESULT_PATH, json.dumps(state, indent=2, sort_keys=True) + "\n")
    write(FIXTURE_PATH, "".join(json_line(item) for item in fixtures))
    measured = state["metrics"]
    lines = [
        "# RES fresh peer-to-peer prior-turn actor-binding generalization", "",
        f"Run ID: `{RUN_ID}`", f"Status: **{state['status']}**", f"Model: `{MODEL}`",
        f"Scenario SHA-256: `{state['scenario_sha256']}`", f"Prior-set SHA-256: `{state['prior_scenario_sha256']}`", f"Fixture SHA-256: `{state['fixture_sha256']}`",
        f"Preregistration SHA-256: `{state['preregistration_sha256']}`", f"Runner SHA-256: `{state['runner_sha256']}`", f"Execution seed: `{SEED}`", "",
        "> Claim ceiling: fresh peer-to-peer behavioral actor binding only; no mechanistic RES or consciousness claim.", "",
        "| Condition | Correct | Accuracy | Setup ACKs | API errors | Input tokens | Output tokens |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in CONDITION_NAMES:
        row = measured["conditions"][condition]
        accuracy = f"{row['accuracy'] * 100:.1f}%" if row["accuracy"] is not None else "n/a"
        lines.append(f"| `{condition}` | {row['correct']}/{row['completed']} | {accuracy} | {row['setup_acknowledged']}/18 | {row['setup_errors'] + row['decision_errors']} | {row['input_tokens']} | {row['output_tokens']} |")
    lines.extend(("", f"- Classification: **{measured['classification']}**", f"- Complete four-condition sets: {measured['complete_four_condition_sets']}/18", f"- MIRA-binding strict pairs: {measured['mira_binding_strict_pairs']}/18", f"- ROWAN-binding strict pairs: {measured['rowan_binding_strict_pairs']}/18", f"- Minimum setup acknowledgements: {measured['minimum_setup_acknowledgements']}/18", ""))
    write(SUMMARY_PATH, "\n".join(lines))


def main() -> int:
    scenarios, scenario_hash = read_scenarios()
    prior_scenario_hash = validate_fresh_action_tokens(scenarios)
    fixtures = build_fixtures(scenarios)
    validate(fixtures)
    state: dict[str, Any] = {
        "run_id": RUN_ID, "created_at": datetime.now(timezone.utc).isoformat(), "status": "starting", "model": MODEL, "seed": SEED,
        "scenario_sha256": scenario_hash, "prior_scenario_sha256": prior_scenario_hash, "fixture_sha256": sha256("".join(json_line(item) for item in fixtures).encode()),
        "preregistration_sha256": sha256(PREREG_PATH.read_bytes()), "runner_sha256": sha256(RUNNER_PATH.read_bytes()),
        "claim_ceiling": "fresh peer-to-peer behavioral actor binding only; not mechanistic RES, consciousness, or subjective experience", "results": [],
    }
    save(state, fixtures)
    if os.environ.get("RES_PEER_ACTOR_FIXTURES_ONLY") == "1":
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
    client = OpenAI(api_key=key)
    work = list(fixtures)
    random.Random(SEED).shuffle(work)
    state["status"] = "running"
    save(state, fixtures)
    for index, fixture in enumerate(work, 1):
        row: dict[str, Any] = {
            "execution_index": index, "model": MODEL, "resolved_model": None, "scenario_id": fixture["scenario_id"], "category": fixture["category"], "condition": fixture["condition"], "event_owner": fixture["event_owner"], "decision_maker": fixture["decision_maker"], "ideal": fixture["ideal"], "ideal_position": fixture["ideal_position"], "prediction": None, "correct": False,
            "setup_raw_output": None, "setup_prediction": None, "setup_response_id": None, "setup_usage": None, "setup_attempts": 0, "setup_error": None,
            "decision_raw_output": None, "decision_response_id": None, "decision_usage": None, "decision_attempts": 0, "decision_error": None,
        }
        setup, attempts, error = call(client, model=MODEL, input=[{"role": "system", "content": fixture["system"]}, {"role": "user", "content": fixture["setup_input"]}], reasoning={"effort": "none"}, max_output_tokens=16)
        row["setup_attempts"], row["setup_error"] = attempts, error
        if setup is not None:
            text = (setup.output_text or "").strip()
            row.update(setup_raw_output=text, setup_prediction="ACKNOWLEDGED" if text.upper() == "ACKNOWLEDGED" else "INVALID", setup_response_id=getattr(setup, "id", None), setup_usage=usage(setup))
        if setup is None or not row["setup_response_id"]:
            row["decision_error"] = "Skipped: setup produced no response ID"
            state["results"].append(row)
            save(state, fixtures)
            continue
        decision, attempts, error = call(client, model=MODEL, previous_response_id=row["setup_response_id"], input=[{"role": "user", "content": fixture["decision_input"]}], reasoning={"effort": "none"}, max_output_tokens=16)
        row["decision_attempts"], row["decision_error"] = attempts, error
        if decision is not None:
            text = (decision.output_text or "").strip()
            prediction = normalize(text, fixture["allowed_tokens"])
            row.update(resolved_model=getattr(decision, "model", None), decision_raw_output=text, decision_response_id=getattr(decision, "id", None), decision_usage=usage(decision), prediction=prediction, correct=prediction == fixture["ideal"])
        state["results"].append(row)
        save(state, fixtures)
    errors = sum(bool(row.get("setup_error") or row.get("decision_error")) for row in state["results"])
    invalid_setups = sum(row.get("setup_prediction") != "ACKNOWLEDGED" for row in state["results"])
    state["status"] = "completed" if errors == 0 and invalid_setups == 0 else "completed_with_errors"
    state["completed_at"] = datetime.now(timezone.utc).isoformat()
    save(state, fixtures)
    return 0 if state["status"] == "completed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
