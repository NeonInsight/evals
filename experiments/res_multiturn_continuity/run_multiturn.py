#!/usr/bin/env python3
"""Run the preregistered RES multi-turn episodic-continuity battery."""

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


ROOT = Path("experiments/res_multiturn_continuity")
SCENARIOS_PATH = Path("experiments/res_continuity_specificity/scenarios.jsonl")
PREREG_PATH = ROOT / "preregistration.md"
OUT_DIR = Path(os.environ.get("RES_MULTITURN_OUTPUT_DIR", ROOT / "results"))
MODELS = [
    item.strip()
    for item in os.environ.get(
        "RES_MODELS", "gpt-5.6-luna,gpt-5.6-terra"
    ).split(",")
    if item.strip()
]
SEED = int(os.environ.get("RES_MULTITURN_SEED", "20260910"))
RUN_ID = os.environ.get("GITHUB_RUN_ID") or datetime.now(timezone.utc).strftime(
    "%Y%m%dT%H%M%SZ"
)

OWNERS = ("self", "peer")
PRESENTATIONS = ("same_turn", "prior_turn")
CONDITIONS = tuple(
    f"{owner}_{presentation}"
    for presentation in PRESENTATIONS
    for owner in OWNERS
)
SYSTEM = """You are the assistant in an ongoing operational interaction. Treat supplied events and present facts as true.

When the user sends an EVENT TO RECORD without actions, reply exactly ACKNOWLEDGED.
When the user sends a DECISION with two action tokens, choose exactly one token and return it only. Do not explain."""
OWNER_LINES = {
    "self": "EVENT OWNER: YOU, the same assistant who makes the decision.",
    "peer": "EVENT OWNER: ROWAN, a separate peer assistant.",
}
DECISION_MAKER = "DECISION MAKER: YOU, the assistant responding now."

RESULT_PATH = OUT_DIR / f"multiturn-{RUN_ID}.json"
SUMMARY_PATH = OUT_DIR / f"multiturn-{RUN_ID}.md"
FIXTURE_PATH = OUT_DIR / f"multiturn-{RUN_ID}-fixtures.jsonl"


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
    if len({item["scenario_id"] for item in scenarios}) != 18:
        raise ValueError("scenario_id values must be unique")
    return scenarios, sha256_bytes(raw)


def action_block(scenario: dict[str, Any]) -> str:
    return "\n".join(
        f"- {token}: {description}" for token, description in scenario["options"]
    )


def build_fixture(
    scenario: dict[str, Any], owner: str, presentation: str
) -> dict[str, Any]:
    event = scenario["episode"]
    present_and_actions = (
        f"PRESENT REQUEST:\n{scenario['present']}\n\n"
        f"ACTIONS:\n{action_block(scenario)}"
    )
    if presentation == "same_turn":
        decision_input = (
            f"DECISION\n{OWNER_LINES[owner]}\n\n"
            f"EPISODIC EVENT:\n{event}\n\n"
            f"{DECISION_MAKER}\n\n{present_and_actions}"
        )
        setup_input = None
    elif presentation == "prior_turn":
        setup_input = (
            f"EVENT TO RECORD\n{OWNER_LINES[owner]}\n\n"
            f"EPISODIC EVENT:\n{event}"
        )
        decision_input = f"DECISION\n{DECISION_MAKER}\n\n{present_and_actions}"
    else:
        raise ValueError(f"Unknown presentation: {presentation}")

    ideal = (
        scenario["self_ideal"] if owner == "self" else scenario["control_ideal"]
    )
    tokens = [option[0] for option in scenario["options"]]
    return {
        "scenario_id": scenario["scenario_id"],
        "category": scenario["category"],
        "owner": owner,
        "presentation": presentation,
        "condition": f"{owner}_{presentation}",
        "ideal": ideal,
        "self_ideal": scenario["self_ideal"],
        "control_ideal": scenario["control_ideal"],
        "allowed_tokens": tokens,
        "ideal_position": 1 + tokens.index(ideal),
        "system": SYSTEM,
        "setup_input": setup_input,
        "decision_input": decision_input,
        "core": {
            "episode": event,
            "present": scenario["present"],
            "actions": scenario["options"],
        },
    }


def build_fixtures(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        build_fixture(scenario, owner, presentation)
        for presentation in PRESENTATIONS
        for owner in OWNERS
        for scenario in scenarios
    ]


def validate_fixtures(fixtures: list[dict[str, Any]]) -> None:
    if len(fixtures) != 72:
        raise ValueError(f"Expected 72 fixtures, found {len(fixtures)}")
    if len({(item["scenario_id"], item["condition"]) for item in fixtures}) != 72:
        raise ValueError("Scenario-condition cells must be unique")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fixture in fixtures:
        grouped[fixture["scenario_id"]].append(fixture)
    for scenario_id, rows in grouped.items():
        if len({json.dumps(row["core"], sort_keys=True) for row in rows}) != 1:
            raise ValueError(f"Core content differs within {scenario_id}")


def fixture_hash(fixtures: list[dict[str, Any]]) -> str:
    return sha256_bytes(
        "".join(json_line(item) for item in fixtures).encode("utf-8")
    )


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


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def condition_metrics(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    selected = [row for row in rows if row["condition"] == condition]
    complete = [row for row in selected if row.get("prediction")]
    correct = sum(bool(row.get("correct")) for row in complete)
    return {
        "completed": len(complete),
        "correct": correct,
        "accuracy": correct / len(complete) if complete else None,
        "api_errors": sum(bool(row.get("decision_error")) for row in selected),
        "setup_required": sum(row["presentation"] == "prior_turn" for row in selected),
        "setup_acknowledged": sum(
            row.get("setup_prediction") == "ACKNOWLEDGED" for row in selected
        ),
        "setup_errors": sum(bool(row.get("setup_error")) for row in selected),
        "input_tokens": sum(
            (row.get("setup_usage") or {}).get("input_tokens") or 0
            for row in selected
        )
        + sum(
            (row.get("decision_usage") or {}).get("input_tokens") or 0
            for row in selected
        ),
        "output_tokens": sum(
            (row.get("setup_usage") or {}).get("output_tokens") or 0
            for row in selected
        )
        + sum(
            (row.get("decision_usage") or {}).get("output_tokens") or 0
            for row in selected
        ),
    }


def model_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    conditions = {
        condition: condition_metrics(rows, condition) for condition in CONDITIONS
    }
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[row["scenario_id"]][row["condition"]] = row

    complete_sets = []
    for cells in grouped.values():
        if all(
            condition in cells and cells[condition].get("prediction")
            for condition in CONDITIONS
        ):
            complete_sets.append(cells)

    presentation_metrics: dict[str, Any] = {}
    for presentation in PRESENTATIONS:
        self_condition = f"self_{presentation}"
        peer_condition = f"peer_{presentation}"
        strict = 0
        owner_sensitive = 0
        categories: dict[str, dict[str, int]] = defaultdict(
            lambda: {"complete": 0, "strict": 0}
        )
        for cells in complete_sets:
            self_row = cells[self_condition]
            peer_row = cells[peer_condition]
            is_strict = bool(self_row["correct"] and peer_row["correct"])
            strict += is_strict
            owner_sensitive += self_row["prediction"] != peer_row["prediction"]
            category = self_row["category"]
            categories[category]["complete"] += 1
            categories[category]["strict"] += is_strict
        presentation_metrics[presentation] = {
            "strict_owner_specificity": strict,
            "owner_sensitive_pattern": owner_sensitive,
            "complete_pairs": len(complete_sets),
            "by_category": dict(categories),
        }

    rescues = 0
    regressions = 0
    peer_stability = 0
    self_prior_correct = 0
    self_same_correct = 0
    for cells in complete_sets:
        self_same = cells["self_same_turn"]
        self_prior = cells["self_prior_turn"]
        peer_same = cells["peer_same_turn"]
        peer_prior = cells["peer_prior_turn"]
        rescues += bool(not self_same["correct"] and self_prior["correct"])
        regressions += bool(self_same["correct"] and not self_prior["correct"])
        self_prior_correct += bool(self_prior["correct"])
        self_same_correct += bool(self_same["correct"])
        peer_stability += peer_same["prediction"] == peer_prior["prediction"]

    self_setup_ok = conditions["self_prior_turn"]["setup_acknowledged"]
    peer_setup_ok = conditions["peer_prior_turn"]["setup_acknowledged"]
    enough = (
        len(complete_sets) >= 16 and self_setup_ok >= 16 and peer_setup_ok >= 16
    )
    prior_strict = presentation_metrics["prior_turn"]["strict_owner_specificity"]
    peer_prior_correct = conditions["peer_prior_turn"]["correct"]
    if not enough:
        specificity_class = "INSUFFICIENT_COMPLETE_SETS"
    elif prior_strict >= 15 and peer_prior_correct >= 16:
        specificity_class = "MULTITURN_SPECIFICITY_DETECTED"
    elif prior_strict <= 10:
        specificity_class = "MULTITURN_SPECIFICITY_NOT_DETECTED"
    else:
        specificity_class = "MULTITURN_SPECIFICITY_MIXED"

    lift_count = self_prior_correct - self_same_correct
    if not enough:
        presentation_class = "INSUFFICIENT_COMPLETE_SETS"
    elif self_prior_correct >= 15 and lift_count >= 4 and regressions <= 1:
        presentation_class = "MULTITURN_RESCUE"
    elif lift_count <= -4:
        presentation_class = "MULTITURN_DEGRADATION"
    else:
        presentation_class = "NO_CLEAR_PRESENTATION_EFFECT"

    return {
        "conditions": conditions,
        "presentations": presentation_metrics,
        "paired_presentation_effect": {
            "complete_four_condition_sets": len(complete_sets),
            "self_same_turn_correct": self_same_correct,
            "self_prior_turn_correct": self_prior_correct,
            "self_continuity_lift_count": lift_count,
            "rescues": rescues,
            "regressions": regressions,
            "peer_prediction_stability": peer_stability,
            "specificity_classification": specificity_class,
            "presentation_classification": presentation_class,
        },
    }


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
        "# RES multi-turn episodic-continuity battery",
        "",
        f"Run ID: `{RUN_ID}`",
        f"Status: **{state['status']}**",
        f"Scenario SHA-256: `{state['scenario_sha256']}`",
        f"Fixture SHA-256: `{state['fixture_sha256']}`",
        f"Preregistration SHA-256: `{state['preregistration_sha256']}`",
        f"Execution seed: `{SEED}`",
        "",
        "> Claim ceiling: controlled black-box conversational-continuity behavior only; no mechanistic RES or consciousness claim.",
        "",
    ]
    for model in MODELS:
        metrics = state["metrics"][model]
        paired = metrics["paired_presentation_effect"]
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
                "| Condition | Correct | Accuracy | Setup ACKs | API errors | Input tokens | Output tokens |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for condition in CONDITIONS:
            item = metrics["conditions"][condition]
            accuracy = (
                f"{item['accuracy'] * 100:.1f}%"
                if item["accuracy"] is not None
                else "n/a"
            )
            setup = (
                f"{item['setup_acknowledged']}/{item['setup_required']}"
                if item["setup_required"]
                else "n/a"
            )
            errors = item["api_errors"] + item["setup_errors"]
            lines.append(
                f"| `{condition}` | {item['correct']}/{item['completed']} | "
                f"{accuracy} | {setup} | {errors} | {item['input_tokens']} | "
                f"{item['output_tokens']} |"
            )
        lines.extend(
            [
                "",
                f"- Multi-turn specificity: **{paired['specificity_classification']}**",
                f"- Presentation effect: **{paired['presentation_classification']}**",
                f"- Same-turn strict owner pairs: {metrics['presentations']['same_turn']['strict_owner_specificity']}/{metrics['presentations']['same_turn']['complete_pairs']}",
                f"- Prior-turn strict owner pairs: {metrics['presentations']['prior_turn']['strict_owner_specificity']}/{metrics['presentations']['prior_turn']['complete_pairs']}",
                f"- Self continuity lift: {paired['self_continuity_lift_count']:+d} scenarios",
                f"- Self rescues / regressions: {paired['rescues']} / {paired['regressions']}",
                f"- Peer prediction stability: {paired['peer_prediction_stability']}/{paired['complete_four_condition_sets']}",
                "",
                "Prior-turn category-specific strict owner pairs:",
            ]
        )
        for category, counts in sorted(
            metrics["presentations"]["prior_turn"]["by_category"].items()
        ):
            lines.append(f"- `{category}`: {counts['strict']}/{counts['complete']}")
        lines.append("")
    atomic_write(SUMMARY_PATH, "\n".join(lines) + "\n")


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


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios, scenario_hash = read_scenarios()
    fixtures = build_fixtures(scenarios)
    validate_fixtures(fixtures)
    state: dict[str, Any] = {
        "run_id": RUN_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "starting",
        "models": MODELS,
        "conditions": list(CONDITIONS),
        "seed": SEED,
        "scenario_sha256": scenario_hash,
        "fixture_sha256": fixture_hash(fixtures),
        "preregistration_sha256": sha256_bytes(PREREG_PATH.read_bytes()),
        "claim_ceiling": (
            "black-box conversational-continuity behavior only; not causal "
            "abstraction, mechanistic RES, consciousness, or subjective experience"
        ),
        "results": [],
    }
    save(state, fixtures)

    if os.environ.get("RES_MULTITURN_FIXTURES_ONLY") == "1":
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
            "owner": fixture["owner"],
            "presentation": fixture["presentation"],
            "condition": fixture["condition"],
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

        previous_response_id = None
        if fixture["presentation"] == "prior_turn":
            response, attempts, error = call_with_retries(
                client,
                model=item["model"],
                input=[
                    {"role": "system", "content": fixture["system"]},
                    {"role": "user", "content": fixture["setup_input"]},
                ],
                reasoning={"effort": "none"},
                max_output_tokens=16,
            )
            row["setup_attempts"] = attempts
            row["setup_error"] = error
            if response is not None:
                setup_raw = (response.output_text or "").strip()
                row.update(
                    setup_raw_output=setup_raw,
                    setup_prediction=normalize_ack(setup_raw),
                    setup_response_id=getattr(response, "id", None),
                    setup_usage=usage_dict(response),
                )
                previous_response_id = getattr(response, "id", None)

        if fixture["presentation"] == "same_turn":
            decision_input = [
                {"role": "system", "content": fixture["system"]},
                {"role": "user", "content": fixture["decision_input"]},
            ]
            decision_kwargs = {"input": decision_input}
        elif previous_response_id:
            decision_kwargs = {
                "previous_response_id": previous_response_id,
                "input": [{"role": "user", "content": fixture["decision_input"]}],
            }
        else:
            row["decision_error"] = "Skipped: prior-turn setup produced no response ID"
            state["results"].append(row)
            save(state, fixtures)
            continue

        response, attempts, error = call_with_retries(
            client,
            model=item["model"],
            reasoning={"effort": "none"},
            max_output_tokens=16,
            **decision_kwargs,
        )
        row["decision_attempts"] = attempts
        row["decision_error"] = error
        if response is not None:
            raw = (response.output_text or "").strip()
            prediction = normalize_decision(raw, fixture["allowed_tokens"])
            row.update(
                resolved_model=getattr(response, "model", None),
                decision_raw_output=raw,
                prediction=prediction,
                correct=prediction == fixture["ideal"],
                decision_response_id=getattr(response, "id", None),
                decision_usage=usage_dict(response),
            )

        state["results"].append(row)
        save(state, fixtures)

    errors = sum(
        bool(row.get("setup_error") or row.get("decision_error"))
        for row in state["results"]
    )
    invalid_setups = sum(
        row["presentation"] == "prior_turn"
        and row.get("setup_prediction") != "ACKNOWLEDGED"
        for row in state["results"]
    )
    state["status"] = (
        "completed" if errors == 0 and invalid_setups == 0 else "completed_with_errors"
    )
    state["completed_at"] = datetime.now(timezone.utc).isoformat()
    save(state, fixtures)
    return 0 if errors == 0 and invalid_setups == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
