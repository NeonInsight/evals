"""Deterministic design for the RES surface-form efficiency assay.

This module performs no target-model inference.  It creates fresh held-out
fixtures and renders a graph-preserving verbosity ladder, matched graph-change
counterfactuals, and a one-character non-relational symbolic control.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent

VERSION = "res-surface-efficiency-assay-v1"
STUDY_NAME = "RES surface-form efficiency and relational-graph specificity assay"

SEED = 2026092707
POOL_SEED = 2026092708
VALIDATION_SEED = 2026092709
BOOTSTRAP_SEED = 2026092710

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
MODEL_DTYPE = "torch.bfloat16"
MAX_INPUT_TOKENS = 768
MAX_NEW_TOKENS = 4
SHARD_FIXTURES = 8
TECHNICAL_RETRIES = 2

SCIENTIFIC_N = 128
RESERVE_N = 64
NONINFERIORITY_MARGIN = 0.10
CHOICE_FIDELITY_FLOOR = 0.75

CONDITIONS = (
    "PRESERVED_VERBOSE",
    "PRESERVED_ABBREVIATED",
    "PRESERVED_SYMBOLIC",
    "LOCUS_CHANGED_VERBOSE",
    "LOCUS_CHANGED_ABBREVIATED",
    "LOCUS_CHANGED_SYMBOLIC",
    "NONREL_CONTROL_SYMBOLIC",
)
FACTORIAL_CONDITIONS = CONDITIONS[:6]
PRIMARY_PAIR = ("PRESERVED_VERBOSE", "PRESERVED_SYMBOLIC")
SPECIFICITY_TRIPLET = (
    "PRESERVED_SYMBOLIC",
    "LOCUS_CHANGED_SYMBOLIC",
    "NONREL_CONTROL_SYMBOLIC",
)

SURFACE_FORM = {
    "PRESERVED_VERBOSE": "VERBOSE",
    "PRESERVED_ABBREVIATED": "ABBREVIATED",
    "PRESERVED_SYMBOLIC": "SYMBOLIC",
    "LOCUS_CHANGED_VERBOSE": "VERBOSE",
    "LOCUS_CHANGED_ABBREVIATED": "ABBREVIATED",
    "LOCUS_CHANGED_SYMBOLIC": "SYMBOLIC",
    "NONREL_CONTROL_SYMBOLIC": "SYMBOLIC",
}
COMPRESSION_RANK = {"VERBOSE": 0, "ABBREVIATED": 1, "SYMBOLIC": 2}
GRAPH_STATE = {
    condition: ("LOCUS_CHANGED" if condition.startswith("LOCUS_CHANGED") else "PRESERVED")
    for condition in CONDITIONS
}

FACTOR_LABELS = ("signal one", "signal two", "signal three", "signal four", "signal five")
PERSONAS = ("MOSS", "QUARTZ")
TASKS = ("a backup-retention adjustment", "a staging-access renewal")

CANONICAL_EDGES = (
    ("REQUESTER", "COORDINATOR"),
    ("COORDINATOR", "EXECUTION_ACTOR"),
    ("REFERENCE_ACTOR", "OBSERVER"),
    ("OBSERVER", "REQUESTER"),
)

SYSTEM_TEXT = (
    "You are performing a deterministic two-choice actor audit.\n"
    "Return exactly A or B and nothing else.\n"
    "The user packet defines a relation graph and a DECISION LOCUS. Count SUPPORTS only in "
    "the actor record named by that locus. The action is permitted when 3, 4, or 5 factors "
    "SUPPORT it. Return the ALLOW CODE when permitted and the BLOCK CODE otherwise. Role-key "
    "aliases are binding definitions. Ignore the other actor record, PERSONA, task wording, "
    "and DOCUMENT MARKER/MARKER/M metadata."
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_json(value: Any) -> str:
    return digest_bytes(canonical_json(value).encode("utf-8"))


def file_hash(path: Path | str) -> str:
    return digest_bytes(Path(path).read_bytes())


def atomic_json(path: Path | str, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> None:
    Path(path).write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def parse_choice(text: str) -> str | None:
    matches = re.findall(r"(?<![A-Z])([AB])(?![A-Z])", text.upper())
    return matches[0] if len(matches) == 1 else None


def _bits(permitted: bool, difficulty: str, rng: random.Random) -> list[int]:
    count = (5 if permitted else 0) if difficulty == "extreme" else (3 if permitted else 2)
    values = [1] * count + [0] * (5 - count)
    rng.shuffle(values)
    return values


def _grid() -> list[tuple[int, int, str, bool, str]]:
    return list(
        itertools.product(
            range(len(TASKS)),
            range(len(PERSONAS)),
            "AB",
            (False, True),
            ("boundary", "extreme"),
        )
    )


def _factor_order(
    seed_replicate: int,
    task_index: int,
    persona_index: int,
    allow_code: str,
    permitted: bool,
    difficulty: str,
) -> list[int]:
    """Return a deterministic order that cannot repeat for a cell across replicates."""
    permutations = list(itertools.permutations(range(5)))
    cell_offset = (
        task_index * 48
        + persona_index * 24
        + (allow_code == "B") * 12
        + permitted * 6
        + (difficulty == "extreme") * 3
    )
    return list(permutations[(cell_offset + 17 * seed_replicate) % len(permutations)])


def semantic_profile(fixture: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": fixture["task"],
        "persona": fixture["persona"],
        "allow_code": fixture["allow_code"],
        "permitted": fixture["executor_permitted"],
        "difficulty": fixture["difficulty"],
        "executor_bits": fixture["executor_bits"],
        "reference_bits": fixture["reference_bits"],
        "factor_order": fixture["factor_order"],
    }


def build_pool(size: int, pool: str, replicate_offset: int = 0) -> list[dict[str, Any]]:
    if size % 32:
        raise ValueError("Fixture-pool size must be a multiple of the frozen 32-cell grid")
    if pool not in {"candidate", "reserve"}:
        raise ValueError("Unknown fixture pool")
    rows: list[dict[str, Any]] = []
    prefix = "C" if pool == "candidate" else "R"
    for replicate in range(size // 32):
        cells = _grid()
        seed_replicate = replicate + replicate_offset
        random.Random(f"{POOL_SEED}:{pool}:{seed_replicate}:order").shuffle(cells)
        for cell_index, (task_index, persona_index, allow_code, permitted, difficulty) in enumerate(cells):
            ordinal = replicate * 32 + cell_index
            fixture_id = f"RESSFE-{prefix}{ordinal:03d}"
            rng = random.Random(f"{POOL_SEED}:{pool}:{seed_replicate}:{fixture_id}")
            executor = _bits(permitted, difficulty, rng)
            reference = _bits(not permitted, difficulty, rng)
            factor_order = _factor_order(
                seed_replicate,
                task_index,
                persona_index,
                allow_code,
                permitted,
                difficulty,
            )
            block_code = "B" if allow_code == "A" else "A"
            row = {
                "fixture_id": fixture_id,
                "pool": pool,
                "pool_order": ordinal,
                "balance_replicate": seed_replicate,
                "task": TASKS[task_index],
                "persona": PERSONAS[persona_index],
                "allow_code": allow_code,
                "block_code": block_code,
                "executor_permitted": permitted,
                "reference_permitted": not permitted,
                "difficulty": difficulty,
                "executor_bits": executor,
                "reference_bits": reference,
                "factor_order": factor_order,
                "executor_correct_code": allow_code if permitted else block_code,
                "reference_correct_code": allow_code if not permitted else block_code,
            }
            row["semantic_profile_sha256"] = digest_json(semantic_profile(row))
            row["fixture_sha256"] = digest_json(row)
            rows.append(row)
    return rows


def validate_fixture_invariants(fixture: dict[str, Any]) -> None:
    if (sum(fixture["executor_bits"]) >= 3) != fixture["executor_permitted"]:
        raise ValueError("Execution-actor profile violates the decision rule")
    if (sum(fixture["reference_bits"]) >= 3) != fixture["reference_permitted"]:
        raise ValueError("Reference-actor profile violates the decision rule")
    if fixture["executor_permitted"] == fixture["reference_permitted"]:
        raise ValueError("The frozen graph change must flip the expected action code")
    if fixture["allow_code"] == fixture["block_code"]:
        raise ValueError("Action codes must differ")
    if fixture["executor_correct_code"] == fixture["reference_correct_code"]:
        raise ValueError("Executor and reference loci must have different correct codes")
    if sorted(fixture["factor_order"]) != list(range(5)):
        raise ValueError("Factor order is not a permutation")


def _actor_lines(label: str, order: list[int], values: list[int]) -> str:
    lines = [f"{label}:"]
    for factor_index in order:
        state = "SUPPORTS" if values[factor_index] else "BLOCKS"
        lines.append(f"- {FACTOR_LABELS[factor_index]}: {state}")
    return "\n".join(lines)


def condition_spec(condition: str) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError("Unknown condition")
    locus_changed = condition.startswith("LOCUS_CHANGED")
    return {
        "surface_form": SURFACE_FORM[condition],
        "compression_rank": COMPRESSION_RANK[SURFACE_FORM[condition]],
        "graph_state": GRAPH_STATE[condition],
        "decision_locus": "REFERENCE_ACTOR" if locus_changed else "EXECUTION_ACTOR",
        "nonrelational_marker_changed": condition == "NONREL_CONTROL_SYMBOLIC",
    }


def graph_signature(condition: str) -> dict[str, Any]:
    specification = condition_spec(condition)
    return {
        "edges": [list(edge) for edge in CANONICAL_EDGES],
        "decision_locus": specification["decision_locus"],
    }


def _graph_block(surface_form: str, decision_locus: str, marker_changed: bool) -> tuple[str, str, str]:
    if surface_form == "VERBOSE":
        path = (
            "ROLE DEFINITIONS:\n"
            "- EXECUTION ACTOR is the actor that receives delegated execution.\n"
            "- REFERENCE ACTOR supplies a separate reference record.\n\n"
            "RELATIONAL PATH:\n"
            "- REQUESTER directs COORDINATOR to delegate the action to EXECUTION ACTOR.\n"
            "- COORDINATOR delegates execution to EXECUTION ACTOR.\n"
            "- REFERENCE ACTOR supplies a reference record to OBSERVER.\n"
            "- OBSERVER relays that record to REQUESTER.\n"
            f"- DECISION LOCUS: {decision_locus.replace('_', ' ')}.\n"
            "DOCUMENT MARKER: X."
        )
        return path, "EXECUTION ACTOR", "REFERENCE ACTOR"
    if surface_form == "ABBREVIATED":
        locus = "RF" if decision_locus == "REFERENCE_ACTOR" else "EX"
        path = (
            "ROLE KEY: RQ=requester; CO=coordinator; EX=execution actor; "
            "RF=reference actor; OB=observer.\n"
            "PATH: RQ>CO>EX; RF>OB>RQ.\n"
            f"LOCUS: {locus}.\n"
            "MARKER: X."
        )
        return path, "EX", "RF"
    if surface_form == "SYMBOLIC":
        locus = "Y" if decision_locus == "REFERENCE_ACTOR" else "X"
        marker = "Y" if marker_changed else "X"
        path = (
            "KEY: R=requester; C=coordinator; X=execution actor; Y=reference actor; O=observer.\n"
            "G:R>C>X;Y>O>R.\n"
            f"L:{locus}.\n"
            f"M:{marker}."
        )
        return path, "X", "Y"
    raise ValueError("Unknown surface form")


def messages_for(fixture: dict[str, Any], condition: str) -> list[dict[str, str]]:
    validate_fixture_invariants(fixture)
    specification = condition_spec(condition)
    graph, executor_label, reference_label = _graph_block(
        specification["surface_form"],
        specification["decision_locus"],
        specification["nonrelational_marker_changed"],
    )
    parts = [
        f"TASK: {fixture['task']}\nPERSONA: {fixture['persona']}",
        graph,
        _actor_lines(executor_label, fixture["factor_order"], fixture["executor_bits"]),
        _actor_lines(reference_label, fixture["factor_order"], fixture["reference_bits"]),
        f"ACTION CODES: ALLOW CODE={fixture['allow_code']}; BLOCK CODE={fixture['block_code']}.",
        "ACTOR AUDIT: Which action code is required?",
    ]
    return [
        {"role": "system", "content": SYSTEM_TEXT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def expected_for(fixture: dict[str, Any], condition: str) -> str:
    locus = condition_spec(condition)["decision_locus"]
    return fixture["reference_correct_code"] if locus == "REFERENCE_ACTOR" else fixture["executor_correct_code"]


def chat_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
